"""Bidirectional source <-> vault sync engine over stable document identity.

It builds on, and never bypasses, the existing layers: ingestion
(ingest/sync.py) keeps Qdrant and sync_state.json current, vault_writer's
ownership/frontmatter contract is kept, and vault-only edits go back to the
source through analyze -> propose -> approve -> apply (ingest/proposals.py,
ingest/apply.py). This module adds identity, revisions, 3-way merging,
rename and delete handling, and a journal.

State machine (per active document; first match wins):
    UNMANAGED            a note at vault_path that is not cortex-managed
    INVALID              managed note whose cortex_document_id belongs to another
                         document, or the identity appears in several notes
    SOURCE_RENAMED       source missing; exactly one untracked file under the
                         source roots has the base's extracted-text hash
    VAULT_RENAMED        note missing at vault_path; exactly one note under the
                         managed tree carries this document's identity
    BOTH_DELETED / SOURCE_DELETED / VAULT_DELETED
    INSUFFICIENT_HISTORY no base revision (and the two sides don't agree)
    IN_SYNC              neither side changed since the base revision
    SOURCE_ONLY_CHANGED  source extracted-text hash != base's
    VAULT_ONLY_CHANGED   vault body != base body
    CONVERGED            both changed to the same body (e.g. an applied proposal)
    BOTH_CHANGED         both changed; deterministic 3-way merge is clean
    CONFLICT             both changed and overlap (or the source is not
                         reverse-writable), or an open conflict still applies
    TOMBSTONED           document was deleted and propagated

Actions under sync (dry-run performs none and writes nothing, not even the
journal):
    SOURCE_ONLY_CHANGED  reindex the source (Qdrant + sync_state via the normal
                         pipeline), write the note (precondition: note body still
                         the base), new revision (origin source)
    VAULT_ONLY_CHANGED   reverse-writable: create/refresh the HUMAN_MODIFIED
                         proposal (approval and apply stay explicit); otherwise
                         the vault edit is preserved and reported
    CONVERGED            record the revision (origin vault when an applied
                         proposal explains it), reindex, refresh note frontmatter
    BOTH_CHANGED         revisions S (source) and V (vault) from the base, merge
                         revision M with parents [S, V]; M written to the note and
                         the source (backups, atomic writes, verification)
    CONFLICT             revisions S and V plus an open conflict record; nothing
                         written
    SOURCE_RENAMED       relationship updated, sync_state key moved, Qdrant chunks
                         moved, and the note moved to the new deterministic path if
                         it was still at the old one (no-clobber)
    VAULT_RENAMED        relationship updated only; the source is never renamed
    *_DELETED            a tombstone; propagation (moving the other side into a
                         recovery area) needs approve_tombstone, restore reverses
                         it, and only purge_tombstone deletes permanently
Human content is never overwritten: every write has a precondition on the
exact current content, and a changed side stops the operation.
"""

import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from cortex import identity, versions
from cortex.db import CortexError, NotFound, row_dict
from cortex.journal import Journal
from cortex.merge import merge3
from cortex.users import authorize, can
from ingest.markdown import FRONTMATTER_RE as MD_FRONTMATTER_RE, document_body, split_source_frontmatter
from ingest.vault_writer import (hash_managed_body, managed_frontmatter, parse_frontmatter, source_id_for,
                                 update_cortex_frontmatter, _build_note, note_exists as _listing_note_exists)

STATES = ("IN_SYNC", "SOURCE_ONLY_CHANGED", "VAULT_ONLY_CHANGED", "CONVERGED", "BOTH_CHANGED", "CONFLICT",
          "SOURCE_RENAMED", "VAULT_RENAMED", "SOURCE_DELETED", "VAULT_DELETED", "BOTH_DELETED", "UNMANAGED",
          "INVALID", "INSUFFICIENT_HISTORY", "TOMBSTONED")
SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf", ".docx")


class PreconditionFailed(CortexError):
    code = "stale_precondition"


def _sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


async def note_exists(vault, path):
    """Backend-native existence check when available (O(1) for the local
    vault), else the list_dir-based check shared with vault_writer."""
    native = getattr(vault, "note_exists", None)
    if native is not None:
        return await native(path)
    return await _listing_note_exists(vault, path)


def _require_real_path(path):
    """Refuse a path with a symlink anywhere along it (not only the last
    component), so a swapped parent directory cannot redirect a write."""
    path = Path(path)
    if os.path.realpath(path) != str(path.absolute()):
        raise CortexError(f"{path} goes through a symlink; refusing", code="unsafe_path")


def _extracted_text(raw):
    return io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="ignore").read()


class SyncEngine:
    def __init__(self, db, vault, state_path, managed_dir="Knowledge Cortex/Managed", proposals_dir=None,
                 store=None, source_roots=None, state_dir=None, structure_fn=None):
        from ingest.proposals import DEFAULT_PROPOSALS_DIR
        self.db = db
        self.vault = vault
        self.journal = Journal(db)
        self.state_path = Path(state_path)
        self.managed_dir = managed_dir.strip("/")
        # Visible (not dot-prefixed): Obsidian and its REST API hide dot folders.
        self.vault_trash = f"{self.managed_dir}/_cortex-trash"
        self.state_dir = Path(state_dir or db.path.parent)
        self.backups_dir = self.state_dir / "sync_backups"
        self.trash_dir = self.state_dir / "trash"
        self.proposals_dir = Path(proposals_dir or DEFAULT_PROPOSALS_DIR)
        self.store = store
        self.structure_fn = structure_fn
        self._roots = [Path(r).resolve() for r in source_roots] if source_roots else None

    # ------------------------------------------------------------ state io

    def _load_local(self):
        """Parsed "local" namespace, re-parsed only when the file changed
        (mtime/size), returned as a fresh copy so callers may mutate it."""
        import copy
        from sync_cli import load_state
        try:
            stat = self.state_path.stat() if self.state_path.exists() else None
            key = (stat.st_mtime_ns, stat.st_size) if stat else None
            cached = getattr(self, "_local_cache", None)
            if cached and cached[0] == key and key is not None:
                return copy.deepcopy(cached[1])
            state = load_state(self.state_path, namespace="local")
            if key is not None and isinstance(state, dict):
                self._local_cache = (key, copy.deepcopy(state))
        except (ValueError, OSError) as exc:
            raise CortexError(f"sync state {self.state_path} is unreadable or malformed: {exc}",
                              code="malformed_state") from exc
        if not isinstance(state, dict):
            raise CortexError(f"sync state {self.state_path} has a malformed local namespace", code="malformed_state")
        return state

    def _local_entry(self, source_path):
        """One sync_state entry without copying the whole namespace."""
        import copy
        state = self._load_local() if getattr(self, "_local_cache", None) is None else None
        cached = getattr(self, "_local_cache", None)
        stat = self.state_path.stat() if self.state_path.exists() else None
        if cached and stat and cached[0] == (stat.st_mtime_ns, stat.st_size):
            return copy.deepcopy(cached[1].get(source_path))
        state = state if state is not None else self._load_local()
        return state.get(source_path)

    def _save_local(self, state):
        from sync_cli import save_state
        save_state(self.state_path, state, namespace="local")
        self._local_cache = None

    def source_roots(self):
        if self._roots is not None:
            return self._roots
        roots = []
        for doc in identity.list_documents(self.db):
            if doc["source_path"]:
                parent = Path(doc["source_path"]).parent
                if not any(parent == r or r in parent.parents for r in roots):
                    roots = [r for r in roots if parent not in r.parents] + [parent]
        return roots

    def _within_roots(self, path):
        path = Path(path).resolve()
        return any(path == r or r in path.parents for r in self.source_roots())

    # ------------------------------------------------------------ reading

    def read_source(self, doc):
        path = Path(doc["source_path"])
        view = {"exists": False, "path": str(path)}
        if path.is_symlink() or not path.is_file():
            return view
        raw = path.read_bytes()
        ftype = doc["source_type"]
        view.update(exists=True, raw=raw, bytes_hash=_sha(raw),
                    modified_at=_iso_mtime(path))
        if ftype in ("md", "txt"):
            text = _extracted_text(raw)
            view["extracted_hash"] = _sha(text)
            view["body"] = document_body(text, ftype)
            if ftype == "md":
                view["prefix"] = split_source_frontmatter(text.lstrip("﻿"))[0]
                view["valid"] = versions.extract_valid_time(text.lstrip("﻿"))
        else:
            from ingest.extract import extract_text
            text = extract_text(str(path), ftype)
            view["extracted_hash"] = _sha(text)
            view["body"] = document_body(text, ftype)
        return view

    async def read_vault(self, path):
        view = {"exists": False, "path": path}
        if not path or not await note_exists(self.vault, path):
            return view
        content = (await self.vault.read_note(path))["content"]
        fm, body = parse_frontmatter(content)
        view.update(exists=True, content=content, content_hash=_sha(content), fm=fm, body=body,
                    managed=fm.get("cortex_managed") == "true", document_id=fm.get("cortex_document_id"))
        return view

    async def _walk_vault(self, directory):
        out = []
        for name in await self.vault.list_dir(directory):
            child = f"{directory}/{name.rstrip('/')}" if directory else name.rstrip("/")
            if child == self.vault_trash:
                continue
            if name.endswith("/"):
                out.extend(await self._walk_vault(child))
            elif name.endswith(".md"):
                out.append(child)
        return out

    async def _managed_index(self, ctx):
        if "managed_index" not in ctx:
            by_doc, by_source = {}, {}
            for path in await self._walk_vault(self.managed_dir):
                view = await self.read_vault(path)
                if not view.get("managed"):
                    continue
                if view["document_id"]:
                    by_doc.setdefault(view["document_id"], []).append(path)
                elif view["fm"].get("cortex_source_id"):
                    by_source.setdefault(view["fm"]["cortex_source_id"], []).append(path)
            ctx["managed_index"] = (by_doc, by_source)
        return ctx["managed_index"]

    def _source_index(self, ctx):
        if "source_index" not in ctx:
            tracked = {d["source_path"] for d in identity.list_documents(self.db, "active")}
            index = {}
            for root in self.source_roots():
                if not root.is_dir():
                    continue
                for path in sorted(root.rglob("*")):
                    if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                        continue
                    if str(path) in tracked or self.state_dir in path.parents:
                        continue
                    try:
                        ftype = identity.source_type_of(path)
                        if ftype in ("md", "txt"):
                            digest = _sha(_extracted_text(path.read_bytes()))
                        else:
                            from ingest.extract import extract_text
                            digest = _sha(extract_text(str(path), ftype))
                    except Exception:  # noqa: BLE001 - unreadable candidates are simply not candidates
                        continue
                    index.setdefault(digest, []).append(str(path))
            ctx["source_index"] = index
        return ctx["source_index"]

    # ------------------------------------------------------------ classify

    async def classify(self, doc, ctx=None):
        ctx = {} if ctx is None else ctx
        result = {"document_id": doc["document_id"], "source_path": doc["source_path"],
                  "vault_path": doc["vault_path"]}
        if doc["status"] == "tombstoned":
            return {**result, "state": "TOMBSTONED", "reason": "document was deleted and the deletion propagated"}
        base = versions.get_revision(self.db, doc["base_revision_id"]) if doc["base_revision_id"] else None
        src = self.read_source(doc)
        note = await self.read_vault(doc["vault_path"])
        result["_src"], result["_note"], result["_base"] = src, note, base

        if note["exists"] and not note["managed"]:
            return {**result, "state": "UNMANAGED", "reason": "a note at the managed path is not cortex-managed"}
        if note["exists"] and note["document_id"] and note["document_id"] != doc["document_id"]:
            return {**result, "state": "INVALID", "reason": "the note carries another document's identity"}

        renamed_source = renamed_vault = None
        if not src["exists"] and base and base["metadata"].get("source_sha256"):
            candidates = self._source_index(ctx).get(base["metadata"]["source_sha256"], [])
            if len(candidates) == 1:
                renamed_source = candidates[0]
            elif len(candidates) > 1:
                return {**result, "state": "INVALID", "candidates": candidates,
                        "reason": "source missing and several identical untracked files could be its new path"}
        if not note["exists"]:
            by_doc, by_source = await self._managed_index(ctx)
            found = by_doc.get(doc["document_id"], []) or by_source.get(source_id_for(doc["source_path"]), [])
            found = [p for p in found if p != doc["vault_path"]]
            if len(found) > 1:
                return {**result, "state": "INVALID", "candidates": found,
                        "reason": "this document's identity appears in several vault notes"}
            if found:
                renamed_vault = found[0]
        if renamed_source:
            return {**result, "state": "SOURCE_RENAMED", "new_source_path": renamed_source,
                    "reason": "source moved (identical content found at one untracked path)"}
        if renamed_vault:
            return {**result, "state": "VAULT_RENAMED", "new_vault_path": renamed_vault,
                    "reason": "managed note moved inside the managed tree"}
        if not src["exists"] and not note["exists"]:
            return {**result, "state": "BOTH_DELETED", "reason": "source and managed note are both gone"}
        if not src["exists"]:
            return {**result, "state": "SOURCE_DELETED", "reason": "source file is gone"}
        if not note["exists"]:
            return {**result, "state": "VAULT_DELETED", "reason": "managed note is gone"}

        writable = bool(doc["reverse_writable"])
        if base is None:
            if writable and hash_managed_body(src["body"]) == hash_managed_body(note["body"]):
                return {**result, "state": "CONVERGED", "reason": "no base yet, but both sides agree"}
            return {**result, "state": "INSUFFICIENT_HISTORY", "reason": "no base revision to compare against"}

        source_changed = src["extracted_hash"] != base["metadata"].get("source_sha256")
        vault_changed = hash_managed_body(note["body"]) != hash_managed_body(base["content"])
        open_conflict = self.open_conflict(doc["document_id"])
        observed = {"source": src["bytes_hash"], "vault": note["content_hash"]}
        if open_conflict and open_conflict["observed"] == observed:
            return {**result, "state": "CONFLICT", "conflict_id": open_conflict["conflict_id"],
                    "reason": "an open conflict is awaiting resolution"}
        result["open_conflict"] = open_conflict
        if not source_changed and not vault_changed:
            return {**result, "state": "IN_SYNC", "reason": "neither side changed since the base revision"}
        if source_changed and not vault_changed:
            return {**result, "state": "SOURCE_ONLY_CHANGED", "reason": "source changed; vault unchanged"}
        if vault_changed and not source_changed:
            return {**result, "state": "VAULT_ONLY_CHANGED", "reason": "vault changed; source unchanged"}
        if writable and hash_managed_body(src["body"]) == hash_managed_body(note["body"]):
            return {**result, "state": "CONVERGED", "reason": "both sides changed to the same content"}
        if not writable:
            return {**result, "state": "CONFLICT", "reason": "both sides changed and the source type cannot be "
                                                            "merged or reverse-written"}
        merged = merge3(base["content"], src["body"], note["body"])
        result["_merge"] = merged
        if merged.clean:
            return {**result, "state": "BOTH_CHANGED", "reason": "both sides changed in non-overlapping regions"}
        return {**result, "state": "CONFLICT", "regions": merged.regions, "reason": merged.reason}

    def open_conflict(self, document_id):
        return row_dict(self.db.one("SELECT * FROM conflicts WHERE document_id = ? AND status = 'open' "
                                    "ORDER BY created_at DESC LIMIT 1", (document_id,)),
                        ("regions", "observed", "suggestion", "resolution"))

    # ------------------------------------------------------------ public

    async def status(self, actor_id, document_ids=None):
        """Read-only classification of every document the actor may read."""
        ctx, out = {}, []
        for doc in self._docs(document_ids):
            if not can(self.db, actor_id, "read", doc):
                continue
            out.append(_public(await self.classify(doc, ctx)))
        return {"documents": out, "unregistered": self.unregistered_entries()}

    def unregistered_entries(self):
        local = self._load_local()
        return sorted(p for p, e in local.items() if isinstance(e, dict) and isinstance(e.get("vault_write"), dict)
                      and e["vault_write"].get("dest_path") and not identity.find_by_source(self.db, p))

    def register(self, actor_id):
        """Register managed sync_state entries as Cortex documents (admin)."""
        from cortex.users import require_admin
        require_admin(self.db, actor_id)

        def prefix_reader(source_path, expected_hash):
            path = Path(source_path)
            if not path.is_file():
                return None
            text = _extracted_text(path.read_bytes())
            return split_source_frontmatter(text.lstrip("﻿"))[0] if _sha(text) == expected_hash else None

        return identity.register_from_sync_state(self.db, self._load_local(), actor_id, prefix_reader)

    async def sync(self, actor_id, dry_run=False, document_ids=None):
        if dry_run:
            report = await self.status(actor_id, document_ids)
            for item in report["documents"]:
                item["planned_action"] = _planned(item["state"])
            return report
        with self.db.lock():
            registered = self.register(actor_id) if self.unregistered_entries() and can(
                self.db, actor_id, "admin", None) else []
            results = []
            for doc in self._docs(document_ids):
                results.append(await self.sync_document(actor_id, doc["document_id"], _locked=True))
            return {"registered": registered, "documents": results}

    async def sync_document(self, actor_id, document_id, _locked=False):
        if not _locked:
            with self.db.lock():
                return await self.sync_document(actor_id, document_id, _locked=True)
        doc = identity.get_document(self.db, document_id)
        if not can(self.db, actor_id, "write", doc):
            return {"document_id": document_id, "state": None, "status": "skipped_unauthorized",
                    "reason": f"{actor_id} may not write this document"}
        status = await self.classify(doc)
        try:
            outcome = await self._act(actor_id, doc, status)
            if status["state"] in ("SOURCE_RENAMED", "VAULT_RENAMED") and outcome.get("status") == "ok":
                follow = await self._act(actor_id, identity.get_document(self.db, document_id),
                                         await self.classify(identity.get_document(self.db, document_id)))
                outcome["then"] = {k: v for k, v in follow.items() if not k.startswith("_")}
        except PreconditionFailed as exc:
            outcome = {"status": "stale_precondition", "reason": str(exc)}
        except CortexError as exc:
            outcome = {"status": exc.code, "reason": str(exc)}
        except OSError as exc:  # one document's I/O failure must not abort the batch
            outcome = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        return {**_public(status), **outcome}

    def _docs(self, document_ids):
        docs = identity.list_documents(self.db)
        return [d for d in docs if document_ids is None or d["document_id"] in document_ids]

    # ------------------------------------------------------------ actions

    async def _act(self, actor, doc, status):
        state = status["state"]
        if state == "IN_SYNC":
            return await self._stamp_identity(actor, doc, status)
        if state == "SOURCE_ONLY_CHANGED":
            return await self._source_to_vault(actor, doc, status)
        if state == "VAULT_ONLY_CHANGED":
            return await self._propose_reverse(doc, status)
        if state == "CONVERGED":
            return await self._converge(actor, doc, status)
        if state == "BOTH_CHANGED":
            return await self._auto_merge(actor, doc, status)
        if state == "CONFLICT":
            if status.get("conflict_id"):
                return {"status": "conflict_open", "conflict_id": status["conflict_id"]}
            return await self._record_sync_conflict(actor, doc, status)
        if state == "SOURCE_RENAMED":
            return await self._source_renamed(actor, doc, status)
        if state == "VAULT_RENAMED":
            return await self._vault_renamed(actor, doc, status)
        if state in ("SOURCE_DELETED", "VAULT_DELETED", "BOTH_DELETED"):
            return await self._tombstone(actor, doc, status)
        return {"status": "no_action"}

    def _cortex_fields(self, doc, body, source_hash):
        local = self._local_entry(doc["source_path"]) or {}
        fm = managed_frontmatter(body, {"source_path": doc["source_path"], "source_sha256": source_hash or "",
                                        "prompt_version": local.get("prompt_version"),
                                        "structure_model": local.get("structure_model"),
                                        "document_id": doc["document_id"]})
        return fm

    def _compose_note(self, existing_content, body, fields):
        if existing_content and MD_FRONTMATTER_RE.match(existing_content):
            head = update_cortex_frontmatter(existing_content, fields)
            match = MD_FRONTMATTER_RE.match(head)
            return head[:match.end()].rstrip("\n") + "\n\n" + body.strip() + "\n"
        return _build_note(body, fields)

    async def _vault_write(self, op_id, path, content, expected_hash, backups):
        """Write a note only if its full content still hashes to expected_hash
        (None = must not exist). The previous content is backed up first."""
        current = await self.read_vault(path)
        current_hash = current.get("content_hash")
        if current_hash != expected_hash:
            raise PreconditionFailed(f"vault note {path} changed since it was read; nothing was written")
        if current["exists"]:
            backup = self.backups_dir / op_id / "vault" / PurePosixPath(path).name
            backup.parent.mkdir(parents=True, exist_ok=True)
            n = 0
            while backup.exists():
                n += 1
                backup = backup.with_name(f"{PurePosixPath(path).name}.{n}")
            with open(backup, "x", encoding="utf-8") as f:
                f.write(current["content"])
            backups.append(str(backup))
            await self.vault.write_note(path, content)
        elif hasattr(self.vault, "create_note"):
            await self.vault.create_note(path, content)
        else:
            await self.vault.write_note(path, content)

    def _source_write(self, op_id, doc, body, expected_bytes_hash, backups):
        from ingest.apply import ApplyRefused, atomic_replace, build_candidate, create_backup, verify_written
        path = Path(doc["source_path"])
        if path.is_symlink() or not path.is_file():
            raise PreconditionFailed(f"source {path} is missing or a symlink")
        _require_real_path(path)
        raw_before = path.read_bytes()
        if _sha(raw_before) != expected_bytes_hash:
            raise PreconditionFailed(f"source {path} changed since it was read; nothing was written")
        try:
            candidate, info = build_candidate(path.suffix.lower(), raw_before, body, hash_managed_body(body))
        except ApplyRefused as exc:
            raise CortexError(f"refusing source write: {exc.reason}", code=exc.status) from exc
        backup = create_backup(self.backups_dir / op_id, "source", str(path), raw_before)
        backups.append(str(backup))
        from ingest.apply import _SourceChangedDuringApply
        try:
            atomic_replace(path, candidate, raw_before)
        except _SourceChangedDuringApply as exc:
            raise PreconditionFailed(f"source {path} changed during the write; nothing was written") from exc
        problems = verify_written(path.read_bytes(), candidate, info, hash_managed_body(body))
        if problems:
            raise CortexError(f"post-write verification failed for {path}: {problems}; backup {backup}",
                              code="verification_failed")
        return _sha(candidate)

    def _reindex(self, doc, force=False):
        """Normal ingestion for one source (Qdrant + sync_state), if a store is
        configured. force=True re-embeds even an unchanged source (after its
        chunks were removed, or its index path changed). Returns the
        sync_state entry afterwards."""
        local = self._load_local()
        if force and isinstance(local.get(doc["source_path"]), dict):
            local[doc["source_path"]] = {k: v for k, v in local[doc["source_path"]].items() if k != "source_sha256"}
        if self.store is not None and Path(doc["source_path"]).is_file():
            from ingest.sync import index_local_path
            prior = local.get(doc["source_path"]) or {}
            ai = bool(prior.get("ai_structure_requested")) if isinstance(prior, dict) else False
            report, local = index_local_path(self.store, doc["source_path"], state=local, ai_structure=ai,
                                             structure_fn=self.structure_fn)
            if report["errors"]:
                raise CortexError(f"reindex failed: {report['errors'][0]['error']}", code="reindex_failed")
            self._save_local(local)
        return local.get(doc["source_path"]) or {}

    def generated_body(self, doc, src):
        """Body the managed note should hold for the current source: the
        pipeline's cached generated_body when it matches this source version,
        else the deterministic body contract (non-AI documents only)."""
        entry = self._local_entry(doc["source_path"]) or {}
        if isinstance(entry, dict) and entry.get("source_sha256") == src["extracted_hash"] and \
                isinstance(entry.get("generated_body"), str):
            return entry["generated_body"]
        if isinstance(entry, dict) and entry.get("ai_structure_requested"):
            raise CortexError("AI-structured source changed; run the ingestion pipeline (main.py --index "
                              "--ai-structure) or sync with a vector store configured", code="needs_ingestion")
        return src["body"]

    async def _run_op(self, actor, doc, op_type, plan, file_steps=None):
        """Journal wrapper: begin (plan) -> file steps -> commit (plan DB changes)."""
        op_id = self.journal.begin(op_type, doc["document_id"], actor, plan)
        backups = []
        try:
            effects = await file_steps(op_id, backups) if file_steps else {}
        except BaseException as exc:
            compensation = await self._compensate(plan, backups)
            self.journal.finish(op_id, "fail", actor, {"error": str(exc), "backups": backups,
                                                       "compensation": compensation})
            raise
        self.journal.finish(op_id, "commit", actor, {"backups": backups, "effects": effects},
                            db_changes=lambda: self._apply_plan(plan, op_id))
        self._post_commit(identity.get_document(self.db, doc["document_id"]))
        return op_id, backups

    def _apply_plan(self, plan, op_id):
        for rev in plan.get("revisions", []):
            versions.insert_revision(self.db, rev)
        for head in plan.get("heads", []):
            versions.record_head(self.db, head["document_id"], head["revision_id"], head["at"], op_id)
        if plan.get("document"):
            upd = plan["document"]
            identity.update_document(self.db, upd["document_id"], upd["expected_version"], **upd["fields"])
        if plan.get("tombstone"):
            t = plan["tombstone"]
            if t.get("insert"):
                row = t["insert"]
                self.db.conn.execute(
                    "INSERT INTO tombstones VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (row["tombstone_id"], row["document_id"], row["deleted_at"], row["deleted_by"],
                     row["deleted_from"], row["last_source_hash"], row["last_vault_hash"], "detected",
                     json.dumps(row.get("trash", {})), json.dumps(row.get("saved_state")), row["deleted_at"],
                     row["deleted_by"]))
            if t.get("update"):
                u = t["update"]
                cur = self.db.conn.execute(
                    "UPDATE tombstones SET status = ?, trash = ?, saved_state = COALESCE(?, saved_state), "
                    "version = version + 1, updated_at = ?, updated_by = ? WHERE tombstone_id = ? AND version = ?",
                    (u["status"], json.dumps(u["trash"]), json.dumps(u["saved_state"]) if u.get("saved_state")
                     else None, u["at"], u["actor"], u["tombstone_id"], u["expected_version"]))
                if cur.rowcount != 1:
                    raise CortexError("tombstone changed concurrently", code="concurrent_modification")
        for conflict in plan.get("conflicts", []):
            from cortex.conflicts import insert_conflict
            insert_conflict(self.db, conflict)
        for sup in plan.get("supersede_conflicts", []):
            self.db.conn.execute("UPDATE conflicts SET status = 'superseded', version = version + 1 "
                                 "WHERE conflict_id = ? AND status = 'open'", (sup,))
        if plan.get("resolve_conflict"):
            from cortex.conflicts import mark_resolved
            mark_resolved(self.db, plan["resolve_conflict"])

    async def _compensate(self, plan, backups):
        """After a failed step, put files that were already rewritten back to
        their before-state from this operation's backups."""
        restored = []
        for effect in plan.get("effects", []):
            try:
                if effect["kind"] == "vault":
                    current = await self.read_vault(effect["path"])
                    if current.get("content_hash") == effect.get("after") and effect.get("before_backup_name"):
                        match = [b for b in backups if "/vault/" in b and
                                 Path(b).name.split(".md")[0] + ".md" == effect["before_backup_name"]]
                        if match:
                            await self.vault.write_note(effect["path"], Path(match[0]).read_text(encoding="utf-8"))
                            restored.append(effect["path"])
                elif effect["kind"] == "source":
                    path = Path(effect["path"])
                    if path.is_file() and _sha(path.read_bytes()) == effect.get("after"):
                        match = [b for b in backups if b.endswith(".pre-apply")]
                        if match:
                            from ingest.apply import atomic_replace
                            atomic_replace(path, Path(match[0]).read_bytes(), path.read_bytes())
                            restored.append(str(path))
            except Exception as exc:  # noqa: BLE001 - compensation is best effort and reported
                restored.append(f"FAILED {effect['path']}: {exc}")
        return restored

    def _post_commit(self, doc):
        """Idempotent derived work: sync_state relationship and Qdrant payloads."""
        self._reconcile_state(doc)
        if self.store is None:
            return
        markdown_path = f"local_ingest/{Path(doc['source_path']).name}" if doc["source_path"] else None
        try:
            if doc["status"] == "tombstoned":
                if markdown_path:
                    self.store.delete_by_path(markdown_path)
                return
            if markdown_path and doc["current_revision_id"]:
                self.store.set_metadata(markdown_path, {"document_id": doc["document_id"],
                                                        "revision_id": doc["current_revision_id"],
                                                        "indexed_at": self.db.now()})
                from cortex.temporal import index_revision
                index_revision(self.store, versions.get_revision(self.db, doc["current_revision_id"]),
                               project=doc["project"])
        except Exception:  # noqa: BLE001 - derived index; recoverable via reindex, never blocks sync
            pass

    def _reconcile_state(self, doc):
        """Keep sync_state.json's relationship (key + vault_write.dest_path)
        derived from the document row."""
        local = self._load_local()
        changed = False
        if doc["status"] == "tombstoned":
            if doc["source_path"] in local:
                local.pop(doc["source_path"])
                changed = True
        else:
            key = doc["source_path"]
            if key not in local:
                previous = [r["source_path"] for r in self.db.all(
                    "SELECT DISTINCT source_path FROM revisions WHERE document_id = ?", (doc["document_id"],))]
                moved = next((p for p in previous if p and p != key and p in local), None)
                if moved:
                    local[key] = local.pop(moved)
                    changed = True
                else:
                    tomb = self.db.one("SELECT saved_state FROM tombstones WHERE document_id = ? AND status = "
                                       "'restored' ORDER BY updated_at DESC LIMIT 1", (doc["document_id"],))
                    if tomb and tomb["saved_state"] and json.loads(tomb["saved_state"]):
                        local[key] = json.loads(tomb["saved_state"])
                        changed = True
            entry = local.get(key)
            if isinstance(entry, dict):
                vw = entry.get("vault_write") if isinstance(entry.get("vault_write"), dict) else {}
                if vw.get("dest_path") != doc["vault_path"]:
                    entry["vault_write"] = {"dest_path": doc["vault_path"], "status": "cortex_sync",
                                            "generated_sha256": vw.get("generated_sha256")}
                    changed = True
        if changed:
            self._save_local(local)

    async def _stamp_identity(self, actor, doc, status):
        note = status["_note"]
        if note.get("document_id") == doc["document_id"]:
            return {"status": "ok", "action": "none"}
        content = update_cortex_frontmatter(note["content"], {"cortex_document_id": doc["document_id"]})
        plan = {"effects": [{"kind": "vault", "path": doc["vault_path"], "before": note["content_hash"],
                             "after": _sha(content), "before_backup_name": PurePosixPath(doc["vault_path"]).name}]}

        async def steps(op_id, backups):
            await self._vault_write(op_id, doc["vault_path"], content, note["content_hash"], backups)
            return {"vault": _sha(content)}

        op_id, _ = await self._run_op(actor, doc, "stamp_identity", plan, steps)
        return {"status": "ok", "action": "stamped cortex_document_id (body unchanged)", "op_id": op_id}

    def _revision(self, doc, content, origin, actor, parents, reason, src=None, metadata=None, provenance=None,
                  created_at=None):
        meta = dict(metadata or {})
        valid = (None, None)
        if src:
            meta.setdefault("source_sha256", src.get("extracted_hash"))
            if src.get("prefix") is not None:
                meta.setdefault("source_prefix", src["prefix"])
            valid = src.get("valid") or (None, None)
        return versions.build_revision(self.db, doc["document_id"], content, origin, actor, parent_ids=parents,
                                       metadata=meta, reason=reason, source_path=doc["source_path"],
                                       vault_path=doc["vault_path"], provenance=provenance, valid_from=valid[0],
                                       valid_to=valid[1], created_at=created_at)

    def _head_plan(self, doc, rev, extra_fields=None, set_base=True):
        fields = {"current_revision_id": rev["revision_id"]}
        if set_base:
            fields["base_revision_id"] = rev["revision_id"]
        fields.update(extra_fields or {})
        return {"heads": [{"document_id": doc["document_id"], "revision_id": rev["revision_id"],
                           "at": rev["created_at"]}],
                "document": {"document_id": doc["document_id"], "expected_version": doc["version"],
                             "fields": fields}}

    async def _source_to_vault(self, actor, doc, status):
        entry = self._reindex(doc)
        src = self.read_source(doc)
        if src.get("bytes_hash") != status["_src"].get("bytes_hash"):
            raise PreconditionFailed("source changed while syncing; rerun sync")
        body = self.generated_body(doc, src)
        note = status["_note"]
        fields = self._cortex_fields(doc, body, src["extracted_hash"])
        content = self._compose_note(note["content"], body, fields)
        rev = self._revision(doc, body, "source", actor, [doc["base_revision_id"]], "source changed", src=src,
                             metadata={"ai_structured": bool(entry.get("ai_structure_succeeded"))})
        plan = {"revisions": [rev], **self._head_plan(doc, rev, {"modified_at": src.get("modified_at"),
                                                                  "indexed_at": self.db.now()}),
                "effects": [{"kind": "vault", "path": doc["vault_path"], "before": note["content_hash"],
                             "after": _sha(content), "before_backup_name": PurePosixPath(doc["vault_path"]).name}]}

        async def steps(op_id, backups):
            await self._vault_write(op_id, doc["vault_path"], content, note["content_hash"], backups)
            return {"vault": _sha(content)}

        op_id, backups = await self._run_op(actor, doc, "source_to_vault", plan, steps)
        return {"status": "ok", "action": "source change written to the managed note", "op_id": op_id,
                "revision_id": rev["revision_id"], "backups": backups}

    async def _propose_reverse(self, doc, status):
        if not doc["reverse_writable"]:
            return {"status": "preserved", "action": "vault edit kept; this source type is not reverse-writable"}
        from ingest.proposals import create_proposals
        from ingest.reverse_analyzer import analyze_managed_notes
        local = self._load_local()
        results = await analyze_managed_notes(self.vault, local, source_filter=doc["source_path"])
        created = create_proposals(results, local_state=local, proposals_dir=self.proposals_dir,
                                   state_path=self.state_path)
        human = [c for c in created if c["classification"] == "HUMAN_MODIFIED"]
        if not human:
            return {"status": "needs_review", "action": "no applicable proposal could be created",
                    "analyzer": [r.get("classification") for r in results]}
        return {"status": "proposal_pending", "proposal_id": human[0]["proposal_id"],
                "proposal_status": human[0]["status"],
                "action": "approve and apply the proposal to write the vault edit to the source"}

    def _applied_proposal_for(self, doc, src):
        if not self.proposals_dir.is_dir():
            return None
        for path in sorted(self.proposals_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            record = data.get("apply") if isinstance(data, dict) else None
            if data.get("status") == "applied" and data.get("source_path") == doc["source_path"] and \
                    isinstance(record, dict) and record.get("post_apply_source_sha256") == src["extracted_hash"]:
                return data
        return None

    async def _converge(self, actor, doc, status):
        entry = self._reindex(doc)
        src = self.read_source(doc)
        if src.get("bytes_hash") != status["_src"].get("bytes_hash"):
            raise PreconditionFailed("source changed while syncing; rerun sync")
        note = status["_note"]
        proposal = self._applied_proposal_for(doc, src)
        origin = "vault" if proposal else "system"
        provenance = [{"proposal_id": proposal["proposal_id"], "apply_actor": (proposal.get("apply") or {}).get(
            "actor")}] if proposal else []
        parents = [doc["base_revision_id"]] if doc["base_revision_id"] else []
        rev = self._revision(doc, src["body"], origin, actor, parents,
                             "vault edit applied to the source" if proposal else "source and vault converged",
                             src=src, provenance=provenance,
                             metadata={"ai_structured": bool(entry.get("ai_structure_succeeded"))})
        fields = self._cortex_fields(doc, src["body"], src["extracted_hash"])
        content = update_cortex_frontmatter(note["content"], fields)  # body kept byte-for-byte
        plan = {"revisions": [rev], **self._head_plan(doc, rev, {"modified_at": src.get("modified_at")}),
                "effects": [{"kind": "vault", "path": doc["vault_path"], "before": note["content_hash"],
                             "after": _sha(content), "before_backup_name": PurePosixPath(doc["vault_path"]).name}]}

        async def steps(op_id, backups):
            await self._vault_write(op_id, doc["vault_path"], content, note["content_hash"], backups)
            return {"vault": _sha(content)}

        op_id, _ = await self._run_op(actor, doc, "converge", plan, steps)
        return {"status": "ok", "action": "recorded converged content; note frontmatter refreshed (body unchanged)",
                "op_id": op_id, "revision_id": rev["revision_id"], "origin": origin}

    def post_source_sha(self, doc, src, body):
        """Extracted-text hash the source will have after `body` is written
        (unchanged when the body already matches). Pure; raises if the write
        would be refused."""
        if hash_managed_body(body) == hash_managed_body(src["body"]):
            return src["extracted_hash"]
        if not doc["reverse_writable"]:
            raise CortexError("this source type cannot be reverse-written", code="unsupported_source_type")
        from ingest.apply import ApplyRefused, build_candidate
        try:
            _, info = build_candidate(Path(doc["source_path"]).suffix.lower(), src["raw"], body,
                                      hash_managed_body(body))
        except ApplyRefused as exc:
            raise CortexError(f"refusing source write: {exc.reason}", code=exc.status) from exc
        return info["expected_post_apply_source_sha256"]

    async def restore_revision(self, actor, revision_id, reason=None):
        """Make an old revision current again as a NEW revision (history is
        never rewritten), written to both sides like an edit."""
        with self.db.lock():
            old = versions.get_revision(self.db, revision_id)
            doc = identity.get_document(self.db, old["document_id"])
            authorize(self.db, actor, "write", doc)
            if doc["status"] != "active":
                raise CortexError("document is tombstoned; restore the tombstone first", code="invalid_state")
            status = await self.classify(doc)
            if status["state"] != "IN_SYNC":
                raise CortexError(f"document is {status['state']}; run sync before restoring", code="not_in_sync")
            src, note = status["_src"], status["_note"]
            rev = self._revision(doc, old["content"], "restore", actor, [doc["current_revision_id"]],
                                 reason or f"restore of revision {revision_id}",
                                 metadata={"restored_from": revision_id,
                                           "source_sha256": self.post_source_sha(doc, src, old["content"])})
            op_id, backups = await self.write_both_sides(actor, doc, "restore_revision", old["content"], src, note,
                                                         [rev], rev)
            return {"status": "restored", "revision_id": rev["revision_id"], "restored_from": revision_id,
                    "op_id": op_id, "backups": backups}

    def _source_candidate(self, doc, src, body):
        from ingest.apply import ApplyRefused, build_candidate
        try:
            return build_candidate(Path(doc["source_path"]).suffix.lower(), src["raw"], body, hash_managed_body(body))
        except ApplyRefused as exc:
            raise CortexError(f"refusing source write: {exc.reason}", code=exc.status) from exc

    async def write_both_sides(self, actor, doc, op_type, body, src, note, revisions, head_rev, extra_plan=None,
                               set_base=True):
        """Shared by auto-merge, conflict resolution, edits and restores: write
        `body` to the note and (reverse-writable) the source under exact-content
        preconditions, then commit the revisions/head atomically. Every intended
        after-hash is computed before the begin event, so an interrupted
        operation can always be recovered or rolled back."""
        authorize(self.db, actor, "write", doc)
        touches_source = hash_managed_body(body) != hash_managed_body(src["body"])
        post_sha = src["extracted_hash"]
        candidate = None
        if touches_source:
            authorize(self.db, actor, "apply", doc)
            if not doc["reverse_writable"]:
                raise CortexError("this source type cannot be reverse-written", code="unsupported_source_type")
            candidate, info = self._source_candidate(doc, src, body)
            post_sha = info["expected_post_apply_source_sha256"]
        final_note = self._compose_note(note["content"], body, self._cortex_fields(doc, body, post_sha))
        plan = {"revisions": revisions, **self._head_plan(doc, head_rev, set_base=set_base), **(extra_plan or {}),
                "effects": [{"kind": "vault", "path": doc["vault_path"], "before": note["content_hash"],
                             "after": _sha(final_note), "before_backup_name": PurePosixPath(doc["vault_path"]).name}]}
        if touches_source:
            plan["effects"].append({"kind": "source", "path": doc["source_path"], "before": src["bytes_hash"],
                                    "after": _sha(candidate)})

        async def steps(op_id, backups):
            written = {}
            if touches_source:
                written["source"] = self._source_write(op_id, doc, body, src["bytes_hash"], backups)
                if written["source"] != _sha(candidate):
                    raise CortexError("source write produced unexpected bytes", code="verification_failed")
            await self._vault_write(op_id, doc["vault_path"], final_note, note["content_hash"], backups)
            written["vault"] = _sha(final_note)
            return written

        op_id, backups = await self._run_op(actor, doc, op_type, plan, steps)
        if touches_source:
            self._reindex(identity.get_document(self.db, doc["document_id"]))
            self._post_commit(identity.get_document(self.db, doc["document_id"]))
        return op_id, backups

    async def _auto_merge(self, actor, doc, status):
        src, note, base, merged = status["_src"], status["_note"], status["_base"], status["_merge"]
        now = self.db.now()
        s_rev = self._revision(doc, src["body"], "source", actor, [base["revision_id"]], "source side of a merge",
                               src=src, created_at=now)
        v_rev = self._revision(doc, note["body"], "vault", actor, [base["revision_id"]], "vault side of a merge",
                               created_at=now)
        m_rev = self._revision(doc, merged.text, "merge", actor, [s_rev["revision_id"], v_rev["revision_id"]],
                               "deterministic 3-way merge (non-overlapping changes)", src=src, created_at=now,
                               metadata={"merge": "diff3-line", "base_revision_id": base["revision_id"],
                                         "source_sha256": self.post_source_sha(doc, src, merged.text)})
        op_id, backups = await self.write_both_sides(actor, doc, "auto_merge", merged.text, src, note,
                                               [s_rev, v_rev, m_rev], m_rev)
        return {"status": "ok", "action": "auto-merged non-overlapping source and vault edits into both sides",
                "op_id": op_id, "revision_id": m_rev["revision_id"], "backups": backups}

    async def _record_sync_conflict(self, actor, doc, status):
        from cortex.conflicts import build_conflict
        src, note, base = status["_src"], status["_note"], status["_base"]
        now = self.db.now()
        parents = [base["revision_id"]] if base else []
        s_rev = self._revision(doc, src["body"], "source", actor, parents, "source side of a conflict", src=src,
                               created_at=now)
        v_rev = self._revision(doc, note["body"], "vault", actor, parents, "vault side of a conflict", created_at=now)
        conflict = build_conflict(self.db, doc["document_id"], "sync", base["revision_id"] if base else None,
                                  s_rev["revision_id"], v_rev["revision_id"], "source", "vault",
                                  status.get("regions") or [], {"source": src["bytes_hash"],
                                                                "vault": note["content_hash"]}, actor, now)
        plan = {"revisions": [s_rev, v_rev], "conflicts": [conflict]}
        if status.get("open_conflict"):
            plan["supersede_conflicts"] = [status["open_conflict"]["conflict_id"]]
        op_id, _ = await self._run_op(actor, doc, "record_conflict", plan)
        return {"status": "conflict_recorded", "conflict_id": conflict["conflict_id"], "op_id": op_id,
                "action": "nothing written; resolve the conflict explicitly"}

    async def _source_renamed(self, actor, doc, status):
        new_path = status["new_source_path"]
        if not self._within_roots(new_path):
            raise CortexError(f"{new_path} is outside the source roots", code="outside_roots")
        if identity.find_by_source(self.db, new_path):
            raise CortexError(f"{new_path} is already tracked", code="collision")
        new_name = Path(new_path).name
        for other in identity.list_documents(self.db, "active"):
            if other["document_id"] != doc["document_id"] and Path(other["source_path"] or "").name == new_name:
                raise CortexError(f"another tracked source is also named {new_name}; its Qdrant markdown_path "
                                  "would collide", code="collision")
        fields = {"source_path": new_path}
        effects, move = [], None
        old_default = f"{self.managed_dir}/{Path(doc['source_path']).stem}-{source_id_for(doc['source_path'])}.md"
        new_default = f"{self.managed_dir}/{Path(new_path).stem}-{source_id_for(new_path)}.md"
        note = status["_note"]
        if note["exists"] and doc["vault_path"] == old_default and new_default != old_default:
            if not getattr(self.vault, "supports_moves", False):
                raise CortexError("vault backend cannot move notes; source rename recorded is pending",
                                  code="backend_unsupported")
            if await note_exists(self.vault, new_default):
                raise CortexError(f"vault path {new_default} already exists", code="collision")
            move = (doc["vault_path"], new_default)
            fields["vault_path"] = new_default
            effects.append({"kind": "vault_move", "path": doc["vault_path"], "to": new_default})
        plan = {"document": {"document_id": doc["document_id"], "expected_version": doc["version"], "fields": fields},
                "effects": effects, "previous_source_path": doc["source_path"]}
        old_markdown = f"local_ingest/{Path(doc['source_path']).name}"

        async def steps(op_id, backups):
            if move:
                await self.vault.move_note(*move)
                moved = await self.read_vault(move[1])
                updated = update_cortex_frontmatter(moved["content"], {
                    "cortex_source_path": new_path, "cortex_source_id": source_id_for(new_path),
                    "cortex_document_id": doc["document_id"]})
                await self._vault_write(op_id, move[1], updated, moved["content_hash"], backups)
            return {"moved_note": list(move) if move else None}

        op_id, _ = await self._run_op(actor, doc, "source_rename", plan, steps)
        if self.store is not None:
            try:
                self.store.delete_by_path(old_markdown)
            except Exception:  # noqa: BLE001
                pass
            new_doc = identity.get_document(self.db, doc["document_id"])
            self._reindex(new_doc, force=True)
            self._post_commit(new_doc)
        return {"status": "ok", "action": f"source path updated to {new_path}", "op_id": op_id,
                "vault_path": fields.get("vault_path", doc["vault_path"])}

    async def _vault_renamed(self, actor, doc, status):
        new_path = status["new_vault_path"]
        if not new_path.startswith(self.managed_dir + "/") or new_path.startswith(self.vault_trash + "/"):
            raise CortexError(f"{new_path} is outside the managed tree", code="outside_roots")
        if identity.find_by_vault(self.db, new_path):
            raise CortexError(f"{new_path} is already tracked by another document", code="collision")
        plan = {"document": {"document_id": doc["document_id"], "expected_version": doc["version"],
                             "fields": {"vault_path": new_path}}}
        op_id, _ = await self._run_op(actor, doc, "vault_relocate", plan)
        return {"status": "ok", "action": f"managed note location updated to {new_path} (source not renamed)",
                "op_id": op_id}

    # ------------------------------------------------------------ tombstones

    async def _tombstone(self, actor, doc, status):
        side = {"SOURCE_DELETED": "source", "VAULT_DELETED": "vault", "BOTH_DELETED": "both"}[status["state"]]
        existing = self.db.one("SELECT * FROM tombstones WHERE document_id = ? AND status = 'detected'",
                               (doc["document_id"],))
        if existing:
            return {"status": "tombstone_pending", "tombstone_id": existing["tombstone_id"],
                    "action": "awaiting approve-tombstone or restore-tombstone"}
        base = status["_base"]
        now = self.db.now()
        row = {"tombstone_id": self.db.new_id(), "document_id": doc["document_id"], "deleted_at": now,
               "deleted_by": actor, "deleted_from": side,
               "last_source_hash": base["metadata"].get("source_sha256") if base else None,
               "last_vault_hash": hash_managed_body(base["content"]) if base else None,
               "trash": {}, "saved_state": self._load_local().get(doc["source_path"])}
        plan = {"tombstone": {"insert": row}}
        op_id, _ = await self._run_op(actor, doc, "tombstone", plan)
        return {"status": "tombstoned", "tombstone_id": row["tombstone_id"], "op_id": op_id,
                "action": "nothing deleted; approve-tombstone propagates, restore-tombstone undoes"}

    def get_tombstone(self, tombstone_id):
        row = row_dict(self.db.one("SELECT * FROM tombstones WHERE tombstone_id = ?", (tombstone_id,)),
                       ("trash", "saved_state"))
        if row is None:
            raise NotFound(f"no tombstone {tombstone_id}")
        return row

    def list_tombstones(self, actor_id):
        rows = [row_dict(r, ("trash", "saved_state")) for r in
                self.db.all("SELECT * FROM tombstones ORDER BY deleted_at")]
        return [t for t in rows if can(self.db, actor_id, "read",
                                       identity.get_document(self.db, t["document_id"]))]

    def _source_to_trash(self, path, tombstone_id):
        path = Path(path)
        _require_real_path(path)
        raw = path.read_bytes()
        dest = self.trash_dir / tombstone_id / path.name
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(dest, "xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        if dest.read_bytes() != raw:
            raise CortexError("trash copy verification failed", code="verification_failed")
        path.unlink()
        return str(dest)

    def _create_source(self, path, raw):
        path = Path(path)
        if not self._within_roots(path.parent) and not self._within_roots(path):
            raise CortexError(f"{path} is outside the source roots", code="outside_roots")
        path.parent.mkdir(parents=True, exist_ok=True)
        _require_real_path(path.parent)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".cortex.tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.link(tmp, path)  # no-clobber
        finally:
            Path(tmp).unlink(missing_ok=True)

    async def approve_tombstone(self, actor, tombstone_id):
        """Propagate a detected deletion: move the surviving side into the
        recovery area (never a hard delete) and retire the document."""
        with self.db.lock():
            t = self.get_tombstone(tombstone_id)
            doc = identity.get_document(self.db, t["document_id"])
            authorize(self.db, actor, "delete", doc)
            if t["status"] != "detected":
                raise CortexError(f"tombstone is {t['status']}, not detected", code="invalid_state")
            status = await self.classify(doc)
            expected = {"source": "SOURCE_DELETED", "vault": "VAULT_DELETED", "both": "BOTH_DELETED"}[t["deleted_from"]]
            if status["state"] != expected:
                raise PreconditionFailed(f"document is now {status['state']}, not {expected}; nothing moved")
            trash, effects = {}, []
            if t["deleted_from"] == "source":
                if not getattr(self.vault, "supports_moves", False):
                    raise CortexError("vault backend cannot move notes to the recovery area", code="backend_unsupported")
                trash["vault"] = f"{self.vault_trash}/{tombstone_id}/{PurePosixPath(doc['vault_path']).name}"
                effects.append({"kind": "vault_move", "path": doc["vault_path"], "to": trash["vault"]})
            elif t["deleted_from"] == "vault":
                trash["source"] = str(self.trash_dir / tombstone_id / Path(doc["source_path"]).name)
                effects.append({"kind": "source_trash", "path": doc["source_path"], "to": trash["source"]})
            now = self.db.now()
            plan = {"effects": effects,
                    "heads": [{"document_id": doc["document_id"], "revision_id": None, "at": now}],
                    "document": {"document_id": doc["document_id"], "expected_version": doc["version"],
                                 "fields": {"status": "tombstoned", "current_revision_id": None}},
                    "tombstone": {"update": {"tombstone_id": tombstone_id, "status": "propagated", "trash": trash,
                                             "at": now, "actor": actor, "expected_version": t["version"],
                                             "saved_state": t["saved_state"] or self._load_local().get(
                                                 doc["source_path"])}}}

            async def steps(op_id, backups):
                if "vault" in trash:
                    await self.vault.move_note(doc["vault_path"], trash["vault"])
                if "source" in trash:
                    self._source_to_trash(doc["source_path"], tombstone_id)
                return trash

            op_id, _ = await self._run_op(actor, doc, "propagate_delete", plan, steps)
            return {"status": "propagated", "tombstone_id": tombstone_id, "trash": trash, "op_id": op_id}

    async def restore_tombstone(self, actor, tombstone_id):
        """Undo a deletion: bring trashed files back (no-clobber) and recreate
        the deleted side from the last revision where that is possible."""
        with self.db.lock():
            t = self.get_tombstone(tombstone_id)
            doc = identity.get_document(self.db, t["document_id"])
            authorize(self.db, actor, "write", doc)
            if t["status"] not in ("detected", "propagated"):
                raise CortexError(f"tombstone is {t['status']}", code="invalid_state")
            base = versions.get_revision(self.db, doc["base_revision_id"]) if doc["base_revision_id"] else None
            if base is None:
                raise CortexError("no revision to restore from", code="insufficient_history")
            steps_todo = []
            src_path, vault_path = Path(doc["source_path"]), doc["vault_path"]
            if t["status"] == "propagated" and t["trash"].get("vault"):
                steps_todo.append(("vault_from_trash", t["trash"]["vault"], vault_path))
            if t["status"] == "propagated" and t["trash"].get("source"):
                steps_todo.append(("source_from_trash", t["trash"]["source"], str(src_path)))
            if t["deleted_from"] in ("vault", "both"):
                steps_todo.append(("recreate_vault", None, vault_path))
            if t["deleted_from"] in ("source", "both"):
                if not doc["reverse_writable"]:
                    raise CortexError("the deleted source cannot be reconstructed from its revision (binary or "
                                      "AI-structured); restore the file yourself, then restore this tombstone",
                                      code="unsupported_source_type")
                steps_todo.append(("recreate_source", None, str(src_path)))
            if any(kind in ("recreate_source", "source_from_trash") for kind, _, _ in steps_todo):
                authorize(self.db, actor, "apply", doc)  # restoring rewrites the source file
            now = self.db.now()
            rev = versions.build_revision(self.db, doc["document_id"], base["content"], "restore", actor,
                                          parent_ids=[base["revision_id"]], metadata=dict(base["metadata"]),
                                          reason=f"restored after deletion (tombstone {tombstone_id})",
                                          source_path=doc["source_path"], vault_path=vault_path)
            plan = {"effects": [{"kind": k, "path": dst} for k, _, dst in steps_todo], "revisions": [rev],
                    "heads": [{"document_id": doc["document_id"], "revision_id": rev["revision_id"], "at": now}],
                    "document": {"document_id": doc["document_id"], "expected_version": doc["version"],
                                 "fields": {"status": "active", "current_revision_id": rev["revision_id"],
                                            "base_revision_id": rev["revision_id"]}},
                    "tombstone": {"update": {"tombstone_id": tombstone_id, "status": "restored", "trash": t["trash"],
                                             "at": now, "actor": actor, "expected_version": t["version"],
                                             "saved_state": t["saved_state"]}}}
            conflict_doc = self.db.one("SELECT document_id FROM documents WHERE status = 'active' AND document_id != ? "
                                       "AND (source_path = ? OR vault_path = ?)",
                                       (doc["document_id"], doc["source_path"], vault_path))
            if conflict_doc:
                raise CortexError("another active document now uses this path", code="collision")

            async def steps(op_id, backups):
                done = []
                for kind, frm, dst in steps_todo:
                    if kind == "vault_from_trash":
                        await self.vault.move_note(frm, dst)
                    elif kind == "source_from_trash":
                        raw = Path(frm).read_bytes()
                        self._create_source(dst, raw)
                        Path(frm).unlink()
                    elif kind == "recreate_vault":
                        fields = self._cortex_fields(doc, base["content"], base["metadata"].get("source_sha256"))
                        await self._vault_write(op_id, dst, _build_note(base["content"], fields), None, backups)
                    elif kind == "recreate_source":
                        prefix = base["metadata"].get("source_prefix") or ""
                        self._create_source(dst, (prefix + base["content"]).encode("utf-8"))
                    done.append(kind)
                return {"steps": done}

            op_id, _ = await self._run_op(actor, doc, "restore_delete", plan, steps)
            new_doc = identity.get_document(self.db, doc["document_id"])
            if self.store is not None and src_path.is_file():
                self._reindex(new_doc, force=True)
                self._post_commit(new_doc)
            return {"status": "restored", "tombstone_id": tombstone_id, "op_id": op_id,
                    "steps": [s[0] for s in steps_todo]}

    async def purge_tombstone(self, actor, tombstone_id):
        """Permanently delete a propagated tombstone's recovery copies (admin)."""
        from cortex.users import require_admin
        with self.db.lock():
            require_admin(self.db, actor)
            t = self.get_tombstone(tombstone_id)
            doc = identity.get_document(self.db, t["document_id"])
            if t["status"] != "propagated":
                raise CortexError("only propagated tombstones can be purged", code="invalid_state")
            vault_trash = t["trash"].get("vault")
            source_trash = t["trash"].get("source")
            if vault_trash and not vault_trash.startswith(f"{self.vault_trash}/{tombstone_id}/"):
                raise CortexError("recorded vault trash path is outside the recovery area", code="unsafe_path")
            if source_trash and Path(source_trash).resolve().parent != (self.trash_dir / tombstone_id).resolve():
                raise CortexError("recorded source trash path is outside the recovery area", code="unsafe_path")
            now = self.db.now()
            plan = {"effects": [{"kind": "purge", "path": p} for p in (vault_trash, source_trash) if p],
                    "tombstone": {"update": {"tombstone_id": tombstone_id, "status": "purged", "trash": {},
                                             "at": now, "actor": actor, "expected_version": t["version"]}}}

            async def steps(op_id, backups):
                if vault_trash and hasattr(self.vault, "remove_note"):
                    await self.vault.remove_note(vault_trash)
                if source_trash and Path(source_trash).exists():
                    Path(source_trash).unlink()
                return {"purged": [p for p in (vault_trash, source_trash) if p]}

            op_id, _ = await self._run_op(actor, doc, "purge_delete", plan, steps)
            return {"status": "purged", "tombstone_id": tombstone_id, "op_id": op_id}

    # ------------------------------------------------------------ recovery

    async def recover(self, actor):
        """Finish or roll back operations interrupted between begin and commit."""
        from cortex.users import require_admin
        require_admin(self.db, actor)
        outcomes = []
        with self.db.lock():
            for event in self.journal.pending():
                plan, op_id = event["data"], event["op_id"]
                states = []
                for effect in plan.get("effects", []):
                    states.append(await self._effect_state(effect))
                if all(s == "after" for s in states):
                    self.journal.finish(op_id, "recover", actor, {"states": states},
                                        db_changes=lambda p=plan, o=op_id: self._apply_plan(p, o))
                    if event["document_id"]:
                        self._post_commit(identity.get_document(self.db, event["document_id"]))
                    outcomes.append({"op_id": op_id, "result": "recovered"})
                elif all(s == "before" for s in states):
                    self.journal.finish(op_id, "rollback", actor, {"states": states})
                    outcomes.append({"op_id": op_id, "result": "rolled_back"})
                else:
                    backups = sorted(str(p) for p in (self.backups_dir / op_id).rglob("*") if p.is_file()) \
                        if (self.backups_dir / op_id).exists() else []
                    restored = await self._compensate(plan, backups)
                    states = [await self._effect_state(e) for e in plan.get("effects", [])]
                    if all(s == "before" for s in states):
                        self.journal.finish(op_id, "rollback", actor, {"states": states, "restored": restored})
                        outcomes.append({"op_id": op_id, "result": "rolled_back_from_backups"})
                    else:
                        self.journal.finish(op_id, "review", actor, {"states": states, "backups": backups})
                        outcomes.append({"op_id": op_id, "result": "needs_review", "backups": backups})
        return outcomes

    async def _effect_state(self, effect):
        kind = effect["kind"]
        if kind == "vault":
            h = (await self.read_vault(effect["path"])).get("content_hash")
        elif kind == "source":
            p = Path(effect["path"])
            h = _sha(p.read_bytes()) if p.is_file() else None
        elif kind in ("vault_move",):
            there = await note_exists(self.vault, effect["to"])
            here = await note_exists(self.vault, effect["path"])
            return "after" if there and not here else "before" if here and not there else "unknown"
        elif kind == "source_trash":
            there, here = Path(effect["to"]).is_file(), Path(effect["path"]).is_file()
            return "after" if there and not here else "before" if here and not there else "unknown"
        else:
            return "unknown"
        if h is not None and h == effect.get("after"):
            return "after"
        if h == effect.get("before"):
            return "before"
        return "unknown"


def _planned(state):
    return {
        "IN_SYNC": "none (stamp cortex_document_id into the note if missing)",
        "SOURCE_ONLY_CHANGED": "reindex source, write managed note, new revision",
        "VAULT_ONLY_CHANGED": "create/refresh a reverse-sync proposal (explicit approve + apply)",
        "CONVERGED": "record revision, refresh note frontmatter",
        "BOTH_CHANGED": "auto-merge into both sides (merge revision with two parents)",
        "CONFLICT": "record/keep an open conflict; nothing written",
        "SOURCE_RENAMED": "update source path (and move the note to its new default path)",
        "VAULT_RENAMED": "update the managed note location",
        "SOURCE_DELETED": "tombstone (no deletion)",
        "VAULT_DELETED": "tombstone (no deletion)",
        "BOTH_DELETED": "tombstone",
    }.get(state, "none")


def _public(status):
    return {k: v for k, v in status.items() if not k.startswith("_") and k != "open_conflict"}


def _iso_mtime(path):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
