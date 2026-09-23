"""Conflict records, explicit resolution, and optimistic-concurrency edits.

Conflicts come from two places:
  sync             source and vault both changed since the base revision in
                   overlapping regions (left = source revision, right = vault)
  concurrent_edit  an edit_document() call whose expected_revision is no
                   longer current and whose change overlaps the change that
                   became current meanwhile (left = current, right = the edit)

A conflict is never auto-resolved. Resolution actions (all explicit, by an
actor with write on the document, plus apply when the source is rewritten):
  accept_left / accept_source   take the left revision's content
  accept_right / accept_vault   take the right revision's content
  manual                        a body supplied by the human
  deterministic_merge           re-run merge3; allowed only if it is now clean
  accept_suggestion             the stored AI suggestion, explicitly accepted
Every resolution first re-checks that the files still hold exactly what was
observed when the conflict was recorded (otherwise: stale, re-sync), then
writes a `resolution` revision with parents [left, right] to both sides and
marks the conflict resolved in the same transaction. Resolved conflicts are
immutable (SQLite trigger), which keeps the audit history.

edit_document(document_id, body, expected_revision_id) is the optimistic-
concurrency entry point: it applies only if expected == current; otherwise
it 3-way merges (base = expected) and either commits a merge revision or
records a concurrent_edit conflict. Nothing is ever last-write-wins.
"""

import hashlib
import json

from cortex import identity, versions
from cortex.db import CortexError, NotFound, canonical_json, row_dict
from cortex.merge import merge3
from cortex.users import authorize

ACTIONS = ("accept_left", "accept_right", "accept_source", "accept_vault", "manual", "deterministic_merge",
           "accept_suggestion")
SUGGESTION_PROMPT_VERSION = 1
_JSON = ("regions", "observed", "suggestion", "resolution")


def build_conflict(db, document_id, kind, base_revision_id, left_id, right_id, left_label, right_label, regions,
                   observed, actor, created_at):
    material = canonical_json([document_id, kind, base_revision_id, left_id, right_id, observed])
    return {"conflict_id": hashlib.sha256(material.encode()).hexdigest()[:24], "document_id": document_id,
            "kind": kind, "base_revision_id": base_revision_id, "left_revision_id": left_id,
            "right_revision_id": right_id, "left_label": left_label, "right_label": right_label,
            "created_at": created_at, "created_by": actor, "regions": regions, "observed": observed}


def insert_conflict(db, c):
    if db.one("SELECT 1 FROM conflicts WHERE conflict_id = ?", (c["conflict_id"],)):
        return
    db.conn.execute(
        "INSERT INTO conflicts (conflict_id, document_id, kind, base_revision_id, left_revision_id, "
        "right_revision_id, left_label, right_label, created_at, created_by, status, regions, observed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
        (c["conflict_id"], c["document_id"], c["kind"], c["base_revision_id"], c["left_revision_id"],
         c["right_revision_id"], c["left_label"], c["right_label"], c["created_at"], c["created_by"],
         canonical_json(c["regions"]), canonical_json(c["observed"])))


def mark_resolved(db, data):
    cur = db.conn.execute(
        "UPDATE conflicts SET status = 'resolved', resolution = ?, resolved_at = ?, resolved_by = ?, "
        "resolution_revision_id = ?, version = version + 1 WHERE conflict_id = ? AND status = 'open'",
        (canonical_json(data["resolution"]), data["resolved_at"], data["resolved_by"], data["revision_id"],
         data["conflict_id"]))
    if cur.rowcount != 1:
        raise CortexError("conflict is no longer open", code="concurrent_modification")


def get_conflict(db, conflict_id):
    c = row_dict(db.one("SELECT * FROM conflicts WHERE conflict_id = ?", (conflict_id,)), _JSON)
    if c is None:
        raise NotFound(f"no conflict {conflict_id}")
    return c


def list_conflicts(db, actor_id, status=None):
    from cortex.users import can
    sql, params = "SELECT * FROM conflicts", ()
    if status:
        sql, params = sql + " WHERE status = ?", (status,)
    rows = [row_dict(r, _JSON) for r in db.all(sql + " ORDER BY created_at", params)]
    return [c for c in rows if can(db, actor_id, "read", identity.get_document(db, c["document_id"]))]


def _resolution_body(db, conflict, action, body):
    left = versions.get_revision(db, conflict["left_revision_id"])
    right = versions.get_revision(db, conflict["right_revision_id"])
    if action in ("accept_source", "accept_vault") and conflict["kind"] != "sync":
        raise CortexError("accept_source/accept_vault apply to sync conflicts; use accept_left/accept_right",
                          code="invalid_action")
    if action in ("accept_left", "accept_source"):
        return left["content"]
    if action in ("accept_right", "accept_vault"):
        return right["content"]
    if action == "manual":
        if not isinstance(body, str) or not body.strip():
            raise CortexError("manual resolution needs a non-empty body", code="invalid_body")
        return body
    if action == "deterministic_merge":
        base = versions.get_revision(db, conflict["base_revision_id"]) if conflict["base_revision_id"] else None
        if base is None:
            raise CortexError("no base revision to merge from", code="insufficient_history")
        merged = merge3(base["content"], left["content"], right["content"])
        if not merged.clean:
            raise CortexError(f"still not cleanly mergeable: {merged.reason}", code="merge_conflict")
        return merged.text
    if action == "accept_suggestion":
        suggestion = conflict.get("suggestion")
        if not suggestion or not isinstance(suggestion.get("text"), str) or not suggestion["text"].strip():
            raise CortexError("no AI suggestion is stored for this conflict", code="no_suggestion")
        return suggestion["text"]
    raise CortexError(f"unknown resolution action {action!r}; one of {ACTIONS}", code="invalid_action")


async def resolve_conflict(engine, actor, conflict_id, action, body=None, note=None):
    db = engine.db
    with db.lock():
        conflict = get_conflict(db, conflict_id)
        if conflict["status"] != "open":
            raise CortexError(f"conflict is {conflict['status']}; resolved conflicts are immutable",
                              code="invalid_state")
        doc = identity.get_document(db, conflict["document_id"])
        authorize(db, actor, "write", doc)
        resolved_body = _resolution_body(db, conflict, action, body)
        src = engine.read_source(doc)
        vault = await engine.read_vault(doc["vault_path"])
        observed = {"source": src.get("bytes_hash"), "vault": vault.get("content_hash")}
        if observed != conflict["observed"]:
            raise CortexError("files changed since the conflict was recorded; run sync (it supersedes this "
                              "conflict) and resolve the new one", code="stale_precondition")
        now = db.now()
        rev = engine._revision(doc, resolved_body, "resolution", actor,
                               [conflict["left_revision_id"], conflict["right_revision_id"]],
                               f"resolution of conflict {conflict_id} ({action})",
                               src=src,
                               metadata={"conflict_id": conflict_id, "action": action,
                                         "source_sha256": engine.post_source_sha(doc, src, resolved_body),
                                         "suggestion_model": (conflict.get("suggestion") or {}).get("model")
                                         if action == "accept_suggestion" else None},
                               created_at=now)
        extra = {"resolve_conflict": {"conflict_id": conflict_id, "resolved_at": now, "resolved_by": actor,
                                      "revision_id": rev["revision_id"],
                                      "resolution": {"action": action, "note": note}}}
        op_id, backups = await engine.write_both_sides(actor, doc, "resolve_conflict", resolved_body, src, vault,
                                                       [rev], rev, extra_plan=extra)
        return {"status": "resolved", "conflict_id": conflict_id, "revision_id": rev["revision_id"], "op_id": op_id,
                "backups": backups}


def suggest_resolution(db, actor, conflict_id, model):
    """Ask a model for a merged body. Stored as a SUGGESTION on the open
    conflict; it is never applied unless someone explicitly resolves with
    accept_suggestion."""
    conflict = get_conflict(db, conflict_id)
    if conflict["status"] != "open":
        raise CortexError("only open conflicts take suggestions", code="invalid_state")
    authorize(db, actor, "read", identity.get_document(db, conflict["document_id"]))
    left = versions.get_revision(db, conflict["left_revision_id"])
    right = versions.get_revision(db, conflict["right_revision_id"])
    base = versions.get_revision(db, conflict["base_revision_id"]) if conflict["base_revision_id"] else None
    system = ("You merge two edited versions of one Markdown document. Keep every change from both sides "
              "when possible; never invent content. Output only the merged document.")
    prompt = (f"BASE:\n{base['content'] if base else ''}\n\n{conflict['left_label'].upper()}:\n{left['content']}"
              f"\n\n{conflict['right_label'].upper()}:\n{right['content']}")
    text = model.generate(system, prompt)
    suggestion = {"status": "suggestion", "applied": False, "text": text, "model": model.name,
                  "provider": model.provider, "prompt_version": SUGGESTION_PROMPT_VERSION,
                  "created_at": db.now(), "created_by": actor,
                  "evidence": [conflict["base_revision_id"], conflict["left_revision_id"],
                               conflict["right_revision_id"]]}
    with db.transaction():
        cur = db.conn.execute("UPDATE conflicts SET suggestion = ?, version = version + 1 WHERE conflict_id = ? "
                              "AND status = 'open'", (canonical_json(suggestion), conflict_id))
        if cur.rowcount != 1:
            raise CortexError("conflict is no longer open", code="concurrent_modification")
    return suggestion


async def edit_document(engine, actor, document_id, body, expected_revision_id, reason=None):
    db = engine.db
    if not isinstance(body, str) or not body.strip():
        raise CortexError("edit body must be non-empty text", code="invalid_body")
    with db.lock():
        doc = identity.get_document(db, document_id)
        authorize(db, actor, "write", doc)
        if doc["status"] != "active":
            raise CortexError("document is tombstoned", code="invalid_state")
        if not doc["reverse_writable"]:
            raise CortexError("edits need a reverse-writable source (.md/.txt, not AI-structured); edit the "
                              "source instead", code="unsupported_source_type")
        status = await engine.classify(doc)
        if status["state"] != "IN_SYNC":
            raise CortexError(f"document is {status['state']}; run sync before editing", code="not_in_sync")
        src, note = status["_src"], status["_note"]
        current = versions.get_revision(db, doc["current_revision_id"])
        now = db.now()
        if expected_revision_id == doc["current_revision_id"]:
            rev = engine._revision(doc, body, "edit", actor, [current["revision_id"]], reason or "edit",
                                   src=src, created_at=now,
                                   metadata={"source_sha256": engine.post_source_sha(doc, src, body)})
            op_id, backups = await engine.write_both_sides(actor, doc, "edit", body, src, note, [rev], rev)
            return {"status": "applied", "revision_id": rev["revision_id"], "op_id": op_id}
        base = versions.get_revision(db, expected_revision_id, missing_ok=True)
        if base is None or base["document_id"] != document_id:
            raise CortexError("expected_revision_id is not a revision of this document", code="invalid_revision")
        if not versions.is_ancestor(db, expected_revision_id, current["revision_id"]):
            raise CortexError("expected_revision_id is not an ancestor of the current revision",
                              code="invalid_revision")
        edit_rev = engine._revision(doc, body, "edit", actor, [expected_revision_id],
                                    reason or "edit (based on an older revision)", created_at=now)
        merged = merge3(base["content"], current["content"], body)
        if merged.clean:
            merge_rev = engine._revision(doc, merged.text, "merge", actor,
                                         [current["revision_id"], edit_rev["revision_id"]],
                                         "deterministic merge of a concurrent edit", created_at=now,
                                         src=src, metadata={"merge": "diff3-line",
                                                            "base_revision_id": expected_revision_id,
                                                            "source_sha256": engine.post_source_sha(doc, src,
                                                                                                    merged.text)})
            op_id, _ = await engine.write_both_sides(actor, doc, "edit_merge", merged.text, src, note,
                                                     [edit_rev, merge_rev], merge_rev)
            return {"status": "merged", "revision_id": merge_rev["revision_id"], "edit_revision_id":
                    edit_rev["revision_id"], "op_id": op_id}
        conflict = build_conflict(db, document_id, "concurrent_edit", expected_revision_id, current["revision_id"],
                                  edit_rev["revision_id"], "current", "edit", merged.regions,
                                  {"source": src["bytes_hash"], "vault": note["content_hash"]}, actor, now)
        op_id, _ = await engine._run_op(actor, doc, "record_conflict", {"revisions": [edit_rev],
                                                                          "conflicts": [conflict]})
        return {"status": "conflict", "conflict_id": conflict["conflict_id"], "edit_revision_id":
                edit_rev["revision_id"], "op_id": op_id,
                "reason": "the edit overlaps a change made after expected_revision_id; nothing was overwritten"}
