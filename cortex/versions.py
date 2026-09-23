"""Immutable revision history ("version control for thought evolution").

A revision is one meaningful content state of a logical document: the
document body in the managed/generated representation (the same space the
vault note body and ingest.markdown.document_body use), plus metadata.
Revisions are append-only (SQLite triggers reject UPDATE/DELETE); merges
have two parents; restoring an old state creates a NEW revision.

revision_id = sha256(canonical JSON of document_id, parent_ids, content_hash,
origin, actor_id, created_at, reason, metadata)[:32], so identical inputs
always give the same id and any field change gives a different one.
content_hash = sha256(content) exactly (no normalization). Side comparisons
elsewhere use body_key() = vault_writer.hash_managed_body (outer whitespace
ignored), the project's existing body-equality contract.

Transaction time vs valid time: created_at is when Cortex recorded the
state. valid_from/valid_to are only set when the content itself declares
them (frontmatter valid_from:/valid_to:, see extract_valid_time); they are
never invented. The `heads` table (append-only) records when each revision
became the document's current state, which is what "as of T" queries use --
conflict-branch revisions that never became current are not heads.
"""

import difflib
import hashlib
import json

from cortex.db import CortexError, NotFound, canonical_json, row_dict, to_bound
from ingest.metadata import parse_frontmatter as parse_metadata_frontmatter
from ingest.vault_writer import hash_managed_body

ORIGINS = ("source", "vault", "merge", "edit", "restore", "resolution", "system")
_JSON = ("parent_ids", "metadata", "provenance")


def content_hash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def body_key(content):
    return hash_managed_body(content) if isinstance(content, str) else None


def compute_revision_id(document_id, parent_ids, content_sha, origin, actor_id, created_at, reason, metadata):
    material = {"document_id": document_id, "parent_ids": list(parent_ids), "content_hash": content_sha,
                "origin": origin, "actor_id": actor_id, "created_at": created_at, "reason": reason,
                "metadata": metadata}
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()[:32]


def extract_valid_time(text):
    """(valid_from, valid_to) from explicit frontmatter keys only."""
    fm = parse_metadata_frontmatter(text or "")
    values = []
    for key, end_of_day in (("valid_from", False), ("valid_to", True)):
        value = fm.get(key)
        try:
            values.append(to_bound(value, end_of_day=end_of_day) if isinstance(value, str) else None)
        except CortexError:
            values.append(None)  # unparseable: unknown, never guessed
    return tuple(values)


def build_revision(db, document_id, content, origin, actor_id, parent_ids=(), metadata=None, reason=None,
                   source_path=None, vault_path=None, provenance=None, valid_from=None, valid_to=None,
                   created_at=None):
    """Pure: a full revision record (not stored). Plans store these so a
    crashed operation can be replayed with identical ids."""
    if origin not in ORIGINS:
        raise CortexError(f"invalid revision origin {origin!r}", code="invalid_origin")
    if not isinstance(content, str):
        raise CortexError("revision content must be text", code="invalid_content")
    created_at = created_at or db.now()
    metadata = metadata or {}
    sha = content_hash(content)
    parents = [p for p in parent_ids if p]
    return {
        "revision_id": compute_revision_id(document_id, parents, sha, origin, actor_id, created_at, reason, metadata),
        "document_id": document_id, "parent_ids": parents, "created_at": created_at, "actor_id": actor_id,
        "origin": origin, "content_hash": sha, "content": content, "metadata": metadata,
        "source_path": source_path, "vault_path": vault_path, "reason": reason,
        "provenance": provenance or [], "valid_from": valid_from, "valid_to": valid_to,
    }


def insert_revision(db, record):
    """Idempotent: re-inserting the identical record (a replayed plan) is a
    no-op; a different record under an existing id is refused."""
    existing = get_revision(db, record["revision_id"], missing_ok=True)
    if existing is not None:
        if existing["content_hash"] != record["content_hash"] or existing["parent_ids"] != record["parent_ids"]:
            raise CortexError(f"revision id collision for {record['revision_id']}", code="revision_collision")
        return existing
    for parent in record["parent_ids"]:
        if get_revision(db, parent, missing_ok=True) is None:
            raise CortexError(f"parent revision {parent} does not exist", code="invalid_parent")
    db.conn.execute(
        "INSERT INTO revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (record["revision_id"], record["document_id"], json.dumps(record["parent_ids"]), record["created_at"],
         record["actor_id"], record["origin"], record["content_hash"], record["content"],
         canonical_json(record["metadata"]), record["source_path"], record["vault_path"], record["reason"],
         canonical_json(record["provenance"]), record["valid_from"], record["valid_to"]))
    return record


def record_head(db, document_id, revision_id, at, op_id=None):
    db.conn.execute("INSERT INTO heads (document_id, revision_id, became_current_at, op_id) VALUES (?, ?, ?, ?)",
                    (document_id, revision_id, at, op_id))


def get_revision(db, revision_id, missing_ok=False):
    rev = row_dict(db.one("SELECT * FROM revisions WHERE revision_id = ?", (revision_id,)), _JSON)
    if rev is None and not missing_ok:
        raise NotFound(f"no revision {revision_id}")
    return rev


def history(db, document_id):
    """All revisions of a document, oldest first, with which ones were heads."""
    heads = {r["revision_id"]: r["became_current_at"] for r in db.all(
        "SELECT revision_id, became_current_at FROM heads WHERE document_id = ? AND revision_id IS NOT NULL",
        (document_id,))}
    revs = [row_dict(r, _JSON) for r in db.all(
        "SELECT * FROM revisions WHERE document_id = ? ORDER BY created_at, rowid", (document_id,))]
    for rev in revs:
        rev["became_current_at"] = heads.get(rev["revision_id"])
    return revs


def head_history(db, document_id):
    return [dict(r) for r in db.all("SELECT * FROM heads WHERE document_id = ? ORDER BY seq", (document_id,))]


def diff_revisions(db, a, b, context=3):
    ra, rb = get_revision(db, a), get_revision(db, b)
    lines = list(difflib.unified_diff(ra["content"].splitlines(), rb["content"].splitlines(),
                                      fromfile=f"revision {a}", tofile=f"revision {b}", lineterm="", n=context))
    return {"from": a, "to": b, "same_document": ra["document_id"] == rb["document_id"],
            "identical": ra["content_hash"] == rb["content_hash"], "diff": lines}


def is_ancestor(db, ancestor, revision_id, limit=10000):
    """True if `ancestor` is reachable from `revision_id` via parents."""
    seen, stack = set(), [revision_id]
    while stack and len(seen) < limit:
        rid = stack.pop()
        if rid == ancestor:
            return True
        if rid in seen:
            continue
        seen.add(rid)
        rev = get_revision(db, rid, missing_ok=True)
        if rev:
            stack.extend(rev["parent_ids"])
    return False


def revisions_current_at(db, at, document_ids=None):
    """{document_id: revision_id} for documents whose head at time `at` is a
    revision (a NULL head means the document was deleted by then)."""
    rows = db.all(
        "SELECT h.document_id, h.revision_id FROM heads h JOIN ("
        "  SELECT document_id, MAX(seq) AS seq FROM heads WHERE became_current_at <= ? GROUP BY document_id"
        ") latest ON latest.seq = h.seq", (at,))
    wanted = set(document_ids) if document_ids is not None else None
    return {r["document_id"]: r["revision_id"] for r in rows
            if r["revision_id"] and (wanted is None or r["document_id"] in wanted)}


def verify_graph(db):
    """Consistency report: every parent exists, every head names an existing
    revision of the same document, and each document's current revision is
    its latest head."""
    problems = []
    ids = {r["revision_id"]: r["document_id"] for r in db.all("SELECT revision_id, document_id FROM revisions")}
    for row in db.all("SELECT revision_id, parent_ids FROM revisions"):
        for parent in json.loads(row["parent_ids"]):
            if parent not in ids:
                problems.append(f"revision {row['revision_id']} has missing parent {parent}")
    for row in db.all("SELECT * FROM heads WHERE revision_id IS NOT NULL"):
        if ids.get(row["revision_id"]) != row["document_id"]:
            problems.append(f"head {row['seq']} names revision {row['revision_id']} of another/no document")
    for doc in db.all("SELECT document_id, current_revision_id, status FROM documents"):
        last = db.one("SELECT revision_id FROM heads WHERE document_id = ? ORDER BY seq DESC LIMIT 1",
                      (doc["document_id"],))
        expected = last["revision_id"] if last else None
        if doc["status"] == "active" and expected != doc["current_revision_id"]:
            problems.append(f"document {doc['document_id']} current revision differs from its latest head")
    return problems
