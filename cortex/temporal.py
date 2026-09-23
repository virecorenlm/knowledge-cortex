"""Temporal context: time as a first-class retrieval dimension.

Two clocks, never conflated:
  transaction time  when Cortex recorded a state: revision created_at and the
                    append-only `heads` table (when each revision became the
                    document's current state; a NULL head = deleted)
  valid time        when the content says it applies: revision valid_from /
                    valid_to, set ONLY from explicit frontmatter
                    (valid_from:/valid_to:). Unknown validity is reported as
                    unknown and excluded from valid-time queries, never guessed.

Historical retrieval needs historical vectors, so every revision's content is
embedded (once, idempotently) into a derived Qdrant collection
`<collection>__history` whose chunks carry document_id / revision_id. It is an
index, rebuildable from SQLite (reindex_history); SQLite stays authoritative.
The main collection and hybrid_search are untouched: temporal_search() with
no temporal argument IS hybrid_search on the current index.

    temporal_search(query, as_of=T)                 what did we know at T
    temporal_search(query, changed_after/before)    what changed in a window
    temporal_search(query, valid_at=T[, as_of])     what was declared valid at T
    changes(after, before, project)                 metadata-only change log
    versions.revisions_current_at / history / get_revision(parent)
                                                    state before revision Z

Date-only bounds are whole days: as_of 2024-05-01 means "end of that day".
All bounds are normalized to the same microsecond UTC format the store uses,
so comparisons are exact.
"""

from qdrant_client import models

from cortex import identity, versions
from cortex.db import CortexError, to_bound  # noqa: F401 (re-exported)
from cortex.users import can, readable_document_ids

HISTORY_SUFFIX = "__history"


def history_store(store):
    from graph.store import VectorStore
    return VectorStore(ollama_url=store.ollama_url, embedding_model=store.embedding_model,
                       collection=store.collection + HISTORY_SUFFIX, ollama_client=store._http,
                       qdrant_client=store.client)


def _history_path(revision_id):
    return f"revision:{revision_id}"


def index_revision(store, rev, project=None, max_chars=1200):
    """Embed one revision into the history collection unless already there."""
    if rev is None or not rev["content"].strip():
        return 0
    hs = history_store(store)
    path = _history_path(rev["revision_id"])
    if hs.collection_exists():
        existing = hs.client.count(hs.collection, count_filter=models.Filter(must=[
            models.FieldCondition(key="path", match=models.MatchValue(value=path))]), exact=True).count
        if existing:
            return 0
    from ingest.chunk import build_chunks
    chunks = build_chunks(path, rev["content"], source="revision", max_chars=max_chars, metadata={
        "document_id": rev["document_id"], "revision_id": rev["revision_id"], "origin": rev["origin"],
        "revision_created_at": rev["created_at"], "valid_from": rev["valid_from"], "valid_to": rev["valid_to"],
        "project": project, "tags": [], "source_file": rev["source_path"], "vault_path": rev["vault_path"],
        "indexed_at": rev["created_at"]})
    hs.upsert_chunks(chunks)
    return len(chunks)


def reindex_history(db, store):
    """(Re)embed every revision missing from the history collection."""
    count = 0
    for row in db.all("SELECT revision_id FROM revisions ORDER BY created_at"):
        rev = versions.get_revision(db, row["revision_id"])
        doc = identity.get_document(db, rev["document_id"])
        count += index_revision(store, rev, project=doc["project"])
    return count


def _indexed_revisions(store, revision_ids):
    hs = history_store(store)
    if not revision_ids or not hs.collection_exists():
        return set()
    found = set()
    points, offset = hs.client.scroll(hs.collection, scroll_filter=models.Filter(must=[models.FieldCondition(
        key="revision_id", match=models.MatchAny(any=list(revision_ids)))]), limit=1000, with_payload=["revision_id"])
    while True:
        found.update(p.payload.get("revision_id") for p in points)
        if offset is None:
            break
        points, offset = hs.client.scroll(hs.collection, scroll_filter=models.Filter(must=[models.FieldCondition(
            key="revision_id", match=models.MatchAny(any=list(revision_ids)))]), limit=1000,
            with_payload=["revision_id"], offset=offset)
    return found


def revisions_changed_in(db, after=None, before=None, document_ids=None):
    sql = "SELECT document_id, revision_id, became_current_at FROM heads WHERE revision_id IS NOT NULL"
    params = []
    if after:
        sql += " AND became_current_at >= ?"
        params.append(after)
    if before:
        sql += " AND became_current_at <= ?"
        params.append(before)
    rows = db.all(sql + " ORDER BY seq", params)
    wanted = set(document_ids) if document_ids is not None else None
    return [dict(r) for r in rows if wanted is None or r["document_id"] in wanted]


def changes(db, actor, after=None, before=None, project=None, document_id=None):
    """Change log (heads that became current in the window) with diff stats."""
    after_b, before_b = to_bound(after), to_bound(before, end_of_day=True)
    readable, _ = readable_document_ids(db, actor)
    out = []
    for head in revisions_changed_in(db, after_b, before_b, readable):
        doc = identity.get_document(db, head["document_id"])
        if (project and doc["project"] != project) or (document_id and doc["document_id"] != document_id):
            continue
        rev = versions.get_revision(db, head["revision_id"])
        parent = versions.get_revision(db, rev["parent_ids"][0], missing_ok=True) if rev["parent_ids"] else None
        added = removed = 0
        if parent:
            import difflib
            for line in difflib.unified_diff(parent["content"].splitlines(), rev["content"].splitlines(), lineterm=""):
                if line.startswith("+") and not line.startswith("+++"):
                    added += 1
                elif line.startswith("-") and not line.startswith("---"):
                    removed += 1
        out.append({"document_id": doc["document_id"], "source_path": rev["source_path"], "project": doc["project"],
                    "revision_id": rev["revision_id"], "parent_revision_id": parent["revision_id"] if parent else None,
                    "became_current_at": head["became_current_at"], "origin": rev["origin"],
                    "actor_id": rev["actor_id"], "reason": rev["reason"], "lines_added": added,
                    "lines_removed": removed})
    return out


def temporal_search(db, store, actor, query, as_of=None, changed_after=None, changed_before=None, valid_at=None,
                    filters=None, prefer=None, limit=10, instruct=None, include_unknown_validity=False):
    from graph.retrieval import hybrid_search
    readable, unreadable = readable_document_ids(db, actor)
    temporal = any(v is not None for v in (as_of, changed_after, changed_before, valid_at))
    if not temporal:
        response = hybrid_search(store, query, limit=limit, filters=filters, prefer=prefer, instruct=instruct,
                                 exclude_document_ids=unreadable)
        response["temporal"] = {"mode": "current"}
        return response

    as_of_b = to_bound(as_of, end_of_day=True) if as_of else None
    info = {"mode": "historical", "as_of": as_of_b}
    if changed_after or changed_before:
        heads = revisions_changed_in(db, to_bound(changed_after), to_bound(changed_before, end_of_day=True), readable)
        candidates = {h["revision_id"]: h["document_id"] for h in heads}
        info.update(changed_after=to_bound(changed_after), changed_before=to_bound(changed_before, end_of_day=True))
    else:
        at = as_of_b or db.now()
        current = versions.revisions_current_at(db, at, readable)
        candidates = {rid: doc for doc, rid in current.items()}
    if valid_at is not None:
        valid_b = to_bound(valid_at, end_of_day=False)
        kept, unknown = {}, 0
        for rid, doc in candidates.items():
            rev = versions.get_revision(db, rid)
            if rev["valid_from"] is None and rev["valid_to"] is None:
                unknown += 1
                if include_unknown_validity:
                    kept[rid] = doc
                continue
            if (rev["valid_from"] is None or rev["valid_from"] <= valid_b) and \
                    (rev["valid_to"] is None or valid_b <= rev["valid_to"]):
                kept[rid] = doc
        candidates = kept
        info.update(valid_at=valid_b, unknown_validity_revisions=unknown)
    ids = sorted(candidates)
    indexed = _indexed_revisions(store, ids)
    info.update(revisions_considered=len(ids), unindexed_revisions=sorted(set(ids) - indexed))
    if not indexed:
        response = {"query": query, "mode": "historical", "limit": limit, "results": [], "truncated": False}
    else:
        combined = dict(filters or {})
        if "revision_id" in combined:
            raise CortexError("revision_id is set by the temporal query itself", code="invalid_filter")
        combined["revision_id"] = sorted(indexed)
        response = hybrid_search(history_store(store), query, limit=limit, filters=combined, prefer=prefer,
                                 instruct=instruct)
    response["temporal"] = info
    return response
