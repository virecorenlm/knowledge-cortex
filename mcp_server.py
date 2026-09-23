"""Knowledge Cortex MCP server: exposes semantic search and ingest tools over
the Obsidian vault + Qdrant index, so Hermes (or any MCP client) can query it
directly instead of re-walking the vault every time.

Run standalone (stdio transport, for `mcp_servers: {command: ...}` in Hermes):
    python mcp_server.py

Run over HTTP instead:
    python mcp_server.py --http --port 8770
"""

import argparse
import json

from mcp.server.mcpserver import MCPServer

from graph.store import VectorStore
from ingest.obsidian_client import ObsidianClient
from ingest.sync import sync_vault_to_index
from sync_cli import DEFAULT_EXCLUDES, DEFAULT_STATE_PATH, load_state, save_state

server = MCPServer("knowledge-cortex")


def _store():
    return VectorStore()


def _obsidian():
    return ObsidianClient()


@server.tool()
def semantic_search(query: str, limit: int = 5, source: list[str] | None = None,
                    project: list[str] | None = None, tags: list[str] | None = None,
                    tag_mode: str = "all", date_from: str = "", date_to: str = "") -> str:
    """Search the Knowledge Cortex vector index (Obsidian vault content,
    embedded with qwen3-embedding:4b) for the chunks most relevant to a
    natural-language question. Returns ranked chunks with source path,
    chunk text, similarity score, and metadata (project, tags, doc_date),
    as a JSON string.

    Optional filters (combined with AND):
      source:    any of these origins: "obsidian" (vault notes), "local_ingest" (local files)
      project:   any of these project names (vault notes default to their top-level folder)
      tags:      tag names without "#"; nested tags match their parents ("a" matches "a/b")
      tag_mode:  "all" (every tag required, default) or "any"
      date_from / date_to: ISO dates (YYYY-MM-DD or full datetime), inclusive,
                 on the document's date; excludes chunks with no known date"""
    store = _store()
    filters = {"source": source, "project": project, "tags": tags, "tag_mode": tag_mode,
               "date_from": date_from or None, "date_to": date_to or None}
    try:
        results = store.search(
            query, limit=limit,
            instruct="Given a web search query, retrieve relevant passages that answer the query",
            filters=filters,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return json.dumps(results, ensure_ascii=False, indent=2)


@server.tool()
def hybrid_search(query: str = "", limit: int = 10, filters: dict | None = None, prefer: dict | None = None,
                  max_per_source: int = 0, include_scores: bool = True) -> str:
    """Hybrid search over the Knowledge Cortex index: semantic similarity plus
    metadata. Returns a JSON object with "results" (ranked chunks with text,
    path/source_file/chunk_index provenance, metadata, and why they matched).

    filters (hard; every condition must match), fields and forms:
      source: "obsidian"|"local_ingest" or a list (any-of)
      project, path, source_file: a string (equality) or a list (any-of)
      tags: "a" (contains), ["a","b"] (contains any), {"contains_all": ["a","b"]}
      ai_structured: true|false (vault notes count as false)
      doc_date: {"gte": "2024-01-01", "lte": "2024-12-31"} (ISO dates, inclusive days)
    prefer (soft; boosts rank, never excludes): same fields/forms, optionally
      {"value": ..., "weight": 0.0-0.1}; the total boost is capped at 0.1
      cosine points, so a much more relevant chunk still wins.
    query may be empty for metadata-only listing (then filters are required).
    max_per_source: cap chunks per source file/note (0 = no cap).
    include_scores: include semantic/boost/final score breakdown."""
    from graph.retrieval import hybrid_search as run_hybrid_search, to_json
    try:
        response = run_hybrid_search(
            _store(), query, limit=limit, filters=filters, prefer=prefer,
            max_per_source=max_per_source or None,
            instruct="Given a web search query, retrieve relevant passages that answer the query",
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    if not include_scores:
        for result in response["results"]:
            result.pop("scores", None)
            result.pop("matched_preferences", None)
    return to_json(response)


def _cortex():
    """(service, owned). The acting user is the server's CORTEX_USER only;
    tools never accept a user argument, so callers cannot impersonate."""
    from cortex.api import CortexService
    return CortexService(), True


def _cortex_call(fn):
    import asyncio as _asyncio
    from cortex.db import CortexError
    svc, owned = _cortex()
    try:
        result = fn(svc)
        if _asyncio.iscoroutine(result):
            result = _run_coroutine(result)
    except CortexError as exc:
        result = {"error": str(exc), "code": exc.code}
    except ValueError as exc:
        result = {"error": str(exc), "code": "invalid_argument"}
    finally:
        if owned:
            svc.close()
    return json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _run_coroutine(coro):
    import asyncio as _asyncio
    import concurrent.futures
    try:
        _asyncio.get_running_loop()
    except RuntimeError:
        return _asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_asyncio.run, coro).result()


def _readable_doc(svc, document_id):
    from cortex import identity, users
    doc = identity.get_document(svc.db, document_id)
    users.authorize(svc.db, svc.user, "read", doc)
    return doc


@server.tool()
def sync_status(document_id: str = "") -> str:
    """Read-only bidirectional sync state (IN_SYNC, SOURCE_ONLY_CHANGED,
    VAULT_ONLY_CHANGED, BOTH_CHANGED, CONFLICT, *_RENAMED, *_DELETED, ...) of
    every Cortex document the server's user may read, or of one document."""
    return _cortex_call(lambda svc: svc.engine.status(svc.user, [document_id] if document_id else None))


@server.tool()
def sync_document(document_id: str) -> str:
    """Synchronize ONE document with the same rules as the CLI: needs write
    permission; human edits are never overwritten; vault-only edits become a
    proposal (explicit approve/apply stays outside MCP); overlapping edits
    become a conflict."""
    return _cortex_call(lambda svc: svc.engine.sync_document(svc.user, document_id))


@server.tool()
def history(document_id: str) -> str:
    """Immutable revision history of a document (with when each revision
    became current)."""
    from cortex import versions

    def call(svc):
        _readable_doc(svc, document_id)
        return {"revisions": versions.history(svc.db, document_id), "heads": versions.head_history(svc.db, document_id)}
    return _cortex_call(call)


@server.tool()
def get_revision(revision_id: str) -> str:
    """One immutable revision (content, parents, origin, actor, times)."""
    from cortex import versions

    def call(svc):
        rev = versions.get_revision(svc.db, revision_id)
        _readable_doc(svc, rev["document_id"])
        return rev
    return _cortex_call(call)


@server.tool()
def temporal_search(query: str, as_of: str = "", changed_after: str = "", changed_before: str = "",
                    valid_at: str = "", filters: dict | None = None, prefer: dict | None = None, limit: int = 10) -> str:
    """Hybrid search with time: as_of (what was known then), changed_after/
    changed_before (what changed in a window), valid_at (declared validity
    only). No time argument = current hybrid search. Visibility-filtered."""
    from cortex import temporal
    return _cortex_call(lambda svc: temporal.temporal_search(
        svc.db, svc.store, svc.user, query, as_of=as_of or None, changed_after=changed_after or None,
        changed_before=changed_before or None, valid_at=valid_at or None, filters=filters, prefer=prefer,
        limit=limit, instruct="Given a web search query, retrieve relevant passages that answer the query"))


@server.tool()
def synthesize(query: str, synthesis_type: str = "SUMMARY", filters: dict | None = None, prefer: dict | None = None,
               as_of: str = "", changed_after: str = "", changed_before: str = "", limit: int = 8) -> str:
    """Synthesize SUMMARY / DECISION / CONCEPT / RELATIONSHIP / CONTRADICTION /
    OPEN_QUESTION / CONSENSUS / CHANGE_SUMMARY / PROJECT_STATE from retrieved
    evidence. Every statement cites evidence (document, revision, chunk, hash);
    the artifact is stored, never written into sources or the evidence index."""
    from cortex import synthesis
    return _cortex_call(lambda svc: synthesis.synthesize(
        svc.db, svc.store, svc.user, query, synthesis_type, model=svc.model, filters=filters, prefer=prefer,
        limit=limit, as_of=as_of or None, changed_after=changed_after or None, changed_before=changed_before or None))


@server.tool()
def list_syntheses(status: str = "") -> str:
    """Stored syntheses visible to the server's user (current/stale/superseded)."""
    from cortex import synthesis
    return _cortex_call(lambda svc: synthesis.list_syntheses(svc.db, svc.user, status or None))


@server.tool()
def get_synthesis(synthesis_id: str) -> str:
    """One synthesis with its body, items, evidence provenance and contradictions."""
    from cortex import synthesis
    return _cortex_call(lambda svc: synthesis.read_synthesis(svc.db, svc.user, synthesis_id))


@server.tool()
def list_conflicts(status: str = "open") -> str:
    """Conflict records (sync or concurrent-edit) visible to the server's user."""
    from cortex import conflicts
    return _cortex_call(lambda svc: conflicts.list_conflicts(svc.db, svc.user, status or None))


@server.tool()
def get_conflict(conflict_id: str) -> str:
    """One conflict: base/left/right revisions, overlapping regions, any AI
    suggestion (never applied automatically), and resolution."""
    from cortex import conflicts

    def call(svc):
        c = conflicts.get_conflict(svc.db, conflict_id)
        _readable_doc(svc, c["document_id"])
        return c
    return _cortex_call(call)


@server.tool()
def resolve_conflict(conflict_id: str, action: str, body: str = "", note: str = "") -> str:
    """Explicitly resolve an open conflict: accept_source | accept_vault |
    accept_left | accept_right | manual (with body) | deterministic_merge |
    accept_suggestion. Needs write permission; refuses if the files changed
    since the conflict was recorded."""
    from cortex import conflicts
    return _cortex_call(lambda svc: conflicts.resolve_conflict(svc.engine, svc.user, conflict_id, action,
                                                               body or None, note or None))


@server.tool()
async def sync_vault(root: str = "", full: bool = False) -> str:
    """Index (or re-index) notes from the Obsidian vault into the Qdrant
    semantic index. Incremental by default (skips unchanged notes via saved
    sha256 state); pass full=true to reindex everything. Returns a JSON
    summary string."""
    obsidian = _obsidian()
    store = _store()
    state = {} if full else load_state(DEFAULT_STATE_PATH)
    report, state = await sync_vault_to_index(
        obsidian, store, root=root, exclude_substrings=DEFAULT_EXCLUDES, state=state,
    )
    save_state(DEFAULT_STATE_PATH, state)
    summary = {
        "indexed_count": len(report["indexed"]),
        "skipped_unchanged_count": len(report["skipped_unchanged"]),
        "error_count": len(report["errors"]),
        "errors": report["errors"],
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


@server.tool()
def index_status() -> str:
    """Report the current Qdrant collection name, point count, and embedding
    model in use, as a JSON string."""
    store = _store()
    exists = store.client.collection_exists(store.collection)
    count = store.client.count(collection_name=store.collection, exact=True).count if exists else 0
    status = {
        "collection": store.collection,
        "collection_exists": exists,
        "point_count": count,
        "embedding_model": store.embedding_model,
    }
    return json.dumps(status, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true", help="Run over streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    if args.http:
        server.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
