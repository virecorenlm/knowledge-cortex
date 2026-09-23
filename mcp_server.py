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
