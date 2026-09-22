import os
import warnings

import requests
from qdrant_client import QdrantClient, models


def _env(key, default=None):
    return os.getenv(key, default)


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_QDRANT_URL = "http://localhost:6333"

# Payload fields semantic search can filter on (see build_search_filter and
# ingest/metadata.extract_filter_metadata, which produces project/tags/doc_date).
PAYLOAD_INDEXES = {
    "path": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "project": models.PayloadSchemaType.KEYWORD,
    "tags": models.PayloadSchemaType.KEYWORD,
    "doc_date": models.PayloadSchemaType.DATETIME,
}

FILTER_KEYS = {"source", "project", "tags", "tag_mode", "date_from", "date_to"}


def _as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _match(key, values):
    if len(values) == 1:
        return models.FieldCondition(key=key, match=models.MatchValue(value=values[0]))
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def build_search_filter(filters):
    """Translate a plain filters dict into a Qdrant Filter (None if empty).

    filters keys (all optional, all combined with AND):
      source:    str or [str] — payload "source" ("obsidian", "local_ingest"); list = any of
      project:   str or [str] — exact project name; list = any of
      tags:      str or [str] — normalized like ingest (lowercase, no "#")
      tag_mode:  "all" (default: every tag required) or "any" (at least one)
      date_from: ISO date/datetime, inclusive lower bound on doc_date
      date_to:   ISO date/datetime, inclusive upper bound on doc_date; a
                 date-only value ("2024-05-31") includes that whole day
    A date bound excludes chunks with no known doc_date.
    """
    from ingest.metadata import end_of_day_exclusive, is_date_only, normalize_tag, to_rfc3339

    if not filters:
        return None
    unknown = set(filters) - FILTER_KEYS
    if unknown:
        raise ValueError(f"unknown search filter(s): {sorted(unknown)}; allowed: {sorted(FILTER_KEYS)}")

    must = []
    for key in ("source", "project"):
        values = [v for v in _as_list(filters.get(key)) if v]
        if values:
            must.append(_match(key, values))

    tags = []
    for raw in _as_list(filters.get("tags")):
        tag = normalize_tag(raw)
        if tag is None:
            raise ValueError(f"invalid tag filter: {raw!r}")
        tags.append(tag)
    tag_mode = filters.get("tag_mode") or "all"
    if tag_mode not in ("all", "any"):
        raise ValueError(f"tag_mode must be 'all' or 'any', got {tag_mode!r}")
    if tags and tag_mode == "any":
        must.append(_match("tags", tags))
    elif tags:
        must.extend(_match("tags", [t]) for t in tags)

    date_range = {}
    if filters.get("date_from"):
        date_range["gte"] = to_rfc3339(filters["date_from"])
        if date_range["gte"] is None:
            raise ValueError(f"invalid date_from: {filters['date_from']!r}")
    if filters.get("date_to"):
        if is_date_only(filters["date_to"]):
            date_range["lt"] = end_of_day_exclusive(filters["date_to"])
        else:
            date_range["lte"] = to_rfc3339(filters["date_to"])
            if date_range["lte"] is None:
                raise ValueError(f"invalid date_to: {filters['date_to']!r}")
    if date_range:
        must.append(models.FieldCondition(key="doc_date", range=models.DatetimeRange(**date_range)))

    return models.Filter(must=must) if must else None


class VectorStore:
    """Thin wrapper around Ollama embeddings + Qdrant storage/search.

    Configuration is read from the environment at construction time (not at
    import time), so tests and callers can override it per-instance without
    mutating process-wide state.
    """

    def __init__(self, ollama_url=None, embedding_model=None, qdrant_url=None,
                 qdrant_api_key=None, collection=None, ollama_client=None,
                 qdrant_client=None):
        self._payload_indexes_ensured = False
        self.ollama_url = (ollama_url or _env("OLLAMA_URL", DEFAULT_OLLAMA_URL) or DEFAULT_OLLAMA_URL).rstrip("/")
        self.embedding_model = embedding_model or _env("EMBEDDING_MODEL", "qwen3-embedding:4b")
        self.collection = collection or _env("QDRANT_COLLECTION", "knowledge_cortex")
        self._http = ollama_client or requests
        self.client = qdrant_client or QdrantClient(
            url=qdrant_url or _env("QDRANT_URL", DEFAULT_QDRANT_URL) or DEFAULT_QDRANT_URL,
            api_key=qdrant_api_key or _env("QDRANT_API_KEY"),
        )

    def embed(self, texts):
        if not texts:
            return []
        response = self._http.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.embedding_model, "input": texts, "truncate": False},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 'invalid'} "
                f"embeddings for {len(texts)} inputs"
            )
        return embeddings

    def ensure_collection(self, vector_size):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            self._ensure_payload_indexes()
            return
        existing = self.client.get_collection(self.collection)
        existing_size = getattr(existing.config.params.vectors, "size", None)
        if existing_size is not None and existing_size != vector_size:
            raise RuntimeError(
                f"Qdrant collection '{self.collection}' uses {existing_size}-dimensional vectors, "
                f"but {self.embedding_model} returned {vector_size}. Use a new collection name or "
                "recreate the collection before indexing."
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        """Create the filterable-field payload indexes (idempotent on a
        Qdrant server; once per VectorStore instance). In-memory/local
        Qdrant ignores payload indexes and warns, so that warning is muted."""
        if self._payload_indexes_ensured:
            return
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field, schema in PAYLOAD_INDEXES.items():
                self.client.create_payload_index(
                    collection_name=self.collection, field_name=field, field_schema=schema, wait=True,
                )
        self._payload_indexes_ensured = True

    def upsert_chunks(self, chunks):
        """Embed and upsert a list of chunk dicts (as produced by build_chunks).

        Returns the vector dimension used. No-op (returns None) on an empty list.
        """
        if not chunks:
            return None
        embeddings = self.embed([c["text"] for c in chunks])
        vector_size = len(embeddings[0])
        if not all(len(e) == vector_size for e in embeddings):
            raise RuntimeError("Ollama returned embeddings of inconsistent dimension")
        self.ensure_collection(vector_size)
        reserved_payload_keys = {"path", "text", "chunk_index", "sha256", "source"}
        points = [
            models.PointStruct(
                id=chunk["id"],
                vector=embedding,
                payload={
                    "path": chunk["path"],
                    "text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "sha256": chunk["sha256"],
                    "source": chunk["source"],
                    "embedding_model": self.embedding_model,
                    **{k: v for k, v in chunk.items() if k not in reserved_payload_keys and k != "id"},
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return vector_size

    def search(self, query_text, limit=5, instruct=None, filters=None):
        """Embed a query and return the top-k matching chunk payloads with scores.

        filters: optional dict restricting results by source/project/tags/
        dates — see build_search_filter. Invalid filters raise ValueError
        before anything is embedded.
        """
        query_filter = build_search_filter(filters)
        text = query_text if instruct is None else f"Instruct: {instruct}\nQuery:{query_text}"
        vector = self.embed([text])[0]
        results = self.client.query_points(
            collection_name=self.collection, query=vector, query_filter=query_filter,
            limit=limit, with_payload=True,
        ).points
        return [
            {"score": r.score, **(r.payload or {})}
            for r in results
        ]

    def index_document(self, path, text, source="obsidian", max_chars=1200, metadata=None):
        """Reindex one document: replace ALL of its existing chunks with a
        fresh set derived from the current text. This is the safe entry
        point for ingest pipelines — content-hash-derived chunk IDs mean a
        bare upsert_chunks() call can leave orphaned chunks behind when a
        document shrinks or is heavily rewritten; deleting by path first
        guarantees the stored chunks always match the current content.

        metadata: optional extra payload fields (e.g. source_file,
        markdown_path) merged into every chunk — see build_chunks.
        """
        from ingest.chunk import build_chunks
        chunks = build_chunks(path, text, source=source, max_chars=max_chars, metadata=metadata)
        self.delete_by_path(path)
        if chunks:
            self.upsert_chunks(chunks)
        return chunks

    def set_metadata(self, path, fields):
        """Overwrite payload fields on every existing chunk of `path` without
        re-embedding (used to backfill filter metadata on unchanged docs)."""
        if not self.client.collection_exists(self.collection):
            return
        self.client.set_payload(
            collection_name=self.collection, payload=fields, wait=True,
            points=models.Filter(
                must=[models.FieldCondition(key="path", match=models.MatchValue(value=path))]
            ),
        )

    def delete_by_path(self, path):
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="path", match=models.MatchValue(value=path))]
                )
            ),
        )
