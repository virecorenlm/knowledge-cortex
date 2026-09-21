import os

import requests
from qdrant_client import QdrantClient, models


def _env(key, default=None):
    return os.getenv(key, default)


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_QDRANT_URL = "http://localhost:6333"


class VectorStore:
    """Thin wrapper around Ollama embeddings + Qdrant storage/search.

    Configuration is read from the environment at construction time (not at
    import time), so tests and callers can override it per-instance without
    mutating process-wide state.
    """

    def __init__(self, ollama_url=None, embedding_model=None, qdrant_url=None,
                 qdrant_api_key=None, collection=None, ollama_client=None,
                 qdrant_client=None):
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
            return
        existing = self.client.get_collection(self.collection)
        existing_size = getattr(existing.config.params.vectors, "size", None)
        if existing_size is not None and existing_size != vector_size:
            raise RuntimeError(
                f"Qdrant collection '{self.collection}' uses {existing_size}-dimensional vectors, "
                f"but {self.embedding_model} returned {vector_size}. Use a new collection name or "
                "recreate the collection before indexing."
            )

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
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return vector_size

    def search(self, query_text, limit=5, instruct=None):
        """Embed a query and return the top-k matching chunk payloads with scores."""
        text = query_text if instruct is None else f"Instruct: {instruct}\nQuery:{query_text}"
        vector = self.embed([text])[0]
        results = self.client.query_points(
            collection_name=self.collection, query=vector, limit=limit, with_payload=True,
        ).points
        return [
            {"score": r.score, **(r.payload or {})}
            for r in results
        ]

    def index_document(self, path, text, source="obsidian", max_chars=1200):
        """Reindex one document: replace ALL of its existing chunks with a
        fresh set derived from the current text. This is the safe entry
        point for ingest pipelines — content-hash-derived chunk IDs mean a
        bare upsert_chunks() call can leave orphaned chunks behind when a
        document shrinks or is heavily rewritten; deleting by path first
        guarantees the stored chunks always match the current content.
        """
        from ingest.chunk import build_chunks
        chunks = build_chunks(path, text, source=source, max_chars=max_chars)
        self.delete_by_path(path)
        if chunks:
            self.upsert_chunks(chunks)
        return chunks

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
