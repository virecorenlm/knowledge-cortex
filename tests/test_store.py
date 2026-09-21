import unittest
from unittest.mock import MagicMock

from ingest.chunk import build_chunks
from graph.store import VectorStore


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeOllama:
    """Deterministic fake: embeds each text into a fixed-size vector derived
    from its content, so identical text -> identical vector, without a
    real Ollama server."""

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []

    def post(self, url, json, timeout):
        self.calls.append(json)
        texts = json["input"]
        vectors = [[float((hash(t) >> (i * 4)) % 97) for i in range(self.dim)] for t in texts]
        return FakeResponse({"embeddings": vectors})


def make_store(dim=8, **kwargs):
    from qdrant_client import QdrantClient
    return VectorStore(
        ollama_client=FakeOllama(dim=dim),
        qdrant_client=QdrantClient(location=":memory:"),
        collection="test_collection",
        embedding_model="fake-model",
        **kwargs,
    )


class UpsertAndSearchTests(unittest.TestCase):
    def test_upsert_then_search_returns_the_matching_chunk(self):
        store = make_store()
        chunks = build_chunks("note.md", "The quick brown fox jumps over the lazy dog.", max_chars=1000)
        size = store.upsert_chunks(chunks)
        self.assertEqual(size, 8)
        results = store.search("The quick brown fox jumps over the lazy dog.", limit=3)
        self.assertTrue(any(r["path"] == "note.md" for r in results))

    def test_upsert_is_idempotent_for_identical_content(self):
        store = make_store()
        chunks = build_chunks("note.md", "same content", max_chars=1000)
        store.upsert_chunks(chunks)
        store.upsert_chunks(chunks)
        count = store.client.count(collection_name="test_collection", exact=True).count
        self.assertEqual(count, len(chunks))

    def test_reindexing_changed_content_does_not_duplicate_ids(self):
        store = make_store()
        store.index_document("note.md", "version one", max_chars=1000)
        store.index_document("note.md", "version two, much longer now", max_chars=1000)
        points, _ = store.client.scroll(collection_name="test_collection", limit=100, with_payload=True)
        texts = [p.payload["text"] for p in points]
        self.assertNotIn("version one", texts)
        self.assertIn("version two, much longer now", texts)

    def test_reindexing_with_no_new_content_clears_old_chunks(self):
        store = make_store()
        store.index_document("note.md", "will be emptied", max_chars=1000)
        store.index_document("note.md", "", max_chars=1000)
        points, _ = store.client.scroll(collection_name="test_collection", limit=100, with_payload=True)
        self.assertEqual(points, [])

    def test_empty_chunk_list_is_a_noop(self):
        store = make_store()
        self.assertIsNone(store.upsert_chunks([]))

    def test_dimension_mismatch_against_existing_collection_raises(self):
        store = make_store(dim=8)
        store.upsert_chunks(build_chunks("a.md", "hello", max_chars=1000))
        store2 = make_store(dim=4)
        store2.client = store.client  # share the same in-memory collection
        with self.assertRaises(RuntimeError):
            store2.upsert_chunks(build_chunks("b.md", "world", max_chars=1000))

    def test_delete_by_path_removes_only_that_paths_points(self):
        store = make_store()
        store.upsert_chunks(build_chunks("keep.md", "keep me", max_chars=1000))
        store.upsert_chunks(build_chunks("drop.md", "drop me", max_chars=1000))
        store.delete_by_path("drop.md")
        points, _ = store.client.scroll(collection_name="test_collection", limit=100, with_payload=True)
        paths = {p.payload["path"] for p in points}
        self.assertEqual(paths, {"keep.md"})

    def test_bad_ollama_response_raises_clear_error(self):
        store = make_store()
        store._http = MagicMock()
        store._http.post.return_value = FakeResponse({"embeddings": [[1.0]]})  # 1 vector for 2 texts
        with self.assertRaises(RuntimeError):
            store.embed(["a", "b"])

    def test_index_document_metadata_is_stored_and_searchable(self):
        store = make_store()
        store.index_document("vault/note.md", "content from a converted pdf", max_chars=1000,
                              metadata={"source_file": "/home/vire/docs/report.pdf", "markdown_path": "vault/note.md"})
        results = store.search("content from a converted pdf", limit=1)
        self.assertEqual(results[0]["source_file"], "/home/vire/docs/report.pdf")
        self.assertEqual(results[0]["markdown_path"], "vault/note.md")

    def test_index_document_without_metadata_has_no_extra_fields(self):
        store = make_store()
        store.index_document("plain.md", "plain content", max_chars=1000)
        results = store.search("plain content", limit=1)
        self.assertNotIn("source_file", results[0])


if __name__ == "__main__":
    unittest.main()
