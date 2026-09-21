import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ingest.sync import index_local_path
from graph.store import VectorStore
from qdrant_client import QdrantClient


class FakeOllama:
    def __init__(self, dim=6, fail_on=None):
        self.dim = dim
        self.fail_on = fail_on  # substring; if present in any input text, raise

    def post(self, url, json, timeout):
        texts = json["input"]
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError("simulated Ollama outage")
        vectors = [[float((hash(t) >> (i * 4)) % 97) for i in range(self.dim)] for t in texts]
        return type("R", (), {"status_code": 200, "raise_for_status": lambda self: None,
                               "json": lambda self: {"embeddings": vectors}})()


def make_store(**kwargs):
    return VectorStore(
        ollama_client=FakeOllama(**{k: v for k, v in kwargs.items() if k in ("dim", "fail_on")}),
        qdrant_client=kwargs.get("qdrant_client") or QdrantClient(location=":memory:"),
        collection="local_ingest_test",
        embedding_model="fake-model",
    )


class IndexLocalPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name, content):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_single_file_ingestion(self):
        f = self.write("report.txt", "quarterly results were strong")
        store = make_store()
        report, state = index_local_path(store, str(f), state={})
        self.assertEqual(len(report["indexed"]), 1)
        entry = report["indexed"][0]
        self.assertEqual(entry["source_file"], str(f))
        self.assertGreater(entry["chunk_count"], 0)
        self.assertEqual(len(state), 1)

    def test_recursive_directory_ingestion(self):
        self.write("a.txt", "alpha document content")
        self.write("sub/b.txt", "beta document content in a subfolder")
        self.write("sub/deeper/c.md", "gamma markdown content")
        store = make_store()
        report, state = index_local_path(store, str(self.root), state={})
        self.assertEqual(len(report["indexed"]), 3)
        self.assertEqual(len(state), 3)

    def test_unsupported_file_type_is_skipped_not_fatal(self):
        self.write("data.bin", "binary-ish content")
        self.write("ok.txt", "readable content here")
        store = make_store()
        report, _ = index_local_path(store, str(self.root), state={})
        self.assertEqual(len(report["indexed"]), 1)
        self.assertEqual(len(report["skipped_unsupported"]), 1)
        self.assertIn(str(self.root / "data.bin"), report["skipped_unsupported"])

    def test_empty_file_is_skipped_not_fatal(self):
        self.write("empty.txt", "   \n  ")
        store = make_store()
        report, _ = index_local_path(store, str(self.root), state={})
        self.assertEqual(report["indexed"], [])
        self.assertEqual(len(report["skipped_empty"]), 1)

    def test_unchanged_file_is_skipped_on_second_run(self):
        f = self.write("stable.txt", "unchanging content")
        store = make_store()
        _, state = index_local_path(store, str(f), state={})
        report2, state2 = index_local_path(store, str(f), state=state)
        self.assertEqual(report2["indexed"], [])
        self.assertEqual(report2["skipped_unchanged"], [str(f)])
        self.assertEqual(state, state2)

    def test_changed_file_is_reindexed_without_orphaned_chunks(self):
        f = self.write("evolving.txt", "short")
        store = make_store()
        _, state = index_local_path(store, str(f), state={})
        f.write_text("a much longer replacement body of text that changes everything", encoding="utf-8")
        report2, state2 = index_local_path(store, str(f), state=state)
        self.assertEqual(len(report2["indexed"]), 1)
        points, _ = store.client.scroll(collection_name="local_ingest_test", limit=100, with_payload=True)
        texts = [p.payload["text"] for p in points if p.payload["path"].endswith("evolving.txt")]
        self.assertTrue(all("short" not in t or "longer replacement" in t for t in texts))
        self.assertTrue(any("longer replacement" in t for t in texts))

    def test_one_bad_file_does_not_abort_the_batch_or_corrupt_state(self):
        self.write("good1.txt", "first good file content")
        self.write("bad.txt", "this one will trigger a simulated failure")
        self.write("good2.txt", "second good file content")
        store = make_store(fail_on="trigger a simulated failure")
        report, state = index_local_path(store, str(self.root), state={})
        self.assertEqual(len(report["indexed"]), 2)
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("bad.txt", report["errors"][0]["path"])
        self.assertEqual(len(state), 2)  # only the successfully indexed files got state recorded

    def test_source_metadata_is_present_in_stored_payload(self):
        f = self.write("meta.txt", "content for metadata verification")
        store = make_store()
        index_local_path(store, str(f), state={})
        results = store.search("content for metadata verification", limit=1)
        self.assertEqual(results[0]["source_file"], str(f))
        self.assertIn("markdown_path", results[0])
        self.assertEqual(results[0]["source"], "local_ingest")

    def test_generated_markdown_is_returned_for_optional_vault_write(self):
        f = self.write("towrite.txt", "content meant to also land in obsidian")
        store = make_store()
        report, _ = index_local_path(store, str(f), state={})
        entry = report["indexed"][0]
        self.assertIn("markdown", entry)
        self.assertIn("content meant to also land in obsidian", entry["markdown"])
        self.assertIn("markdown_path", entry)


class DigestHelperSanity(unittest.TestCase):
    def test_sha256_is_deterministic(self):
        a = hashlib.sha256(b"x").hexdigest()
        b = hashlib.sha256(b"x").hexdigest()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
