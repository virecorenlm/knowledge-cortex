import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient

from graph.store import VectorStore
from main import extract_to_markdown, index_to_qdrant


class FakeOllama:
    def __init__(self, dim=6):
        self.dim = dim

    def post(self, url, json, timeout):
        texts = json["input"]
        vectors = [[float((hash(t) >> (i * 4)) % 97) for i in range(self.dim)] for t in texts]
        return type("R", (), {"status_code": 200, "raise_for_status": lambda self: None,
                               "json": lambda self: {"embeddings": vectors}})()


def make_store():
    return VectorStore(
        ollama_client=FakeOllama(),
        qdrant_client=QdrantClient(location=":memory:"),
        collection="main_cli_test",
        embedding_model="fake-model",
    )


class ExtractToMarkdownBackCompatTests(unittest.TestCase):
    """Default (non --index) behavior must be unchanged: extract only, no
    embedding, no Qdrant, Markdown written to --out."""

    def test_writes_markdown_without_touching_any_store(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input").mkdir()
            (root / "input" / "note.txt").write_text("plain content", encoding="utf-8")
            out_dir = root / "vault"
            extract_to_markdown(str(root / "input"), str(out_dir))
            written = list(out_dir.glob("*.md"))
            self.assertEqual(len(written), 1)
            self.assertIn("plain content", written[0].read_text())


class IndexToQdrantTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.state_path = self.root / "sync_state.json"

    def write(self, name, content):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_indexes_and_writes_markdown_when_out_given(self):
        f = self.write("input/doc.txt", "content for the index-to-qdrant test")
        out_dir = self.root / "vault"
        store = make_store()
        code = index_to_qdrant(str(self.root / "input"), str(out_dir), self.state_path, store=store)
        self.assertEqual(code, 0)
        self.assertTrue((out_dir / "doc.md").exists())
        results = store.search("content for the index-to-qdrant test", limit=1)
        self.assertEqual(results[0]["source_file"], str(f.resolve()))

    def test_indexes_without_writing_markdown_when_out_omitted(self):
        self.write("input/doc.txt", "content indexed without a vault copy")
        store = make_store()
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store)
        self.assertEqual(code, 0)
        results = store.search("content indexed without a vault copy", limit=1)
        self.assertTrue(results)

    def test_state_persists_across_calls_for_incremental_reindex(self):
        self.write("input/doc.txt", "stable content across two runs")
        store = make_store()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store)
        self.assertTrue(self.state_path.exists())
        # second call, same store+state path: should skip as unchanged (no error, no crash)
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store)
        self.assertEqual(code, 0)

    def test_errors_produce_nonzero_exit_code(self):
        self.write("input/bad.txt", "will fail on embed")

        class FailingOllama(FakeOllama):
            def post(self, url, json, timeout):
                raise RuntimeError("simulated failure")

        failing_store = VectorStore(
            ollama_client=FailingOllama(), qdrant_client=QdrantClient(location=":memory:"),
            collection="main_cli_fail_test", embedding_model="fake-model",
        )
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=failing_store)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
