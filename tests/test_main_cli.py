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


class MainCliAiStructureTests(unittest.TestCase):
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

    def test_ai_structure_requires_index_flag_at_parser_level(self):
        import subprocess
        import sys
        f = self.write("input/doc.txt", "content")
        result = subprocess.run(
            [sys.executable, "main.py", str(f), "--ai-structure"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--ai-structure requires --index", result.stderr)

    def test_index_to_qdrant_ai_structure_flag_reaches_index_local_path(self):
        from unittest.mock import patch
        f = self.write("input/doc.txt", "content for the flag-plumbing test")
        store = make_store()
        fake_structure_fn = lambda text: type("R", (), {"ok": True, "text": text, "model": "m", "reason": None})()
        with patch("ingest.sync.index_local_path", wraps=__import__("ingest.sync", fromlist=["index_local_path"]).index_local_path) as spy:
            index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                             ai_structure=True, structure_fn=fake_structure_fn)
            self.assertTrue(spy.called)
            self.assertTrue(spy.call_args.kwargs.get("ai_structure"))
            self.assertIs(spy.call_args.kwargs.get("structure_fn"), fake_structure_fn)

    def test_existing_vault_note_is_not_overwritten_when_ai_structure_enabled(self):
        f = self.write("input/doc.txt", "new content that would structure differently")
        out_dir = self.root / "vault"
        out_dir.mkdir()
        existing_note = out_dir / "doc.md"
        existing_note.write_text("# Human-authored note\n\nDo not overwrite me.", encoding="utf-8")
        store = make_store()
        fake_structure_fn = lambda text: type("R", (), {"ok": True, "text": text, "model": "m", "reason": None})()
        index_to_qdrant(str(self.root / "input"), str(out_dir), self.state_path, store=store,
                         ai_structure=True, structure_fn=fake_structure_fn)
        self.assertEqual(existing_note.read_text(), "# Human-authored note\n\nDo not overwrite me.")
        # File is still indexed even though the Markdown write was skipped.
        results = store.search("new content that would structure differently", limit=1)
        self.assertTrue(results)

    def test_source_file_itself_is_never_modified_or_deleted(self):
        f = self.write("input/doc.txt", "original untouched content")
        original_bytes = f.read_bytes()
        store = make_store()
        fake_structure_fn = lambda text: type("R", (), {"ok": True, "text": text, "model": "m", "reason": None})()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                         ai_structure=True, structure_fn=fake_structure_fn)
        self.assertTrue(f.exists())
        self.assertEqual(f.read_bytes(), original_bytes)


class MainCliWriteVaultTests(unittest.TestCase):
    """--write-vault: opt-in, requires --index, no live MCP server needed
    (an in-memory FakeObsidian is injected)."""

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

    def test_write_vault_requires_index_flag_at_parser_level(self):
        import subprocess
        import sys
        f = self.write("input/doc.txt", "content")
        result = subprocess.run(
            [sys.executable, "main.py", str(f), "--write-vault"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--write-vault requires --index", result.stderr)

    def test_write_vault_is_opt_in_indexing_without_it_never_touches_obsidian(self):
        from tests.test_vault_writer import FakeObsidian
        self.write("input/doc.txt", "content indexed without vault write-back")
        store = make_store()
        ob = FakeObsidian()
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store, obsidian=ob)
        self.assertEqual(code, 0)
        self.assertEqual(ob.files, {})  # nothing written; write_vault was never even consulted

    def test_write_vault_creates_a_managed_note_for_each_indexed_file(self):
        from tests.test_vault_writer import FakeObsidian
        self.write("input/doc.txt", "content that should land in the managed vault subtree")
        store = make_store()
        ob = FakeObsidian()
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                                write_vault=True, obsidian=ob)
        self.assertEqual(code, 0)
        self.assertEqual(len(ob.files), 1)
        dest_path = next(iter(ob.files))
        self.assertTrue(dest_path.startswith("Knowledge Cortex/Managed/"))
        self.assertIn("content that should land in the managed vault subtree", ob.files[dest_path])

    def test_write_vault_conflict_is_reported_and_does_not_error_the_batch(self):
        from tests.test_vault_writer import FakeObsidian, source_id_for
        f = self.write("input/doc.txt", "content that will conflict")
        store = make_store()
        source_id = source_id_for(str(f.resolve()))
        dest_path = f"Knowledge Cortex/Managed/doc-{source_id}.md"
        ob = FakeObsidian(files={dest_path: "# A human-authored note already lives here"})
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                                write_vault=True, obsidian=ob)
        self.assertEqual(code, 0)  # a vault conflict is reported, not treated as a fatal error
        self.assertEqual(ob.files[dest_path], "# A human-authored note already lives here")

    def test_custom_managed_vault_dir_is_respected(self):
        from tests.test_vault_writer import FakeObsidian
        self.write("input/doc.txt", "content for a custom destination")
        store = make_store()
        ob = FakeObsidian()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                         write_vault=True, managed_vault_dir="Custom/Spot", obsidian=ob)
        dest_path = next(iter(ob.files))
        self.assertTrue(dest_path.startswith("Custom/Spot/"))

    def test_unchanged_source_and_unchanged_managed_note_reports_skip_with_no_rewrite(self):
        from tests.test_vault_writer import FakeObsidian
        f = self.write("input/doc.txt", "stable content across two write-vault runs")
        store = make_store()
        ob = FakeObsidian()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                         write_vault=True, obsidian=ob)
        writes_after_first_run = len(ob.write_calls)

        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                                write_vault=True, obsidian=ob)
        self.assertEqual(code, 0)
        self.assertEqual(len(ob.write_calls), writes_after_first_run)  # no rewrite occurred

    def test_unchanged_source_but_human_edited_note_is_still_checked_and_reports_conflict(self):
        """The scenario the hardening pass targets: Qdrant must skip (no
        re-extraction/re-structuring/re-embedding) while the vault layer
        STILL inspects and correctly flags the human edit as a conflict,
        using the cached generated_body rather than reprocessing the file."""
        from tests.test_vault_writer import FakeObsidian
        f = self.write("input/doc.txt", "content that a human will later edit in the vault")
        store = make_store()
        ob = FakeObsidian()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                         write_vault=True, obsidian=ob)
        dest_path = next(iter(ob.files))

        # Simulate a human edit directly in the vault note.
        from ingest.vault_writer import _parse_frontmatter
        content = ob.files[dest_path]
        _, body = _parse_frontmatter(content)
        ob.files[dest_path] = content.replace(body.strip(), "A human edited this note directly.")

        # Source file is completely untouched -- Qdrant must skip it.
        import subprocess
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                                write_vault=True, obsidian=ob)
        self.assertEqual(code, 0)  # a vault conflict is reported, not a fatal error
        # Human edit must survive untouched.
        self.assertIn("A human edited this note directly.", ob.files[dest_path])

    def test_unchanged_source_but_deleted_managed_note_is_recreated(self):
        """Missing-note semantics: if the managed note at cortex's own
        deterministic path no longer exists, --write-vault recreates it
        (there is nothing there that could be a human's content to lose)
        rather than treating "missing" as a permanent block."""
        from tests.test_vault_writer import FakeObsidian
        f = self.write("input/doc.txt", "content whose managed note gets deleted")
        store = make_store()
        ob = FakeObsidian()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                         write_vault=True, obsidian=ob)
        dest_path = next(iter(ob.files))
        del ob.files[dest_path]  # simulate deletion of the managed note

        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                                write_vault=True, obsidian=ob)
        self.assertEqual(code, 0)
        self.assertIn(dest_path, ob.files)  # recreated
        self.assertIn("content whose managed note gets deleted", ob.files[dest_path])

    def test_generated_content_unchanged_but_volatile_timestamp_differs_no_false_conflict(self):
        """cortex_last_write changes on every write; that alone must never
        cause a false conflict or false update on an otherwise-identical
        unchanged run."""
        from tests.test_vault_writer import FakeObsidian
        f = self.write("input/doc.txt", "content whose timestamp will differ across runs")
        store = make_store()
        ob = FakeObsidian()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                         write_vault=True, obsidian=ob)
        dest_path = next(iter(ob.files))
        first_write_count = len(ob.write_calls)

        import time
        time.sleep(0.01)  # ensure a real wall-clock difference is possible
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                                write_vault=True, obsidian=ob)
        self.assertEqual(code, 0)
        self.assertEqual(len(ob.write_calls), first_write_count)  # skipped, not rewritten as "updated"

    def test_changed_source_still_replaces_qdrant_chunks_and_safely_updates_vault(self):
        from tests.test_vault_writer import FakeObsidian
        f = self.write("input/doc.txt", "version one of the content")
        store = make_store()
        ob = FakeObsidian()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                         write_vault=True, obsidian=ob)
        dest_path = next(iter(ob.files))

        f.write_text("version two of the content, completely different now", encoding="utf-8")
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                                write_vault=True, obsidian=ob)
        self.assertEqual(code, 0)
        self.assertIn("version two of the content", ob.files[dest_path])
        points, _ = store.client.scroll(collection_name="main_cli_test", limit=100, with_payload=True)
        texts = [p.payload["text"] for p in points if p.payload["path"].endswith("doc.txt")]
        self.assertEqual(len(texts), 1)  # no orphaned chunk from version one

    def test_one_conflict_in_a_directory_does_not_block_other_files_write_vault(self):
        from tests.test_vault_writer import FakeObsidian
        self.write("input/a.txt", "first document, will conflict")
        self.write("input/b.txt", "second document, unrelated and fine")
        store = make_store()
        ob = FakeObsidian()
        index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                         write_vault=True, obsidian=ob)
        self.assertEqual(len(ob.files), 2)
        dest_a = next(p for p in ob.files if "a-" in p)

        # Human-edit note "a" only.
        from ingest.vault_writer import _parse_frontmatter
        content = ob.files[dest_a]
        _, body = _parse_frontmatter(content)
        ob.files[dest_a] = content.replace(body.strip(), "human edit on file a only")

        # Force reprocessing of both by changing both sources.
        (self.root / "input" / "a.txt").write_text("first document, changed content", encoding="utf-8")
        (self.root / "input" / "b.txt").write_text("second document, changed content too", encoding="utf-8")
        code = index_to_qdrant(str(self.root / "input"), None, self.state_path, store=store,
                                write_vault=True, obsidian=ob)
        self.assertEqual(code, 0)
        self.assertIn("human edit on file a only", ob.files[dest_a])  # conflict preserved human content
        dest_b = next(p for p in ob.files if "b-" in p)
        self.assertIn("second document, changed content too", ob.files[dest_b])  # unrelated write succeeded


if __name__ == "__main__":
    unittest.main()
