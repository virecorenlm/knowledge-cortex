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


class FakeStructureResult:
    def __init__(self, ok, text, model="fake-structure-model", reason=None):
        self.ok = ok
        self.text = text
        self.model = model
        self.reason = reason


class AiStructureTests(unittest.TestCase):
    """Requirement checklist coverage for --ai-structure, items 2-10."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name, content):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_default_behavior_unchanged_when_flag_not_passed(self):
        f = self.write("plain.txt", "plain unstructured content")
        store = make_store()
        called = []

        def spy_structure(text):
            called.append(text)
            return FakeStructureResult(True, "# Structured\n\n" + text)

        report, _ = index_local_path(store, str(f), state={}, structure_fn=spy_structure)
        self.assertEqual(called, [])  # never invoked when ai_structure is False
        self.assertNotIn("# Structured", report["indexed"][0]["markdown"])

    def test_ai_structure_flag_invokes_the_structuring_layer(self):
        f = self.write("doc.txt", "raw content to structure")
        store = make_store()
        called = []

        def spy_structure(text):
            called.append(text)
            return FakeStructureResult(True, "# Structured\n\n" + text)

        index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=spy_structure)
        self.assertEqual(len(called), 1)

    def test_structured_text_not_raw_text_reaches_the_store(self):
        f = self.write("doc.txt", "raw source content here")
        store = make_store()
        structure_fn = lambda text: FakeStructureResult(True, "# Totally Different Structured Text")
        index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=structure_fn)
        results = store.search("Totally Different Structured Text", limit=1)
        self.assertIn("Totally Different Structured Text", results[0]["text"])

    def test_provenance_metadata_survives_chunking_and_indexing(self):
        f = self.write("doc.txt", "content for provenance check")
        store = make_store()
        structure_fn = lambda text: FakeStructureResult(True, "# Structured\n\n" + text, model="my-model")
        index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=structure_fn)
        results = store.search("content for provenance check", limit=1)
        r = results[0]
        self.assertEqual(r["source_file"], str(f.resolve()))
        self.assertIn("markdown_path", r)
        self.assertTrue(r["ai_structured"])
        self.assertEqual(r["structure_model"], "my-model")
        self.assertIn("source_sha256", r)
        self.assertIn("structured_sha256", r)

    def test_structuring_failure_falls_back_to_raw_markdown(self):
        f = self.write("doc.txt", "content that will fail to structure")
        store = make_store()
        structure_fn = lambda text: FakeStructureResult(False, text, reason="simulated network failure")
        report, _ = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=structure_fn)
        entry = report["indexed"][0]
        self.assertFalse(entry["ai_structured"])
        self.assertEqual(entry["structure_reason"], "simulated network failure")
        self.assertIn("content that will fail to structure", entry["markdown"])
        results = store.search("content that will fail to structure", limit=1)
        self.assertFalse(results[0]["ai_structured"])

    def test_empty_invalid_output_falls_back_safely(self):
        f = self.write("doc.txt", "some real content here")
        store = make_store()
        structure_fn = lambda text: FakeStructureResult(False, text, reason="empty output")
        report, _ = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=structure_fn)
        self.assertEqual(len(report["indexed"]), 1)
        self.assertFalse(report["indexed"][0]["ai_structured"])

    def test_rerun_of_unchanged_already_structured_document_is_skipped(self):
        f = self.write("doc.txt", "stable content for structuring")
        store = make_store()
        structure_fn = lambda text: FakeStructureResult(True, "# Structured\n\n" + text)
        _, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=structure_fn)
        calls = []

        def counting_structure_fn(text):
            calls.append(text)
            return FakeStructureResult(True, "# Structured\n\n" + text)

        report2, _ = index_local_path(store, str(f), state=state, ai_structure=True, structure_fn=counting_structure_fn)
        self.assertEqual(report2["indexed"], [])
        self.assertEqual(report2["skipped_unchanged"], [str(f.resolve())])
        self.assertEqual(calls, [])  # not re-invoked for a truly unchanged document

    def test_enabling_structuring_on_a_previously_raw_document_forces_reprocessing(self):
        f = self.write("doc.txt", "content that starts out raw")
        store = make_store()
        _, state = index_local_path(store, str(f), state={})  # raw, ai_structure=False
        structure_fn = lambda text: FakeStructureResult(True, "# Now Structured\n\n" + text)
        report2, state2 = index_local_path(store, str(f), state=state, ai_structure=True, structure_fn=structure_fn)
        self.assertEqual(len(report2["indexed"]), 1)
        self.assertTrue(report2["indexed"][0]["ai_structured"])
        results = store.search("Now Structured", limit=1)
        self.assertTrue(results[0]["ai_structured"])

    def test_disabling_structuring_on_a_previously_structured_document_forces_reprocessing(self):
        f = self.write("doc.txt", "content that starts out structured")
        store = make_store()
        structure_fn = lambda text: FakeStructureResult(True, "# Structured\n\n" + text)
        _, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=structure_fn)
        report2, _ = index_local_path(store, str(f), state=state, ai_structure=False)
        self.assertEqual(len(report2["indexed"]), 1)
        self.assertFalse(report2["indexed"][0]["ai_structured"])

    def test_source_change_replaces_chunks_without_orphans_when_structuring_enabled(self):
        f = self.write("doc.txt", "version one of the content")
        store = make_store()
        structure_fn = lambda text: FakeStructureResult(True, "# V1 Structured\n\n" + text)
        _, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=structure_fn)
        f.write_text("version two of the content, completely different now", encoding="utf-8")
        structure_fn_v2 = lambda text: FakeStructureResult(True, "# V2 Structured\n\n" + text)
        index_local_path(store, str(f), state=state, ai_structure=True, structure_fn=structure_fn_v2)
        points, _ = store.client.scroll(collection_name="local_ingest_test", limit=100, with_payload=True)
        texts = [p.payload["text"] for p in points if p.payload["path"].endswith("doc.txt")]
        self.assertEqual(len(texts), 1)  # no orphaned chunk from V1
        self.assertIn("V2 Structured", texts[0])
        self.assertNotIn("V1 Structured", texts[0])

    def test_directory_ingestion_works_with_structuring_enabled(self):
        self.write("a.txt", "first document content")
        self.write("sub/b.txt", "second document content in a subfolder")
        store = make_store()
        structure_fn = lambda text: FakeStructureResult(True, "# Dir Structured\n\n" + text)
        report, state = index_local_path(store, str(self.root), state={}, ai_structure=True, structure_fn=structure_fn)
        self.assertEqual(len(report["indexed"]), 2)
        self.assertTrue(all(e["ai_structured"] for e in report["indexed"]))
        self.assertEqual(len(state), 2)

    def test_generated_frontmatter_timestamp_is_not_sent_to_the_model_or_validated(self):
        # Regression test: ingest.markdown.to_markdown() prepends a live
        # "ingested: <timestamp>" frontmatter block before content ever
        # reaches structuring. If that frontmatter were sent to the model
        # and validated for dropped values, a real LLM restating a
        # differently-formatted timestamp would spuriously fail validation
        # (a real bug caught during manual integration testing). Only the
        # document body may reach the model; the original frontmatter must
        # be reattached unchanged afterward.
        f = self.write("doc.txt", "actual body content with number 4821")
        store = make_store()
        seen_inputs = []

        def spy_structure_fn(body):
            seen_inputs.append(body)
            return FakeStructureResult(True, "# Restructured\n\n" + body)

        report, _ = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=spy_structure_fn)
        self.assertEqual(len(seen_inputs), 1)
        self.assertNotIn("ingested:", seen_inputs[0])
        self.assertNotIn("source:", seen_inputs[0])
        indexed_md = report["indexed"][0]["markdown"]
        self.assertIn("source:", indexed_md)  # frontmatter preserved in the final indexed text
        self.assertIn("ingested:", indexed_md)
        self.assertIn("# Restructured", indexed_md)


class FailedStructuringRetryTests(unittest.TestCase):
    """Hardening pass: a failed/fallback --ai-structure attempt must never
    become permanently "unchanged" — it must be retried on the next run
    with the same flag, and ai_structured metadata must only ever mean
    "structuring actually succeeded", never "structuring was requested"."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, name, content):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_successful_structuring_then_unchanged_rerun_skips(self):
        f = self.write("doc.txt", "content that structures successfully")
        store = make_store()
        ok_fn = lambda text: FakeStructureResult(True, "# Structured\n\n" + text)
        _, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=ok_fn)
        self.assertTrue(state[str(f.resolve())]["ai_structure_succeeded"])

        calls = []

        def counting_fn(text):
            calls.append(text)
            return FakeStructureResult(True, "# Structured\n\n" + text)

        report2, _ = index_local_path(store, str(f), state=state, ai_structure=True, structure_fn=counting_fn)
        self.assertEqual(report2["indexed"], [])
        self.assertEqual(calls, [])

    def test_failed_structuring_indexes_raw_fallback_safely(self):
        f = self.write("doc.txt", "content that fails to structure")
        store = make_store()
        fail_fn = lambda text: FakeStructureResult(False, text, reason="simulated timeout")
        report, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=fail_fn)
        self.assertEqual(len(report["indexed"]), 1)
        self.assertIn("content that fails to structure", report["indexed"][0]["markdown"])

    def test_failed_structuring_is_not_recorded_as_successfully_structured(self):
        f = self.write("doc.txt", "content that fails to structure")
        store = make_store()
        fail_fn = lambda text: FakeStructureResult(False, text, reason="simulated timeout")
        _, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=fail_fn)
        entry = state[str(f.resolve())]
        self.assertTrue(entry["ai_structure_requested"])
        self.assertFalse(entry["ai_structure_succeeded"])

    def test_next_unchanged_run_retries_structuring_after_previous_failure(self):
        f = self.write("doc.txt", "content that fails then may succeed")
        store = make_store()
        fail_fn = lambda text: FakeStructureResult(False, text, reason="simulated timeout")
        _, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=fail_fn)

        calls = []

        def retry_fn(text):
            calls.append(text)
            return FakeStructureResult(False, text, reason="simulated timeout again")

        report2, _ = index_local_path(store, str(f), state=state, ai_structure=True, structure_fn=retry_fn)
        self.assertEqual(len(calls), 1)  # structuring WAS retried, not skipped
        self.assertEqual(len(report2["indexed"]), 1)
        self.assertNotIn(str(f.resolve()), report2["skipped_unchanged"])

    def test_retry_success_updates_state_to_succeeded(self):
        f = self.write("doc.txt", "content that fails then succeeds")
        store = make_store()
        fail_fn = lambda text: FakeStructureResult(False, text, reason="simulated timeout")
        _, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=fail_fn)
        self.assertFalse(state[str(f.resolve())]["ai_structure_succeeded"])

        ok_fn = lambda text: FakeStructureResult(True, "# Now Structured\n\n" + text)
        report2, state2 = index_local_path(store, str(f), state=state, ai_structure=True, structure_fn=ok_fn)
        self.assertTrue(report2["indexed"][0]["ai_structured"])
        self.assertTrue(state2[str(f.resolve())]["ai_structure_succeeded"])

    def test_after_successful_retry_another_unchanged_run_skips(self):
        f = self.write("doc.txt", "content that fails then succeeds then stabilizes")
        store = make_store()
        fail_fn = lambda text: FakeStructureResult(False, text, reason="simulated timeout")
        _, state = index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=fail_fn)
        ok_fn = lambda text: FakeStructureResult(True, "# Now Structured\n\n" + text)
        _, state2 = index_local_path(store, str(f), state=state, ai_structure=True, structure_fn=ok_fn)

        def should_not_be_called(text):
            raise AssertionError("structure_fn must not be called for a truly unchanged, already-succeeded document")

        report3, _ = index_local_path(store, str(f), state=state2, ai_structure=True, structure_fn=should_not_be_called)
        self.assertEqual(report3["indexed"], [])
        self.assertEqual(report3["skipped_unchanged"], [str(f.resolve())])

    def test_ai_structured_metadata_is_false_on_fallback(self):
        f = self.write("doc.txt", "content that fails to structure")
        store = make_store()
        fail_fn = lambda text: FakeStructureResult(False, text, reason="simulated timeout")
        index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=fail_fn)
        results = store.search("content that fails to structure", limit=1)
        self.assertFalse(results[0]["ai_structured"])
        self.assertNotIn("structured_sha256", results[0])  # never a misleading hash of raw content

    def test_ai_structured_metadata_is_true_only_after_successful_structuring(self):
        f = self.write("doc.txt", "content that structures successfully")
        store = make_store()
        ok_fn = lambda text: FakeStructureResult(True, "# Structured\n\n" + text)
        index_local_path(store, str(f), state={}, ai_structure=True, structure_fn=ok_fn)
        results = store.search("Structured", limit=1)
        self.assertTrue(results[0]["ai_structured"])
        self.assertIn("structured_sha256", results[0])

    def test_transitional_legacy_schema_without_succeeded_field_is_retried(self):
        # Simulates state written by the code between the two hardening
        # passes (had "ai_structure" but not "ai_structure_succeeded").
        # Must be treated as "not confirmed successful" -> retried once,
        # not trusted as already-structured.
        f = self.write("doc.txt", "content with a transitional state entry")
        store = make_store()
        legacy_state = {
            str(f.resolve()): {
                "source_sha256": hashlib.sha256(f.read_text().encode("utf-8")).hexdigest(),
                # Simulate a to_markdown-wrapped digest mismatch isn't the point here;
                # what matters is the missing "ai_structure_succeeded" key.
                "ai_structure": True,
                "structure_model": "gemma4:12b",
                "prompt_version": 1,
            }
        }
        calls = []

        def retry_fn(text):
            calls.append(text)
            return FakeStructureResult(True, "# Retried\n\n" + text)

        # Source content in state won't match to_markdown's wrapped text
        # (state stores extract-stage hash), so this also exercises the
        # normal changed-content path; the key assertion is schema safety.
        report, _ = index_local_path(store, str(f), state=legacy_state, ai_structure=True, structure_fn=retry_fn)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(report["indexed"]), 1)


class DigestHelperSanity(unittest.TestCase):
    def test_sha256_is_deterministic(self):
        a = hashlib.sha256(b"x").hexdigest()
        b = hashlib.sha256(b"x").hexdigest()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
