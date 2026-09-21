import hashlib
import re
import tempfile
import unittest
from pathlib import Path

from ingest.reverse_analyzer import analyze_managed_notes
from ingest.vault_writer import write_managed_note, source_id_for, parse_frontmatter
from tests.test_vault_writer import FakeObsidian, default_metadata


def make_state_entry(source_path, generated_body, source_sha256=None,
                      dest_path=None, ai_structure_succeeded=False, structure_model=None):
    dest_path = dest_path or f"Knowledge Cortex/Managed/{Path(source_path).stem}-{source_id_for(source_path)}.md"
    return {
        "source_sha256": source_sha256,
        "ai_structure_requested": bool(structure_model),
        "ai_structure_succeeded": ai_structure_succeeded,
        "structure_model": structure_model,
        "prompt_version": 1 if structure_model else None,
        "generated_body": generated_body,
        "vault_write": {"dest_path": dest_path, "status": "created", "generated_sha256": "unused"},
    }


def sha256_of(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReverseAnalyzerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def make_source(self, name, content="matching source content for a given test"):
        """Create a REAL file on disk whose extracted-text sha256 equals
        sha256_of(content) exactly (ingest.extract's plain-text path is a
        straight read, so this holds without invoking extract_text)."""
        p = self.root / name
        p.write_text(content, encoding="utf-8")
        return str(p.resolve())

    async def _seeded_note(self, ob, source_path, body):
        """Write a real managed note via write_managed_note so its
        frontmatter is byte-identical to what production write-back
        produces (rather than hand-constructing frontmatter in tests)."""
        meta = default_metadata(source_path=source_path)
        dest_path = f"Knowledge Cortex/Managed/{Path(source_path).stem}-{source_id_for(source_path)}.md"
        await write_managed_note(ob, dest_path, body, meta)
        return dest_path

    async def test_in_sync(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_a.txt", "stable source text")
        dest_path = await self._seeded_note(ob, source_path, "stable body content")
        state = {source_path: make_state_entry(source_path, "stable body content",
                                                source_sha256=sha256_of("stable source text"), dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["classification"], "IN_SYNC")
        self.assertEqual(results[0]["proposed_action"], "none")

    async def test_human_modified(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_b.txt", "source b text")
        dest_path = await self._seeded_note(ob, source_path, "original body")
        content = ob.files[dest_path]
        fm, body = parse_frontmatter(content)
        ob.files[dest_path] = content.replace(body.strip(), "a human changed this content")
        state = {source_path: make_state_entry(source_path, "original body",
                                                source_sha256=sha256_of("source b text"), dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "HUMAN_MODIFIED")
        self.assertEqual(results[0]["proposed_action"], "review_human_changes")

    async def test_missing(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_c.txt")
        dest_path = f"Knowledge Cortex/Managed/doc_c-{source_id_for(source_path)}.md"
        state = {source_path: make_state_entry(source_path, "never written body",
                                                source_sha256=sha256_of("matching source content for a given test"),
                                                dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "MISSING")
        self.assertEqual(results[0]["proposed_action"], "recreate_missing_managed_note")

    async def test_unmanaged_at_target(self):
        source_path = self.make_source("doc_d.txt")
        dest_path = f"Knowledge Cortex/Managed/doc_d-{source_id_for(source_path)}.md"
        ob = FakeObsidian(files={dest_path: "# A human-authored note, never touched by cortex"})
        state = {source_path: make_state_entry(source_path, "some generated body",
                                                source_sha256=sha256_of("matching source content for a given test"),
                                                dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "UNMANAGED_AT_TARGET")
        self.assertEqual(results[0]["proposed_action"], "investigate_provenance")

    async def test_source_changed(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_e.txt", "this is the NEW content on disk")
        dest_path = await self._seeded_note(ob, source_path, "generated from OLD content")
        state = {source_path: make_state_entry(source_path, "generated from OLD content",
                                                source_sha256="stale-hash-that-wont-match", dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "SOURCE_CHANGED")
        self.assertEqual(results[0]["proposed_action"], "source_changed_reprocess_required")

    async def test_source_missing(self):
        ob = FakeObsidian()
        source_path = str(self.root / "does_not_exist" / "doc_f.txt")  # never created
        dest_path = await self._seeded_note(ob, source_path, "generated body for a since-deleted source")
        state = {source_path: make_state_entry(source_path, "generated body for a since-deleted source",
                                                source_sha256="whatever", dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "SOURCE_MISSING")
        self.assertTrue(results[0]["flags"]["source_missing"])

    async def test_invalid_managed_note_missing_required_fields(self):
        source_path = self.make_source("doc_g.txt")
        dest_path = f"Knowledge Cortex/Managed/doc_g-{source_id_for(source_path)}.md"
        malformed = "---\ncortex_managed: true\n---\n\nMissing cortex_source_id and friends."
        ob = FakeObsidian(files={dest_path: malformed})
        state = {source_path: make_state_entry(source_path, "some body",
                                                source_sha256=sha256_of("matching source content for a given test"),
                                                dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "INVALID_MANAGED_NOTE")

    async def test_invalid_managed_note_source_id_mismatch(self):
        source_path = self.make_source("doc_h.txt")
        dest_path = f"Knowledge Cortex/Managed/doc_h-{source_id_for(source_path)}.md"
        malformed = (
            "---\ncortex_managed: true\ncortex_source_id: wrongvalue123\n"
            f"cortex_source_path: {source_path}\ncortex_generated_sha256: "
            + "a" * 64 + "\n---\n\nBody content."
        )
        ob = FakeObsidian(files={dest_path: malformed})
        state = {source_path: make_state_entry(source_path, "some body",
                                                source_sha256=sha256_of("matching source content for a given test"),
                                                dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "INVALID_MANAGED_NOTE")
        self.assertIn("cortex_source_id mismatch", results[0]["reason"])

    async def test_volatile_timestamp_change_does_not_produce_false_human_modified(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_i.txt", "source i text")
        dest_path = await self._seeded_note(ob, source_path, "identical content")
        content = ob.files[dest_path]
        edited = re.sub(r"cortex_last_write: .*", "cortex_last_write: 2099-01-01T00:00:00", content)
        ob.files[dest_path] = edited
        state = {source_path: make_state_entry(source_path, "identical content",
                                                source_sha256=sha256_of("source i text"), dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "IN_SYNC")

    async def test_body_change_produces_human_modified(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_j.txt", "source j text")
        dest_path = await self._seeded_note(ob, source_path, "line one\nline two\nline three")
        content = ob.files[dest_path]
        fm, body = parse_frontmatter(content)
        ob.files[dest_path] = content.replace(body.strip(), "line one\nline TWO EDITED\nline three")
        state = {source_path: make_state_entry(source_path, "line one\nline two\nline three",
                                                source_sha256=sha256_of("source j text"), dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "HUMAN_MODIFIED")

    async def test_readable_unified_diff_is_generated(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_k.txt", "source k text")
        dest_path = await self._seeded_note(ob, source_path, "alpha\nbeta\ngamma")
        content = ob.files[dest_path]
        fm, body = parse_frontmatter(content)
        ob.files[dest_path] = content.replace(body.strip(), "alpha\nBETA CHANGED\ngamma")
        state = {source_path: make_state_entry(source_path, "alpha\nbeta\ngamma",
                                                source_sha256=sha256_of("source k text"), dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state)
        diff_text = "\n".join(results[0]["diff"])
        self.assertIn("-beta", diff_text)
        self.assertIn("+BETA CHANGED", diff_text)
        self.assertFalse(results[0]["diff_truncated"])

    async def test_diff_truncation_behaves_predictably(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_l.txt", "source l text")
        original = "\n".join(f"line {i}" for i in range(300))
        dest_path = await self._seeded_note(ob, source_path, original)
        content = ob.files[dest_path]
        fm, body = parse_frontmatter(content)
        edited = "\n".join(f"CHANGED line {i}" for i in range(300))
        ob.files[dest_path] = content.replace(body.strip(), edited)
        state = {source_path: make_state_entry(source_path, original,
                                                source_sha256=sha256_of("source l text"), dest_path=dest_path)}
        results = await analyze_managed_notes(ob, state, max_diff_lines=10)
        self.assertTrue(results[0]["diff_truncated"])
        self.assertEqual(len(results[0]["diff"]), 10)
        self.assertGreater(results[0]["diff_total_lines"], 10)

    async def test_analyzer_does_not_modify_vault_note(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_m.txt", "source m text")
        dest_path = await self._seeded_note(ob, source_path, "content that must survive analysis untouched")
        before = ob.files[dest_path]
        writes_after_seeding = len(ob.write_calls)
        state = {source_path: make_state_entry(source_path, "content that must survive analysis untouched",
                                                source_sha256=sha256_of("source m text"), dest_path=dest_path)}
        await analyze_managed_notes(ob, state)
        self.assertEqual(ob.files[dest_path], before)
        self.assertEqual(len(ob.write_calls), writes_after_seeding)  # no additional write during analysis

    async def test_analyzer_does_not_modify_source_file(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_n.txt", "original untouched source content")
        before_bytes = Path(source_path).read_bytes()
        dest_path = await self._seeded_note(ob, source_path, "generated body")
        state = {source_path: make_state_entry(source_path, "generated body",
                                                source_sha256=sha256_of("original untouched source content"),
                                                dest_path=dest_path)}
        await analyze_managed_notes(ob, state)
        self.assertEqual(Path(source_path).read_bytes(), before_bytes)

    async def test_analyzer_does_not_modify_sync_state_dict(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_o.txt", "source o text")
        dest_path = await self._seeded_note(ob, source_path, "stable content")
        state = {source_path: make_state_entry(source_path, "stable content",
                                                source_sha256=sha256_of("source o text"), dest_path=dest_path)}
        import copy
        state_before = copy.deepcopy(state)
        await analyze_managed_notes(ob, state)
        self.assertEqual(state, state_before)

    async def test_multiple_managed_notes_analyzed_in_one_run(self):
        ob = FakeObsidian()
        state = {}
        for i in range(3):
            source_path = self.make_source(f"doc_multi_{i}.txt", f"source multi {i}")
            dest_path = await self._seeded_note(ob, source_path, f"content {i}")
            state[source_path] = make_state_entry(source_path, f"content {i}",
                                                   source_sha256=sha256_of(f"source multi {i}"), dest_path=dest_path)
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r["classification"] == "IN_SYNC" for r in results))

    async def test_one_invalid_note_does_not_prevent_analysis_of_others(self):
        ob = FakeObsidian()
        good_source = self.make_source("doc_good.txt", "good source text")
        good_dest = await self._seeded_note(ob, good_source, "good content")
        bad_source = self.make_source("doc_bad.txt", "bad source text")
        bad_dest = f"Knowledge Cortex/Managed/doc_bad-{source_id_for(bad_source)}.md"
        ob.files[bad_dest] = "---\ncortex_managed: true\n---\n\nmalformed, missing fields"
        state = {
            good_source: make_state_entry(good_source, "good content",
                                           source_sha256=sha256_of("good source text"), dest_path=good_dest),
            bad_source: make_state_entry(bad_source, "some body",
                                          source_sha256=sha256_of("bad source text"), dest_path=bad_dest),
        }
        results = await analyze_managed_notes(ob, state)
        classifications = {r["source_path"]: r["classification"] for r in results}
        self.assertEqual(classifications[good_source], "IN_SYNC")
        self.assertEqual(classifications[bad_source], "INVALID_MANAGED_NOTE")

    async def test_legacy_state_without_generated_body_classifies_clearly(self):
        ob = FakeObsidian()
        source_path = self.make_source("doc_legacy.txt", "source legacy text")
        dest_path = await self._seeded_note(ob, source_path, "some body")
        entry = make_state_entry(source_path, "some body",
                                  source_sha256=sha256_of("source legacy text"), dest_path=dest_path)
        del entry["generated_body"]  # simulate pre-hardening-pass state
        state = {source_path: entry}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results[0]["classification"], "ANALYSIS_INSUFFICIENT_STATE")

    async def test_entries_never_write_vaulted_are_skipped_entirely(self):
        ob = FakeObsidian()
        # An entry from a plain --index run (no --write-vault ever) has no
        # "vault_write" key -- it is not a "managed note" and must not be
        # reported as MISSING or anything else; it's simply out of scope.
        state = {"/tmp/never_write_vaulted.txt": {
            "source_sha256": "x", "ai_structure_requested": False, "ai_structure_succeeded": False,
            "structure_model": None, "prompt_version": None, "generated_body": "body",
        }}
        results = await analyze_managed_notes(ob, state)
        self.assertEqual(results, [])

    async def test_source_filter_analyzes_only_the_requested_source(self):
        ob = FakeObsidian()
        state = {}
        paths = []
        for i in range(2):
            source_path = self.make_source(f"doc_filter_{i}.txt", f"source filter {i}")
            paths.append(source_path)
            dest_path = await self._seeded_note(ob, source_path, f"content {i}")
            state[source_path] = make_state_entry(source_path, f"content {i}",
                                                   source_sha256=sha256_of(f"source filter {i}"), dest_path=dest_path)
        results = await analyze_managed_notes(ob, state, source_filter=paths[1])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_path"], paths[1])


if __name__ == "__main__":
    unittest.main()
