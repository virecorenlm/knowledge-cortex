import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ingest.proposals import (
    compute_proposal_id, create_proposals, load_proposal, list_proposals,
    approve_proposal, reject_proposal, SCHEMA_VERSION,
)


def make_analyzer_result(classification="HUMAN_MODIFIED", source_path="/tmp/doc.txt",
                          managed_note_path="Knowledge Cortex/Managed/doc-abc123.md",
                          source_sha256="src111", expected_generated_sha256="exp222",
                          current_vault_sha256="vault333", proposed_action="review_human_changes",
                          diff=None, reason="test reason", ai_structured=False, structure_model=None,
                          live_body="the current vault body text"):
    return {
        "classification": classification,
        "proposed_action": proposed_action,
        "source_path": source_path,
        "managed_note_path": managed_note_path,
        "current_source_sha256": source_sha256,
        "recorded_source_sha256": source_sha256,
        "expected_generated_sha256": expected_generated_sha256,
        "current_vault_sha256": current_vault_sha256,
        "ai_structured": ai_structured,
        "structure_model": structure_model,
        "diff": diff or ["--- cortex_generated", "+++ vault_current", "@@ -1 +1 @@", "-old", "+new"],
        "diff_truncated": False,
        "diff_total_lines": 5,
        "reason": reason,
        "_live_vault_body": live_body,
    }


class FakeObsidianForProposals:
    """Minimal fake matching what ingest.proposals._current_fingerprints needs:
    list_dir (via note_exists) and read_note."""

    def __init__(self, files=None):
        self.files = dict(files or {})

    async def list_dir(self, path=""):
        prefix = f"{path}/" if path else ""
        names = set()
        for p in self.files:
            if not p.startswith(prefix):
                continue
            rest = p[len(prefix):]
            names.add(rest.split("/")[0] if "/" not in rest else rest.split("/")[0] + "/")
        return sorted(names)

    async def read_note(self, path):
        return {"content": self.files[path], "path": path}


class ProposalIdentityTests(unittest.TestCase):
    def test_id_is_deterministic_for_identical_material_facts(self):
        a = compute_proposal_id("/tmp/x.txt", "note.md", "HUMAN_MODIFIED", "review_human_changes",
                                 "s1", "e1", "v1")
        b = compute_proposal_id("/tmp/x.txt", "note.md", "HUMAN_MODIFIED", "review_human_changes",
                                 "s1", "e1", "v1")
        self.assertEqual(a, b)

    def test_id_changes_when_source_hash_changes(self):
        a = compute_proposal_id("/tmp/x.txt", "note.md", "HUMAN_MODIFIED", "review_human_changes",
                                 "s1", "e1", "v1")
        b = compute_proposal_id("/tmp/x.txt", "note.md", "HUMAN_MODIFIED", "review_human_changes",
                                 "s2", "e1", "v1")
        self.assertNotEqual(a, b)

    def test_id_changes_when_vault_hash_changes(self):
        a = compute_proposal_id("/tmp/x.txt", "note.md", "HUMAN_MODIFIED", "review_human_changes",
                                 "s1", "e1", "v1")
        b = compute_proposal_id("/tmp/x.txt", "note.md", "HUMAN_MODIFIED", "review_human_changes",
                                 "s1", "e1", "v2")
        self.assertNotEqual(a, b)

    def test_id_changes_when_classification_changes(self):
        a = compute_proposal_id("/tmp/x.txt", "note.md", "HUMAN_MODIFIED", "review_human_changes",
                                 "s1", "e1", "v1")
        b = compute_proposal_id("/tmp/x.txt", "note.md", "SOURCE_CHANGED", "source_changed_reprocess_required",
                                 "s1", "e1", "v1")
        self.assertNotEqual(a, b)


class CreateProposalsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.proposals_dir = Path(self._tmp.name) / "proposals"

    def test_human_modified_generates_a_pending_proposal(self):
        results = [make_analyzer_result(classification="HUMAN_MODIFIED")]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["status"], "pending")
        self.assertTrue(created[0]["created"])

    def test_in_sync_generates_no_proposal(self):
        results = [make_analyzer_result(classification="IN_SYNC", proposed_action="none")]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        self.assertEqual(created, [])
        self.assertEqual(list(self.proposals_dir.glob("*.json")) if self.proposals_dir.exists() else [], [])

    def test_analysis_insufficient_state_generates_no_proposal(self):
        results = [make_analyzer_result(classification="ANALYSIS_INSUFFICIENT_STATE",
                                         proposed_action="reprocess_required")]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        self.assertEqual(created, [])

    def test_proposal_json_round_trips_correctly(self):
        results = [make_analyzer_result()]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        proposal, error = load_proposal(self.proposals_dir, created[0]["proposal_id"])
        self.assertIsNone(error)
        self.assertEqual(proposal["schema_version"], SCHEMA_VERSION)
        self.assertEqual(proposal["classification"], "HUMAN_MODIFIED")
        self.assertEqual(proposal["source_path"], "/tmp/doc.txt")
        self.assertEqual(proposal["fingerprints"]["source_sha256"], "src111")

    def test_proposal_writes_are_atomic_no_tmp_file_left_behind(self):
        results = [make_analyzer_result()]
        create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        files = list(self.proposals_dir.iterdir())
        self.assertTrue(all(f.suffix == ".json" for f in files))
        self.assertTrue(all(".tmp" not in f.name for f in files))

    def test_exact_reviewed_body_is_preserved_for_future_apply(self):
        results = [make_analyzer_result(live_body="the EXACT body a human reviewed")]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        proposal, _ = load_proposal(self.proposals_dir, created[0]["proposal_id"])
        self.assertEqual(proposal["reviewed_live_vault_body"], "the EXACT body a human reviewed")

    def test_regenerating_identical_situation_does_not_duplicate_or_reset(self):
        results = [make_analyzer_result()]
        first = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        # Approve it, then "re-analyze" the same unchanged situation.
        obsidian = FakeObsidianForProposals(files={
            "Knowledge Cortex/Managed/doc-abc123.md": "irrelevant, approval doesn't re-read this path"
        })
        second = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0]["proposal_id"], first[0]["proposal_id"])
        self.assertTrue(second[0]["already_existed"])
        self.assertFalse(second[0]["created"])
        # Only one file on disk, not two.
        self.assertEqual(len(list(self.proposals_dir.glob("*.json"))), 1)

    def test_multiple_proposals_do_not_interfere(self):
        results = [
            make_analyzer_result(source_path="/tmp/a.txt", managed_note_path="Knowledge Cortex/Managed/a.md"),
            make_analyzer_result(source_path="/tmp/b.txt", managed_note_path="Knowledge Cortex/Managed/b.md",
                                  classification="SOURCE_CHANGED", proposed_action="source_changed_reprocess_required"),
        ]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        self.assertEqual(len(created), 2)
        self.assertNotEqual(created[0]["proposal_id"], created[1]["proposal_id"])
        self.assertEqual(len(list(self.proposals_dir.glob("*.json"))), 2)


class LoadListMalformedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.proposals_dir = Path(self._tmp.name) / "proposals"
        self.proposals_dir.mkdir(parents=True)

    def test_malformed_proposal_file_fails_safely(self):
        (self.proposals_dir / "badjson.json").write_text("{not valid json", encoding="utf-8")
        proposal, error = load_proposal(self.proposals_dir, "badjson")
        self.assertIsNone(proposal)
        self.assertIn("not valid JSON", error)

    def test_missing_proposal_file_fails_safely(self):
        proposal, error = load_proposal(self.proposals_dir, "doesnotexist")
        self.assertIsNone(proposal)
        self.assertIn("no proposal found", error)

    def test_unsupported_schema_version_fails_safely(self):
        (self.proposals_dir / "futureversion.json").write_text(
            json.dumps({"schema_version": 999, "proposal_id": "futureversion",
                        "status": "pending", "source_path": "/tmp/x", "fingerprints": {}}),
            encoding="utf-8",
        )
        proposal, error = load_proposal(self.proposals_dir, "futureversion")
        self.assertIsNone(proposal)
        self.assertIn("unsupported schema_version", error)

    def test_missing_required_field_fails_safely(self):
        (self.proposals_dir / "incomplete.json").write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "proposal_id": "incomplete"}),
            encoding="utf-8",
        )
        proposal, error = load_proposal(self.proposals_dir, "incomplete")
        self.assertIsNone(proposal)
        self.assertIn("missing required field", error)

    def test_listing_includes_malformed_entries_with_their_error(self):
        (self.proposals_dir / "bad.json").write_text("not json at all", encoding="utf-8")
        results = list(self.proposals_dir.glob("*.json"))
        self.assertEqual(len(results), 1)
        entries = list_proposals(self.proposals_dir)
        self.assertEqual(len(entries), 1)
        proposal, error = entries[0]
        self.assertIsNone(proposal)
        self.assertIsNotNone(error)

    def test_listing_nonexistent_directory_returns_empty(self):
        entries = list_proposals(Path(self.proposals_dir) / "does_not_exist_at_all")
        self.assertEqual(entries, [])


class ApprovalRejectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.proposals_dir = Path(self._tmp.name) / "proposals"
        self.source_dir = Path(self._tmp.name) / "sources"
        self.source_dir.mkdir(parents=True)
        self.state_path = Path(self._tmp.name) / "sync_state.json"

    def make_source(self, name, content):
        p = self.source_dir / name
        p.write_text(content, encoding="utf-8")
        return str(p.resolve())

    def sha(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def write_state(self, source_path, generated_body, extra=None):
        """Write a disposable sync_state.json 'local' namespace entry
        matching what ingest.sync.index_local_path would have recorded,
        so approve_proposal's generated-baseline check has something real
        to verify against."""
        import sync_cli
        entry = {"source_sha256": self.sha("irrelevant-for-this-check"),
                 "ai_structure_requested": False, "ai_structure_succeeded": False,
                 "structure_model": None, "prompt_version": None,
                 "generated_body": generated_body}
        if extra:
            entry.update(extra)
        sync_cli.save_state(self.state_path, {source_path: entry}, namespace="local")

    async def test_approval_succeeds_when_fingerprints_unchanged(self):
        source_path = self.make_source("doc.txt", "unchanged source content")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "the unchanged live vault body"
        generated_body = "the cortex-generated baseline body"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(
            source_path=source_path, managed_note_path=note_path,
            source_sha256=self.sha("unchanged source content"),
            expected_generated_sha256=self.sha(generated_body),
            current_vault_sha256=self.sha(vault_body), live_body=vault_body,
        )]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        result = await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian,
                                         state_path=self.state_path)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "approved")

    async def test_approval_only_changes_proposal_metadata(self):
        source_path = self.make_source("doc.txt", "content x")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body x"
        generated_body = "generated baseline x"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(
            source_path=source_path, managed_note_path=note_path,
            source_sha256=self.sha("content x"), expected_generated_sha256=self.sha(generated_body),
            current_vault_sha256=self.sha(vault_body), live_body=vault_body,
        )]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        vault_before = dict(obsidian.files)
        source_before = Path(source_path).read_bytes()
        state_before = self.state_path.read_bytes()

        await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian, state_path=self.state_path)

        self.assertEqual(obsidian.files, vault_before)  # no vault write
        self.assertEqual(Path(source_path).read_bytes(), source_before)  # no source write
        self.assertEqual(self.state_path.read_bytes(), state_before)  # no state write
        proposal, _ = load_proposal(self.proposals_dir, created[0]["proposal_id"])
        self.assertEqual(proposal["status"], "approved")
        self.assertEqual(proposal["decision"]["status"], "approved")

    async def test_approval_does_not_modify_source(self):
        source_path = self.make_source("doc.txt", "content y")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body y"
        generated_body = "generated baseline y"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("content y"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        before = Path(source_path).read_bytes()
        await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian, state_path=self.state_path)
        self.assertEqual(Path(source_path).read_bytes(), before)

    async def test_approval_does_not_modify_vault(self):
        source_path = self.make_source("doc.txt", "content z")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body z"
        generated_body = "generated baseline z"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("content z"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        content = f"---\nx: 1\n---\n\n{vault_body}"
        obsidian = FakeObsidianForProposals(files={note_path: content})
        await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian, state_path=self.state_path)
        self.assertEqual(obsidian.files[note_path], content)

    async def test_approval_does_not_touch_sync_state_or_qdrant(self):
        from unittest.mock import patch
        # Runtime guards around the actual approval are also asserted in
        # test_proposal_hardening; here exercise the original full metadata test.
        with patch("graph.store.VectorStore", side_effect=AssertionError("Qdrant store construction")), \
             patch("qdrant_client.QdrantClient", side_effect=AssertionError("Qdrant client construction")):
            await self.test_approval_only_changes_proposal_metadata()

    async def test_changed_source_before_approval_becomes_stale(self):
        source_path = self.make_source("doc.txt", "original content")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body unchanged"
        generated_body = "generated baseline unchanged"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("original content"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        Path(source_path).write_text("CHANGED content after proposal creation", encoding="utf-8")
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        result = await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian,
                                         state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")
        proposal, _ = load_proposal(self.proposals_dir, created[0]["proposal_id"])
        self.assertEqual(proposal["status"], "stale")

    async def test_changed_vault_body_before_approval_becomes_stale(self):
        source_path = self.make_source("doc.txt", "stable content")
        note_path = "Knowledge Cortex/Managed/doc.md"
        generated_body = "generated baseline stable"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("stable content"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha("original vault body"),
                                         live_body="original vault body")]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        obsidian = FakeObsidianForProposals(files={note_path: "---\nx: 1\n---\n\nSOMEONE EDITED THIS"})
        result = await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian,
                                         state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")

    async def test_missing_source_before_approval_becomes_stale(self):
        source_path = str(self.source_dir / "will_be_deleted.txt")
        Path(source_path).write_text("temp content", encoding="utf-8")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body"
        generated_body = "generated baseline temp"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("temp content"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        Path(source_path).unlink()
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        result = await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian,
                                         state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")

    async def test_missing_note_before_approval_becomes_stale(self):
        source_path = self.make_source("doc.txt", "content w")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body w"
        generated_body = "generated baseline w"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("content w"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        obsidian = FakeObsidianForProposals(files={})  # note deleted
        result = await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian,
                                         state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")

    async def test_stale_proposal_cannot_become_approved(self):
        source_path = self.make_source("doc.txt", "content v")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body v"
        generated_body = "generated baseline v"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("content v"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        Path(source_path).write_text("changed to force staleness", encoding="utf-8")
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian,
                                state_path=self.state_path)  # -> stale

        # Restore original content and try again -- a stale proposal must
        # not be retroactively approvable; it stays stale forever.
        Path(source_path).write_text("content v", encoding="utf-8")
        result = await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian,
                                         state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")

    async def test_rejection_works_even_if_live_inputs_changed(self):
        source_path = self.make_source("doc.txt", "content u")
        note_path = "Knowledge Cortex/Managed/doc.md"
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("content u"))]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        Path(source_path).write_text("completely different now", encoding="utf-8")
        result = reject_proposal(self.proposals_dir, created[0]["proposal_id"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "rejected")

    async def test_approved_proposal_cannot_silently_become_rejected(self):
        source_path = self.make_source("doc.txt", "content t")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body t"
        generated_body = "generated baseline t"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("content t"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian, state_path=self.state_path)
        result = reject_proposal(self.proposals_dir, created[0]["proposal_id"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "approved")

    async def test_rejected_proposal_cannot_silently_become_approved(self):
        source_path = self.make_source("doc.txt", "content s")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "vault body s"
        generated_body = "generated baseline s"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("content s"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        reject_proposal(self.proposals_dir, created[0]["proposal_id"])
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        result = await approve_proposal(self.proposals_dir, created[0]["proposal_id"], obsidian,
                                         state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "rejected")


class GeneratedBaselineStalenessTests(unittest.IsolatedAsyncioTestCase):
    """Hardening pass: approval must also verify Cortex's own generated
    baseline (sync_state.json's cached generated_body for the EXACT
    source path) has not drifted since the proposal was created."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.proposals_dir = Path(self._tmp.name) / "proposals"
        self.source_dir = Path(self._tmp.name) / "sources"
        self.source_dir.mkdir(parents=True)
        self.state_path = Path(self._tmp.name) / "sync_state.json"

    def make_source(self, name, content):
        p = self.source_dir / name
        p.write_text(content, encoding="utf-8")
        return str(p.resolve())

    def sha(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def write_state(self, source_path, generated_body):
        import sync_cli
        entry = {"source_sha256": "irrelevant", "ai_structure_requested": False,
                  "ai_structure_succeeded": False, "structure_model": None,
                  "prompt_version": None, "generated_body": generated_body}
        sync_cli.save_state(self.state_path, {source_path: entry}, namespace="local")

    def _setup(self, source_content="src", vault_body="vault", generated_body="generated"):
        source_path = self.make_source("doc.txt", source_content)
        note_path = "Knowledge Cortex/Managed/doc.md"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha(source_content),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        return source_path, note_path, created[0]["proposal_id"], obsidian

    async def test_unchanged_source_vault_and_generated_state_approves(self):
        source_path, note_path, proposal_id, obsidian = self._setup()
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "approved")

    async def test_generated_state_changed_after_proposal_creation_becomes_stale(self):
        source_path, note_path, proposal_id, obsidian = self._setup()
        # Cortex re-runs and produces a DIFFERENT generated body for the
        # same source, without the vault or source file itself changing.
        self.write_state(source_path, "a completely different generated baseline")
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")
        self.assertIn("generated baseline", result["reason"])

    async def test_generated_state_entry_disappears_becomes_stale(self):
        source_path, note_path, proposal_id, obsidian = self._setup()
        # Simulate the state entry vanishing entirely (e.g. state file
        # regenerated without this source).
        import sync_cli
        sync_cli.save_state(self.state_path, {}, namespace="local")
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")
        self.assertIn("unverifiable", result["reason"])

    async def test_generated_state_entry_malformed_becomes_stale(self):
        source_path, note_path, proposal_id, obsidian = self._setup()
        import sync_cli
        # Malformed: a plain string instead of an object (legacy pre-
        # AI-structure format, which has no generated_body at all).
        sync_cli.save_state(self.state_path, {source_path: "just-a-sha256-string"}, namespace="local")
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")
        self.assertIn("unverifiable", result["reason"])

    async def test_generated_state_entry_for_different_source_identity_not_accepted(self):
        # A different source path happening to have a matching generated
        # hash must NOT satisfy the check -- identity is by exact path.
        source_path, note_path, proposal_id, obsidian = self._setup(generated_body="shared-baseline-text")
        import sync_cli
        other_source = str(self.source_dir / "other.txt")
        # Replace the state entry keyed on source_path with one for a
        # DIFFERENT path (simulates the real entry disappearing while an
        # unrelated entry happens to have the same generated_body/hash).
        entry = {"source_sha256": "irrelevant", "ai_structure_requested": False,
                  "ai_structure_succeeded": False, "structure_model": None,
                  "prompt_version": None, "generated_body": "shared-baseline-text"}
        sync_cli.save_state(self.state_path, {other_source: entry}, namespace="local")
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "stale")
        self.assertIn("unverifiable", result["reason"])

    async def test_generated_baseline_check_never_writes_state(self):
        source_path, note_path, proposal_id, obsidian = self._setup()
        state_before = self.state_path.read_bytes()
        await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertEqual(self.state_path.read_bytes(), state_before)

        # Also verify the stale path (a real write attempt) doesn't touch state.
        self.write_state(source_path, "different now")
        state_before_stale_attempt = self.state_path.read_bytes()
        await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertEqual(self.state_path.read_bytes(), state_before_stale_attempt)


class ProposalInternalConsistencyTests(unittest.IsolatedAsyncioTestCase):
    """Hardening pass: approval must fail closed on a proposal whose own
    fields are internally inconsistent (accidental/manual edits), not
    silently trust or 'repair' it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.proposals_dir = Path(self._tmp.name) / "proposals"
        self.source_dir = Path(self._tmp.name) / "sources"
        self.source_dir.mkdir(parents=True)
        self.state_path = Path(self._tmp.name) / "sync_state.json"

    def make_source(self, name, content):
        p = self.source_dir / name
        p.write_text(content, encoding="utf-8")
        return str(p.resolve())

    def sha(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def write_state(self, source_path, generated_body):
        import sync_cli
        entry = {"source_sha256": "irrelevant", "ai_structure_requested": False,
                  "ai_structure_succeeded": False, "structure_model": None,
                  "prompt_version": None, "generated_body": generated_body}
        sync_cli.save_state(self.state_path, {source_path: entry}, namespace="local")

    def _create_valid_proposal(self):
        source_path = self.make_source("doc.txt", "source content")
        note_path = "Knowledge Cortex/Managed/doc.md"
        vault_body = "the live vault body"
        generated_body = "the generated baseline"
        self.write_state(source_path, generated_body)
        results = [make_analyzer_result(source_path=source_path, managed_note_path=note_path,
                                         source_sha256=self.sha("source content"),
                                         expected_generated_sha256=self.sha(generated_body),
                                         current_vault_sha256=self.sha(vault_body), live_body=vault_body)]
        created = create_proposals(results, proposals_dir=self.proposals_dir, state_path=getattr(self, "state_path", None),
            local_state=json.loads(self.state_path.read_text())["local"] if hasattr(self, "state_path") and self.state_path.exists() else None)
        obsidian = FakeObsidianForProposals(files={note_path: f"---\nx: 1\n---\n\n{vault_body}"})
        return created[0]["proposal_id"], obsidian

    async def test_valid_proposal_round_trip_still_approves(self):
        proposal_id, obsidian = self._create_valid_proposal()
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "approved")

    async def test_manually_altered_proposal_id_is_refused(self):
        proposal_id, obsidian = self._create_valid_proposal()
        path = self.proposals_dir / f"{proposal_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["proposal_id"] = "0000000000000000"  # tampered
        path.write_text(json.dumps(data), encoding="utf-8")
        # Must be looked up by its (now-mismatched) filename id.
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertIn("does not match its own material fields", result["reason"])
        # Must remain exactly as found -- not relabeled to "stale" either.
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["status"], "pending")

    async def test_manually_altered_fingerprint_without_matching_id_is_refused(self):
        proposal_id, obsidian = self._create_valid_proposal()
        path = self.proposals_dir / f"{proposal_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["fingerprints"]["source_sha256"] = "tampered-hash-value"  # id no longer matches
        path.write_text(json.dumps(data), encoding="utf-8")
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertIn("does not match its own material fields", result["reason"])

    async def test_altered_reviewed_body_without_matching_hash_is_refused(self):
        proposal_id, obsidian = self._create_valid_proposal()
        path = self.proposals_dir / f"{proposal_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["reviewed_live_vault_body"] = "this text was never actually reviewed"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertIn("reviewed_live_vault_body does not hash", result["reason"])

    async def test_invalid_fingerprint_type_is_refused(self):
        proposal_id, obsidian = self._create_valid_proposal()
        path = self.proposals_dir / f"{proposal_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["fingerprints"]["source_sha256"] = 12345  # wrong type
        path.write_text(json.dumps(data), encoding="utf-8")
        result = await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertIn("invalid type", result["reason"])

    async def test_consistency_failure_touches_no_external_system(self):
        proposal_id, obsidian = self._create_valid_proposal()
        path = self.proposals_dir / f"{proposal_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["proposal_id"] = "tampered0000000"
        path.write_text(json.dumps(data), encoding="utf-8")
        vault_before = dict(obsidian.files)
        state_before = self.state_path.read_bytes()
        await approve_proposal(self.proposals_dir, proposal_id, obsidian, state_path=self.state_path)
        self.assertEqual(obsidian.files, vault_before)
        self.assertEqual(self.state_path.read_bytes(), state_before)


if __name__ == "__main__":
    unittest.main()
