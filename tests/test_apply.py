"""Apply layer tests. Every authority is disposable: temp dirs, a fake
Obsidian client, a fake Ollama client, and an in-memory Qdrant."""
import asyncio
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ingest import apply as apply_module
from ingest.apply import ApplyRefused, apply_proposal, build_candidate, extracted_sha256, verify_written
from ingest.markdown import FRONTMATTER_RE
from ingest.proposals import approve_proposal, create_proposals, reject_proposal
from ingest.reverse_analyzer import _current_source_hash, analyze_managed_notes
from ingest.vault_writer import hash_managed_body
from main import apply_proposal_cli, index_to_qdrant, propose_vault_changes
from sync_cli import load_state
from tests.test_main_cli import make_store
from tests.test_proposals import make_analyzer_result
from tests.test_vault_writer import FakeObsidian

REPO = Path(__file__).resolve().parent.parent
MD_SOURCE = "---\ntitle: Keep Me\ntags: alpha\n---\n\nOriginal line one\nOriginal line two\n"
HUMAN_BODY = "# Human heading\n\nHuman edited content.\n"


def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class ApplyFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.input = self.root / "input"
        self.input.mkdir()
        self.state_path = self.root / "sync_state.json"
        self.proposals_dir = self.root / "state" / "proposals"
        self.backups_dir = self.root / "backups"
        self.store = make_store()
        self.ob = FakeObsidian()

    def seed(self, name="doc.md", content=MD_SOURCE, human_body=HUMAN_BODY, approve=True):
        """Real pipeline: ingest + write-vault, human edit, propose, approve."""
        self.source = self.input / name
        self.source.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
        _quiet(index_to_qdrant, str(self.source), None, self.state_path, store=self.store,
               write_vault=True, obsidian=self.ob)
        self.note_path = next(iter(self.ob.files))
        self.human_edit(human_body)
        buf = io.StringIO()
        with redirect_stdout(buf):
            propose_vault_changes(self.state_path, proposals_dir=self.proposals_dir, json_output=True,
                                  obsidian=self.ob)
        created = json.loads(buf.getvalue())
        self.assertEqual(created[0]["classification"], "HUMAN_MODIFIED")
        self.proposal_id = created[0]["proposal_id"]
        self.proposal_file = self.proposals_dir / f"{self.proposal_id}.json"
        if approve:
            result = asyncio.run(approve_proposal(self.proposals_dir, self.proposal_id, self.ob,
                                                  state_path=self.state_path))
            self.assertTrue(result["ok"], result)
        self.ob.write_calls.clear()
        return self.proposal_id

    def human_edit(self, body):
        content = self.ob.files[self.note_path]
        match = FRONTMATTER_RE.match(content)
        self.ob.files[self.note_path] = content[:match.end()] + body

    def apply(self, dry_run=False, **kwargs):
        kwargs.setdefault("backups_dir", self.backups_dir)
        return asyncio.run(apply_proposal(self.proposals_dir, self.proposal_id, self.ob,
                                          state_path=self.state_path, dry_run=dry_run, **kwargs))

    def proposal(self):
        return json.loads(self.proposal_file.read_text())

    def set_proposal(self, data):
        self.proposal_file.write_text(json.dumps(data))

    def points(self):
        points, _ = self.store.client.scroll(self.store.collection, limit=1000, with_payload=True,
                                             with_vectors=True)
        return sorted((str(p.id), json.dumps(p.payload, sort_keys=True), list(p.vector)) for p in points)

    def snapshot(self):
        files = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        return files, dict(self.ob.files), self.points()


class SuccessfulApplyTests(ApplyFixture):
    def test_markdown_applies_exact_reviewed_body_and_preserves_source_frontmatter(self):
        self.seed()
        result = self.apply()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(self.source.read_text(), "---\ntitle: Keep Me\ntags: alpha\n---\n\n" + HUMAN_BODY)
        self.assertTrue(result["apply"]["frontmatter_preserved"])

    def test_cortex_ownership_frontmatter_never_leaks_into_source(self):
        self.seed()
        self.assertIn("cortex_managed: true", self.ob.files[self.note_path])
        self.apply()
        self.assertNotIn("cortex_", self.source.read_text())

    def test_markdown_without_frontmatter_gets_body_verbatim(self):
        self.seed(content="Just text\nno frontmatter\n")
        self.assertTrue(self.apply()["ok"])
        self.assertEqual(self.source.read_text(), HUMAN_BODY)

    def test_txt_applies_exact_reviewed_text_without_any_frontmatter(self):
        self.seed(name="notes.txt", content="---\nnot: frontmatter in txt\n---\nplain\n",
                  human_body="Edited plain text.\nSecond line.\n")
        result = self.apply()
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.source.read_text(), "Edited plain text.\nSecond line.\n")
        self.assertFalse(result["apply"]["frontmatter_preserved"])

    def test_backup_contains_exact_pre_apply_bytes(self):
        self.seed()
        before = self.source.read_bytes()
        result = self.apply()
        backup = Path(result["apply"]["backup_path"])
        self.assertEqual(backup, self.backups_dir / self.proposal_id / "doc.md.pre-apply")
        self.assertEqual(backup.read_bytes(), before)
        self.assertEqual(stat.S_IMODE(backup.stat().st_mode), 0o600)

    def test_default_backup_dir_is_next_to_proposals_dir(self):
        self.seed()
        result = self.apply(backups_dir=None)
        self.assertEqual(Path(result["apply"]["backup_path"]).parent,
                         self.root / "state" / "apply_backups" / self.proposal_id)

    def test_existing_backup_is_never_overwritten(self):
        self.seed()
        old = self.backups_dir / self.proposal_id / "doc.md.pre-apply"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"an older backup")
        result = self.apply()
        self.assertEqual(old.read_bytes(), b"an older backup")
        self.assertEqual(result["apply"]["backup_path"], str(old) + ".1")

    def test_apply_records_metadata_without_mutating_reviewed_snapshot(self):
        self.seed()
        before = self.proposal()
        pre_bytes = self.source.read_bytes()
        result = self.apply()
        after = self.proposal()
        self.assertEqual(after["status"], "applied")
        for key in ("fingerprints", "reviewed_live_vault_body", "expected_generated_body", "decision",
                    "proposal_id", "classification", "created_at"):
            self.assertEqual(after[key], before[key], key)
        record = after["apply"]
        self.assertEqual(record, result["apply"])
        self.assertEqual(record["source_path"], str(self.source.resolve()))
        self.assertEqual(record["pre_apply_source_sha256"], before["fingerprints"]["source_sha256"])
        self.assertEqual(record["pre_apply_source_bytes_sha256"], hashlib.sha256(pre_bytes).hexdigest())
        self.assertEqual(record["post_apply_source_sha256"], _current_source_hash(str(self.source))[0])
        self.assertEqual(record["post_apply_source_bytes_sha256"],
                         hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertFalse(record["recovered"])
        self.assertIsNotNone(record["applied_at"])
        self.assertNotIn("apply_intent", after)

    def test_apply_does_not_touch_sync_state_qdrant_or_vault(self):
        self.seed()
        state_before = self.state_path.read_bytes()
        points_before = self.points()
        vault_before = dict(self.ob.files)
        self.assertTrue(self.apply()["ok"])
        self.assertEqual(self.state_path.read_bytes(), state_before)
        self.assertEqual(self.points(), points_before)
        self.assertEqual(self.ob.files, vault_before)
        self.assertEqual(self.ob.write_calls, [])

    def test_apply_module_never_imports_vector_store_or_writes_vault(self):
        text = Path(apply_module.__file__).read_text()
        self.assertNotIn("VectorStore", text.replace("VectorStore is\nnever imported", ""))
        self.assertNotIn("write_note", text)
        self.assertNotIn("save_state", text)

    def test_file_mode_is_preserved(self):
        self.seed()
        os.chmod(self.source, 0o640)
        self.assertTrue(self.apply()["ok"])
        self.assertEqual(stat.S_IMODE(self.source.stat().st_mode), 0o640)

    def test_no_temp_files_left_after_success(self):
        self.seed()
        self.apply()
        self.assertEqual(sorted(p.name for p in self.input.iterdir()), ["doc.md"])


class DriftAfterApplyTests(ApplyFixture):
    def test_analyzer_then_normal_ingestion_observe_source_drift(self):
        self.seed()
        self.apply()
        local = load_state(self.state_path, namespace="local")
        result = asyncio.run(analyze_managed_notes(self.ob, local))[0]
        # The vault still differs from the cached generated body, and the
        # source now differs from the recorded source hash.
        self.assertEqual(result["classification"], "HUMAN_MODIFIED")
        self.assertTrue(result["flags"]["source_changed"])

        code = _quiet(index_to_qdrant, str(self.source), None, self.state_path, store=self.store)
        self.assertEqual(code, 0)
        entry = load_state(self.state_path, namespace="local")[str(self.source.resolve())]
        self.assertEqual(entry["source_sha256"], _current_source_hash(str(self.source))[0])
        payload_texts = [json.loads(p[1])["text"] for p in self.points()]
        self.assertTrue(any("Human edited content." in t for t in payload_texts))
        self.assertFalse(any("Original line one" in t for t in payload_texts))
        paths = {json.loads(p[1])["path"] for p in self.points()}
        self.assertEqual(paths, {"local_ingest/doc.md"})  # no orphans under another path


class RefusalTests(ApplyFixture):
    def assert_refused_without_writes(self, status, dry_run=False):
        files, vault, points = self.snapshot()
        result = self.apply(dry_run=dry_run)
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["status"], status, result)
        new_files, new_vault, new_points = self.snapshot()
        source_key = str(self.source)
        self.assertEqual(new_files.get(source_key), files.get(source_key))
        self.assertEqual({k: v for k, v in new_files.items() if not k.endswith(".json")},
                         {k: v for k, v in files.items() if not k.endswith(".json")})
        self.assertEqual(new_vault, vault)
        self.assertEqual(new_points, points)
        self.assertFalse(self.backups_dir.exists())
        return result

    def test_pending_proposal_refuses(self):
        self.seed(approve=False)
        before = self.proposal_file.read_bytes()
        self.assert_refused_without_writes("not_approved")
        self.assertEqual(self.proposal_file.read_bytes(), before)

    def test_rejected_proposal_refuses(self):
        self.seed(approve=False)
        reject_proposal(self.proposals_dir, self.proposal_id)
        self.assert_refused_without_writes("rejected")

    def test_stale_proposal_refuses(self):
        self.seed(approve=False)
        self.human_edit("changed before approval")
        asyncio.run(approve_proposal(self.proposals_dir, self.proposal_id, self.ob, state_path=self.state_path))
        self.assertEqual(self.proposal()["status"], "stale")
        self.assert_refused_without_writes("stale")

    def test_source_changed_after_approval_refuses_and_marks_stale(self):
        self.seed()
        self.source.write_text(MD_SOURCE + "a later source edit\n")
        result = self.assert_refused_without_writes("stale")
        self.assertIn("source_sha256 changed", result["reason"])
        data = self.proposal()
        self.assertEqual(data["status"], "stale")
        self.assertEqual(data["decision"]["status"], "approved")  # history kept
        self.assertIn("source_sha256 changed", data["stale_on_apply"]["reasons"][0])
        self.assertEqual(self.apply()["status"], "stale")

    def test_vault_changed_after_approval_refuses(self):
        self.seed()
        self.human_edit("an even newer human edit")
        result = self.assert_refused_without_writes("stale")
        self.assertIn("live vault body changed", result["reason"])

    def test_managed_note_deleted_after_approval_refuses(self):
        self.seed()
        del self.ob.files[self.note_path]
        result = self.assert_refused_without_writes("stale")
        self.assertIn("managed note", result["reason"])

    def test_generated_baseline_changed_after_approval_refuses(self):
        self.seed()
        data = json.loads(self.state_path.read_text())
        data["local"][str(self.source.resolve())]["generated_body"] = "a regenerated baseline"
        self.state_path.write_text(json.dumps(data))
        result = self.assert_refused_without_writes("stale")
        self.assertIn("Cortex generated baseline changed", result["reason"])

    def test_generated_baseline_missing_refuses(self):
        self.seed()
        self.state_path.write_text(json.dumps({"local": {}}))
        result = self.assert_refused_without_writes("stale")
        self.assertIn("unverifiable", result["reason"])

    def test_source_deleted_after_approval_refuses(self):
        self.seed()
        self.source.unlink()
        result = self.apply()
        self.assertEqual(result["status"], "stale")
        self.assertFalse(self.source.exists())
        self.assertFalse(self.backups_dir.exists())

    def test_tampered_reviewed_body_refuses_and_leaves_proposal_unchanged(self):
        self.seed()
        data = self.proposal()
        data["reviewed_live_vault_body"] = "content nobody reviewed"
        self.set_proposal(data)
        before = self.proposal_file.read_bytes()
        result = self.assert_refused_without_writes("invalid_proposal")
        self.assertIn("reviewed_live_vault_body", result["reason"])
        self.assertEqual(self.proposal_file.read_bytes(), before)

    def test_tampered_records_refuse(self):
        self.seed()
        original = self.proposal()
        mutations = [
            ("fingerprints", {"source_sha256": "0" * 64}),
            ("classification", "SOURCE_CHANGED"),
            ("source_path", str(self.root / "elsewhere.md")),
            ("decision", {"status": None, "decided_at": None, "note": None}),
            ("schema_version", 2),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                data = json.loads(json.dumps(original))
                if isinstance(value, dict) and field == "fingerprints":
                    data[field].update(value)
                else:
                    data[field] = value
                self.set_proposal(data)
                result = self.apply()
                self.assertFalse(result["ok"])
                self.assertEqual(result["status"], "invalid_proposal", result)
                self.assertEqual(self.source.read_text(), MD_SOURCE)
        self.proposal_file.write_text("{not json")
        self.assertEqual(self.apply()["status"], "invalid_proposal")
        self.assertFalse(self.backups_dir.exists())

    def test_ai_structured_proposal_refuses(self):
        self.seed()
        data = self.proposal()
        data["ai_structured"] = True
        self.set_proposal(data)
        result = self.assert_refused_without_writes("ineligible_classification")
        self.assertIn("AI-structured", result["reason"])

    def test_state_path_mismatch_refuses(self):
        self.seed()
        other = self.root / "other_state.json"
        other.write_bytes(self.state_path.read_bytes())
        result = asyncio.run(apply_proposal(self.proposals_dir, self.proposal_id, self.ob, state_path=other,
                                            backups_dir=self.backups_dir))
        self.assertEqual(result["status"], "state_path_mismatch")
        self.assertEqual(self.source.read_text(), MD_SOURCE)

    def test_read_only_source_refuses(self):
        self.seed()
        os.chmod(self.source, 0o444)
        self.addCleanup(os.chmod, self.source, 0o644)
        self.assert_refused_without_writes("source_not_writable")

    def test_symlinked_source_refuses(self):
        self.seed()
        real = self.root / "real.md"
        real.write_bytes(self.source.read_bytes())
        self.source.unlink()
        self.source.symlink_to(real)
        result = self.apply()
        self.assertEqual(result["status"], "unsupported_source_file")
        self.assertTrue(self.source.is_symlink())
        self.assertEqual(real.read_text(), MD_SOURCE)

    def test_non_utf8_source_refuses(self):
        raw = MD_SOURCE.encode("utf-8") + b"\xff\xfe latin junk\n"
        self.seed(content=raw)
        self.assert_refused_without_writes("unsupported_source_encoding")
        self.assertEqual(self.source.read_bytes(), raw)


class IneligibleProposalTests(unittest.TestCase):
    """Proposals that are not HUMAN_MODIFIED .md/.txt are refused before any
    external read. They are built directly and force-approved in JSON, since
    the gate must hold even for records that claim approval."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.state_path = self.root / "state.json"
        self.proposals_dir = self.root / "proposals"
        self.backups_dir = self.root / "backups"

    def make(self, source_name, classification="HUMAN_MODIFIED", action="review_human_changes",
             live_body="human body"):
        source = self.root / source_name
        source.write_bytes(b"original source bytes")
        self.state_path.write_text(json.dumps({"local": {str(source): {"generated_body": "generated"}}}))
        result = make_analyzer_result(
            classification=classification, proposed_action=action, source_path=str(source),
            managed_note_path="test/note.md", source_sha256=hashlib.sha256(b"x").hexdigest(),
            expected_generated_sha256=hash_managed_body("generated"),
            current_vault_sha256=hash_managed_body(live_body) if live_body else None, live_body=live_body)
        created = create_proposals([result], local_state=json.loads(self.state_path.read_text())["local"],
                                   proposals_dir=self.proposals_dir, state_path=self.state_path)
        path = self.proposals_dir / f"{created[0]['proposal_id']}.json"
        data = json.loads(path.read_text())
        data["status"] = "approved"
        data["decision"] = {"status": "approved", "decided_at": "2026-01-01T00:00:00", "note": None}
        path.write_text(json.dumps(data))
        return source, created[0]["proposal_id"]

    def run_apply(self, proposal_id, dry_run=False):
        class ExplodingObsidian:
            def __getattr__(self, name):
                raise AssertionError(f"obsidian.{name} must not be called")
        return asyncio.run(apply_proposal(self.proposals_dir, proposal_id, ExplodingObsidian(),
                                          state_path=self.state_path, dry_run=dry_run,
                                          backups_dir=self.backups_dir))

    def test_unsupported_source_types_refuse_with_zero_writes(self):
        for name in ("doc.pdf", "doc.docx", "image.png", "blob.bin", "noextension"):
            with self.subTest(name=name):
                source, proposal_id = self.make(name)
                result = self.run_apply(proposal_id)
                self.assertEqual(result["status"], "unsupported_source_type", result)
                self.assertEqual(source.read_bytes(), b"original source bytes")
                self.assertFalse(self.backups_dir.exists())
                self.assertEqual(json.loads((self.proposals_dir / f"{proposal_id}.json").read_text())["status"],
                                 "approved")

    def test_non_human_modified_classifications_refuse(self):
        cases = [
            ("SOURCE_CHANGED", "source_changed_reprocess_required", "b"),
            ("SOURCE_MISSING", "source_missing_review_required", "b"),
            ("UNMANAGED_AT_TARGET", "investigate_provenance", "b"),
            ("INVALID_MANAGED_NOTE", "investigate_provenance", "b"),
            ("MISSING", "recreate_missing_managed_note", None),
            ("HUMAN_MODIFIED", "some_other_action", "b"),
        ]
        for classification, action, body in cases:
            with self.subTest(classification=classification, action=action):
                source, proposal_id = self.make(f"{classification}-{action}.md", classification, action, body)
                result = self.run_apply(proposal_id)
                self.assertEqual(result["status"], "ineligible_classification", result)
                self.assertEqual(source.read_bytes(), b"original source bytes")
        self.assertFalse(self.backups_dir.exists())

    def test_in_sync_and_insufficient_state_never_produce_applicable_proposals(self):
        for classification in ("IN_SYNC", "ANALYSIS_INSUFFICIENT_STATE"):
            result = make_analyzer_result(classification=classification)
            self.assertEqual(create_proposals([result], proposals_dir=self.proposals_dir,
                                              state_path=self.state_path), [])


class DryRunTests(ApplyFixture):
    def test_dry_run_reports_plan_and_writes_nothing(self):
        self.seed()
        before = self.snapshot()
        proposal_before = self.proposal_file.read_bytes()
        result = self.apply(dry_run=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["status"], "would_apply")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.proposal_file.read_bytes(), proposal_before)
        self.assertFalse(self.backups_dir.exists())
        self.assertEqual(self.ob.write_calls, [])
        self.assertEqual(result["backup_path"], str(self.backups_dir / self.proposal_id / "doc.md.pre-apply"))
        self.assertIn("+Human edited content.", result["diff"])
        self.assertIn("-Original line one", result["diff"])
        self.assertTrue(result["frontmatter_preserved"])

    def test_dry_run_expected_post_hash_matches_real_apply(self):
        self.seed()
        planned = self.apply(dry_run=True)
        applied = self.apply()
        self.assertEqual(planned["post_apply_source_sha256"], applied["apply"]["post_apply_source_sha256"])

    def test_dry_run_on_drift_reports_stale_without_mutating_proposal(self):
        self.seed()
        self.human_edit("newer edit")
        before = self.proposal_file.read_bytes()
        result = self.apply(dry_run=True)
        self.assertEqual(result["status"], "stale")
        self.assertEqual(self.proposal_file.read_bytes(), before)


class IdempotencyTests(ApplyFixture):
    def test_already_applied_is_a_no_op(self):
        self.seed()
        first = self.apply()
        content = self.source.read_bytes()
        mtime = self.source.stat().st_mtime_ns
        proposal_bytes = self.proposal_file.read_bytes()
        with patch.object(apply_module, "atomic_replace", side_effect=AssertionError("rewrite")):
            second = self.apply()
        self.assertTrue(second["ok"])
        self.assertEqual(second["status"], "already_applied")
        self.assertEqual(second["apply"], first["apply"])
        self.assertEqual(self.source.read_bytes(), content)
        self.assertEqual(self.source.stat().st_mtime_ns, mtime)
        self.assertEqual(self.proposal_file.read_bytes(), proposal_bytes)
        self.assertEqual(len(list((self.backups_dir / self.proposal_id).iterdir())), 1)

    def test_applied_proposal_cannot_be_approved_or_rejected(self):
        self.seed()
        self.apply()
        self.assertFalse(asyncio.run(approve_proposal(self.proposals_dir, self.proposal_id, self.ob,
                                                      state_path=self.state_path))["ok"])
        self.assertFalse(reject_proposal(self.proposals_dir, self.proposal_id)["ok"])
        self.assertEqual(self.proposal()["status"], "applied")


class FailureTests(ApplyFixture):
    def assert_source_intact_and_no_temp(self):
        self.assertEqual(self.source.read_text(), MD_SOURCE)
        self.assertEqual(sorted(p.name for p in self.input.iterdir()), ["doc.md"])

    def test_replace_failure_preserves_original_and_removes_temp(self):
        self.seed()
        real_replace = os.replace

        def failing_for_source(src, dst):
            if Path(dst) == self.source:
                raise OSError("disk full")
            return real_replace(src, dst)

        with patch.object(apply_module.os, "replace", side_effect=failing_for_source):
            result = self.apply()
        self.assertEqual(result["status"], "write_failed")
        self.assert_source_intact_and_no_temp()
        data = self.proposal()
        self.assertEqual(data["status"], "approved")
        self.assertNotIn("apply_intent", data)
        # A later retry works normally.
        self.assertEqual(self.apply()["status"], "applied")

    def test_fsync_failure_preserves_original(self):
        self.seed()
        real_fsync = os.fsync

        def flaky(fd):
            if os.readlink(f"/proc/self/fd/{fd}").endswith(".cortex-apply.tmp"):
                raise OSError("fsync failed")
            return real_fsync(fd)

        with patch.object(apply_module.os, "fsync", side_effect=flaky):
            result = self.apply()
        self.assertEqual(result["status"], "write_failed")
        self.assert_source_intact_and_no_temp()

    def test_read_only_directory_preserves_original(self):
        self.seed()
        os.chmod(self.input, 0o555)
        self.addCleanup(os.chmod, self.input, 0o755)
        result = self.apply()
        self.assertEqual(result["status"], "write_failed")
        self.assertEqual(self.source.read_text(), MD_SOURCE)

    def test_backup_failure_prevents_source_write(self):
        self.seed()
        with patch.object(apply_module, "create_backup", side_effect=OSError("no space")):
            result = self.apply()
        self.assertEqual(result["status"], "backup_failed")
        self.assertEqual(self.source.read_text(), MD_SOURCE)
        self.assertEqual(self.proposal()["status"], "approved")

    def test_intent_write_failure_prevents_source_write(self):
        self.seed()
        with patch.object(apply_module, "_atomic_write_json", side_effect=OSError("read-only fs")):
            result = self.apply()
        self.assertEqual(result["status"], "proposal_write_failed")
        self.assertEqual(self.source.read_text(), MD_SOURCE)

    def test_source_changed_between_validation_and_swap_is_not_overwritten(self):
        self.seed()
        real_mkstemp = apply_module.tempfile.mkstemp

        def racing_mkstemp(*args, **kwargs):
            self.source.write_text("a concurrent edit\n")
            return real_mkstemp(*args, **kwargs)

        with patch.object(apply_module.tempfile, "mkstemp", side_effect=racing_mkstemp):
            result = self.apply()
        self.assertEqual(result["status"], "stale")
        self.assertEqual(self.source.read_text(), "a concurrent edit\n")
        self.assertEqual(self.proposal()["status"], "stale")
        self.assertEqual(sorted(p.name for p in self.input.iterdir()), ["doc.md"])

    def test_post_write_verification_failure_is_reported(self):
        self.seed()
        real_replace = apply_module.atomic_replace

        def corrupting(target, data, expected):
            real_replace(target, data + b"corruption", expected)

        with patch.object(apply_module, "atomic_replace", side_effect=corrupting):
            result = self.apply()
        self.assertEqual(result["status"], "verification_failed")
        self.assertIn("VERIFICATION FAILED", result["reason"])
        self.assertIn(result["backup_path"], result["reason"])
        data = self.proposal()
        self.assertEqual(data["status"], "approved")
        self.assertIn("apply_intent", data)
        # Neither pre-apply nor approved bytes: never overwritten automatically.
        self.assertEqual(self.apply()["status"], "manual_review_required")


class PartialTransactionTests(ApplyFixture):
    def fail_final_proposal_write(self):
        real = apply_module._atomic_write_json
        calls = {"n": 0}

        def flaky(path, data):
            calls["n"] += 1
            if calls["n"] == 2:  # 1 = apply_intent, 2 = status "applied"
                raise OSError("proposal disk went away")
            return real(path, data)

        return patch.object(apply_module, "_atomic_write_json", side_effect=flaky)

    def test_metadata_failure_after_replace_is_reported_and_recoverable(self):
        self.seed()
        with self.fail_final_proposal_write():
            result = self.apply()
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "partial_failure")
        self.assertIn("PARTIAL TRANSACTION", result["reason"])
        applied_bytes = self.source.read_bytes()
        self.assertTrue(applied_bytes.endswith(HUMAN_BODY.encode()))
        data = self.proposal()
        self.assertEqual(data["status"], "approved")
        intent = data["apply_intent"]
        self.assertEqual(intent["candidate_bytes_sha256"], hashlib.sha256(applied_bytes).hexdigest())

        mtime = self.source.stat().st_mtime_ns
        dry = self.apply(dry_run=True)
        self.assertEqual(dry["status"], "would_recover")
        self.assertEqual(self.proposal()["status"], "approved")
        with patch.object(apply_module, "atomic_replace", side_effect=AssertionError("must not rewrite")), \
                patch.object(apply_module, "create_backup", side_effect=AssertionError("must not back up")):
            retry = self.apply()
        self.assertTrue(retry["ok"], retry)
        self.assertEqual(retry["status"], "recovered")
        self.assertEqual(retry["warnings"], [])
        self.assertEqual(self.source.read_bytes(), applied_bytes)
        self.assertEqual(self.source.stat().st_mtime_ns, mtime)
        data = self.proposal()
        self.assertEqual(data["status"], "applied")
        self.assertTrue(data["apply"]["recovered"])
        self.assertTrue(data["apply"]["frontmatter_preserved"])
        self.assertEqual(data["apply"]["backup_path"], intent["backup_path"])
        self.assertNotIn("apply_intent", data)
        self.assertEqual(self.apply()["status"], "already_applied")

    def test_retry_after_partial_failure_with_foreign_source_edit_refuses(self):
        self.seed()
        with self.fail_final_proposal_write():
            self.apply()
        self.source.write_text("someone edited it after the partial apply\n")
        result = self.apply()
        self.assertEqual(result["status"], "manual_review_required")
        self.assertEqual(self.source.read_text(), "someone edited it after the partial apply\n")
        self.assertEqual(self.proposal()["status"], "approved")

    def test_intent_without_replace_reruns_full_validation(self):
        self.seed()
        with patch.object(apply_module, "atomic_replace", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.apply()
        self.assertIn("apply_intent", self.proposal())
        self.assertEqual(self.source.read_text(), MD_SOURCE)
        self.human_edit("drifted after the interruption")
        self.assertEqual(self.apply()["status"], "stale")
        self.assertEqual(self.source.read_text(), MD_SOURCE)

    def test_recovery_records_vault_drift_as_warning(self):
        self.seed()
        with self.fail_final_proposal_write():
            self.apply()
        self.human_edit("edited after the source was already written")
        result = self.apply()
        self.assertEqual(result["status"], "recovered")
        self.assertTrue(any("live vault body changed" in w for w in result["warnings"]))


class CandidateUnitTests(unittest.TestCase):
    def build(self, suffix, source, body):
        return build_candidate(suffix, source, body, hash_managed_body(body.replace("\r\n", "\n")))

    def test_crlf_and_bom_are_carried_over(self):
        source = b"\xef\xbb\xbf---\r\ntitle: x\r\n---\r\n\r\nold\r\n"
        candidate, info = self.build(".md", source, "new\nbody\n")
        self.assertEqual(candidate, b"\xef\xbb\xbf---\r\ntitle: x\r\n---\r\n\r\nnew\r\nbody\r\n")
        self.assertTrue(info["crlf"] and info["bom"] and info["frontmatter_preserved"])
        self.assertEqual(verify_written(candidate, candidate, info, hash_managed_body("new\nbody\n")), [])

    def test_expected_post_hash_uses_the_extraction_contract(self):
        candidate, info = self.build(".md", b"---\r\na: b\r\n---\r\nold\r\n", "x\ny\n")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.md"
            path.write_bytes(candidate)
            self.assertEqual(_current_source_hash(str(path))[0], info["expected_post_apply_source_sha256"])
            self.assertEqual(extracted_sha256(candidate), info["expected_post_apply_source_sha256"])

    def test_cortex_frontmatter_in_body_is_refused(self):
        body = "---\ncortex_managed: true\ncortex_source_id: abc\n---\n\ncontent\n"
        with self.assertRaises(ApplyRefused) as ctx:
            self.build(".md", b"old\n", body)
        self.assertEqual(ctx.exception.status, "cortex_metadata_in_candidate")

    def test_hash_mismatch_is_refused(self):
        with self.assertRaises(ApplyRefused) as ctx:
            build_candidate(".txt", b"old", "new body", hash_managed_body("something else"))
        self.assertEqual(ctx.exception.status, "candidate_hash_mismatch")

    def test_empty_body_is_refused(self):
        with self.assertRaises(ApplyRefused) as ctx:
            self.build(".txt", b"old", "  \n")
        self.assertEqual(ctx.exception.status, "empty_candidate")

    def test_txt_never_preserves_frontmatter(self):
        candidate, info = self.build(".txt", b"---\na: b\n---\n\nold\n", "new\n")
        self.assertEqual(candidate, b"new\n")
        self.assertFalse(info["frontmatter_preserved"])

    def test_verification_detects_frontmatter_and_body_tampering(self):
        candidate, info = self.build(".md", b"---\na: b\n---\n\nold\n", "new\n")
        self.assertIn("preserved source frontmatter is not intact",
                      verify_written(b"new\n", candidate, info, hash_managed_body("new\n")))
        self.assertIn("source body does not hash to the approved live_vault_sha256",
                      verify_written(b"---\na: b\n---\n\nother\n", candidate, info, hash_managed_body("new\n")))


class ApplyCliTests(ApplyFixture):
    def run_cli(self, **kwargs):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = apply_proposal_cli(self.proposal_id, proposals_dir=self.proposals_dir,
                                      state_path=self.state_path, backups_dir=self.backups_dir,
                                      obsidian=self.ob, **kwargs)
        return code, buf.getvalue()

    def test_dry_run_then_apply_then_already_applied(self):
        self.seed()
        code, out = self.run_cli(dry_run=True)
        self.assertEqual(code, 0)
        self.assertIn("would_apply (dry run)", out)
        self.assertIn("+Human edited content.", out)
        self.assertEqual(self.source.read_text(), MD_SOURCE)
        code, out = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn(": applied", out)
        self.assertIn("backup:", out)
        code, out = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("already_applied", out)

    def test_refusal_exits_nonzero_and_json_output(self):
        self.seed(approve=False)
        code, out = self.run_cli(json_output=True)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["status"], "not_approved")

    def test_parser_requires_apply_proposal_for_dry_run_and_backup_dir(self):
        for flags in (["--dry-run"], ["--backup-dir", "x"]):
            proc = subprocess.run([sys.executable, str(REPO / "main.py"), *flags],
                                  capture_output=True, text=True, cwd=REPO)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("requires --apply-proposal", proc.stderr)


if __name__ == "__main__":
    unittest.main()
