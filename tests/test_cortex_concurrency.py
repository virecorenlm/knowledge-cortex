"""Phase G: optimistic concurrency, conflict records, explicit resolution,
and AI suggestions that never auto-apply."""
import asyncio
import sqlite3

from cortex import conflicts, users, versions
from cortex.db import CortexError
from cortex.users import AuthorizationError
from tests.cortex_fixtures import CortexFixture, FakeModel

run = asyncio.run
PREFIX = "---\ntitle: Plan\nproject: kc\n---\n\n"


class ConcurrentEditTests(CortexFixture):
    def setUp(self):
        super().setUp()
        self.seed()
        self.r0 = self.doc()["current_revision_id"]

    def edit(self, body, expected, actor="local"):
        return run(conflicts.edit_document(self.engine, actor, self.doc_id, body, expected))

    def test_edit_with_current_expected_revision_applies_to_both_sides(self):
        r = self.edit("A\nB\nC-edit\nD\nE\n", self.r0)
        self.assertEqual(r["status"], "applied")
        rev = versions.get_revision(self.db, r["revision_id"])
        self.assertEqual((rev["origin"], rev["parent_ids"]), ("edit", [self.r0]))
        self.assertEqual(self.note_body(), "A\nB\nC-edit\nD\nE\n")
        self.assertEqual(self.source.read_text(), PREFIX + "A\nB\nC-edit\nD\nE\n")
        self.assertEqual(self.result()["state"], "IN_SYNC")

    def test_stale_expected_revision_with_overlap_is_a_conflict_not_an_overwrite(self):
        self.edit("A\nB\nC-first\nD\nE\n", self.r0)
        current = self.doc()["current_revision_id"]
        note, src = self.note_path.read_text(), self.source.read_text()
        r = self.edit("A\nB\nC-second\nD\nE\n", self.r0)     # based on r0, not current
        self.assertEqual(r["status"], "conflict")
        self.assertEqual((self.note_path.read_text(), self.source.read_text()), (note, src))
        self.assertEqual(self.doc()["current_revision_id"], current)
        c = conflicts.get_conflict(self.db, r["conflict_id"])
        self.assertEqual((c["kind"], c["base_revision_id"], c["left_revision_id"]), ("concurrent_edit", self.r0, current))
        self.assertEqual(c["right_revision_id"], r["edit_revision_id"])
        self.assertEqual(c["regions"][0]["left"], "C-first\n")

    def test_stale_expected_revision_without_overlap_merges(self):
        self.edit("A-first\nB\nC\nD\nE\n", self.r0)
        first = self.doc()["current_revision_id"]
        r = self.edit("A\nB\nC\nD\nE-second\n", self.r0)
        self.assertEqual(r["status"], "merged")
        merge = versions.get_revision(self.db, r["revision_id"])
        self.assertEqual(merge["parent_ids"], [first, r["edit_revision_id"]])
        self.assertEqual(self.note_body(), "A-first\nB\nC\nD\nE-second\n")
        self.assertEqual(self.source.read_text(), PREFIX + "A-first\nB\nC\nD\nE-second\n")

    def test_edit_refused_when_files_are_out_of_sync_or_revision_is_foreign(self):
        self.edit_note("E\n", "E-human\n")
        with self.assertRaises(CortexError) as ctx:
            self.edit("x\n", self.r0)
        self.assertEqual(ctx.exception.code, "not_in_sync")
        self.assertIn("E-human", self.note_body())
        self.edit_note("E-human\n", "E\n")
        with self.assertRaises(CortexError) as ctx:
            self.edit("x\n", "0" * 32)
        self.assertEqual(ctx.exception.code, "invalid_revision")

    def test_unauthorized_or_viewer_edit_refused(self):
        users.add_user(self.db, "local", "bob", "member")
        users.add_user(self.db, "local", "vic", "viewer")
        users.set_visibility(self.db, "local", self.doc_id, "shared")
        for actor in ("bob", "vic"):
            with self.subTest(actor=actor), self.assertRaises(AuthorizationError):
                self.edit("A\n", self.r0, actor=actor)
        users.grant(self.db, "local", self.doc_id, "bob", "write")
        with self.assertRaises(AuthorizationError):  # rewriting the source also needs apply
            self.edit("A-bob\nB\nC\nD\nE\n", self.r0, actor="bob")
        users.grant(self.db, "local", self.doc_id, "bob", "apply")
        self.assertEqual(self.edit("A-bob\nB\nC\nD\nE\n", self.r0, actor="bob")["status"], "applied")
        self.assertEqual(versions.get_revision(self.db, self.doc()["current_revision_id"])["actor_id"], "bob")


class ConflictResolutionTests(CortexFixture):
    def setUp(self):
        super().setUp()
        self.seed()
        self.edit_source("C\n", "C-src\n")
        self.edit_note("C\n", "C-vault\n")
        self.cid = self.result()["conflict_id"]

    def resolve(self, action, body=None, actor="local"):
        return run(conflicts.resolve_conflict(self.engine, actor, self.cid, action, body))

    def test_accept_source(self):
        r = self.resolve("accept_source")
        self.assertEqual(self.note_body(), "A\nB\nC-src\nD\nE\n")
        self.assertIn("C-src", self.source.read_text())
        rev = versions.get_revision(self.db, r["revision_id"])
        c = conflicts.get_conflict(self.db, self.cid)
        self.assertEqual(rev["parent_ids"], [c["left_revision_id"], c["right_revision_id"]])
        self.assertEqual(self.result()["state"], "IN_SYNC")

    def test_accept_vault(self):
        self.resolve("accept_vault")
        self.assertEqual(self.source.read_text(), PREFIX + "A\nB\nC-vault\nD\nE\n")
        self.assertEqual(self.result()["state"], "IN_SYNC")

    def test_manual_resolution_and_immutable_audit(self):
        r = self.resolve("manual", body="A\nB\nC-both\nD\nE\n")
        c = conflicts.get_conflict(self.db, self.cid)
        self.assertEqual((c["status"], c["resolved_by"], c["resolution_revision_id"]),
                         ("resolved", "local", r["revision_id"]))
        self.assertEqual(c["resolution"]["action"], "manual")
        with self.assertRaises(sqlite3.DatabaseError):
            self.db.conn.execute("UPDATE conflicts SET status = 'open'")
        with self.assertRaises(sqlite3.DatabaseError):
            self.db.conn.execute("DELETE FROM conflicts")
        with self.assertRaises(CortexError):
            self.resolve("accept_source")
        self.assertEqual(self.result()["state"], "IN_SYNC")

    def test_deterministic_merge_refused_while_still_overlapping(self):
        with self.assertRaises(CortexError) as ctx:
            self.resolve("deterministic_merge")
        self.assertEqual(ctx.exception.code, "merge_conflict")
        self.assertEqual(conflicts.get_conflict(self.db, self.cid)["status"], "open")

    def test_stale_conflict_resolution_refused(self):
        self.edit_note("A\n", "A-later\n")
        with self.assertRaises(CortexError) as ctx:
            self.resolve("accept_source")
        self.assertEqual(ctx.exception.code, "stale_precondition")
        self.assertIn("A-later", self.note_body())

    def test_ai_suggestion_is_stored_but_never_applied(self):
        model = FakeModel("A\nB\nC-src and C-vault\nD\nE\n")
        note, src = self.note_path.read_text(), self.source.read_text()
        suggestion = conflicts.suggest_resolution(self.db, "local", self.cid, model)
        self.assertEqual((suggestion["status"], suggestion["applied"], suggestion["model"]),
                         ("suggestion", False, "fake-model"))
        self.assertEqual((self.note_path.read_text(), self.source.read_text()), (note, src))
        self.assertEqual(conflicts.get_conflict(self.db, self.cid)["status"], "open")
        self.assertEqual(self.result()["status"], "conflict_open")  # sync does not use it either
        r = self.resolve("accept_suggestion")                       # only an explicit action applies it
        self.assertEqual(self.note_body(), "A\nB\nC-src and C-vault\nD\nE\n")
        self.assertEqual(versions.get_revision(self.db, r["revision_id"])["metadata"]["suggestion_model"],
                         "fake-model")

    def test_unauthorized_resolution_refused(self):
        users.add_user(self.db, "local", "bob", "member")
        with self.assertRaises(AuthorizationError):
            self.resolve("accept_source", actor="bob")
        self.assertEqual(conflicts.get_conflict(self.db, self.cid)["status"], "open")
        self.assertEqual(conflicts.list_conflicts(self.db, "bob"), [])

    def test_invalid_actions_refused(self):
        for action, body in (("accept_left_maybe", None), ("manual", ""), ("accept_suggestion", None)):
            with self.subTest(action=action), self.assertRaises(CortexError):
                self.resolve(action, body)
