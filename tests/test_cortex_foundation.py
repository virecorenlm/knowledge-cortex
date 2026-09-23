"""Phase A (identity/migration), C (versions), F (users/authorization) and
the deterministic merge algorithm."""
import json
import sqlite3
import unittest

from cortex import identity, users, versions
from cortex.db import CortexDB, CortexError
from cortex.merge import merge3
from cortex.users import AuthorizationError
from tests.cortex_fixtures import MD, Clock, CortexFixture


class MergeTests(unittest.TestCase):
    def test_non_overlapping_changes_merge(self):
        r = merge3("A\nB\nC\n", "A1\nB\nC\n", "A\nB\nC1\n")
        self.assertTrue(r.clean)
        self.assertEqual(r.text, "A1\nB\nC1\n")

    def test_overlapping_changes_conflict_with_regions(self):
        r = merge3("A\nB\nC\n", "A\nB-left\nC\n", "A\nB-right\nC\n")
        self.assertFalse(r.clean)
        self.assertIsNone(r.text)
        self.assertEqual(r.regions, [{"base_start_line": 2, "base_end_line": 2, "base": "B\n", "left": "B-left\n",
                                      "right": "B-right\n"}])

    def test_adjacent_edits_are_conservatively_a_conflict(self):
        self.assertFalse(merge3("A\nB\nC\n", "A1\nB\nC\n", "A\nB1\nC\n").clean)

    def test_identical_changes_on_both_sides_merge(self):
        self.assertEqual(merge3("A\nB\n", "A\nX\n", "A\nX\n").text, "A\nX\n")
        r = merge3("A\nB\nC\nD\n", "A\nX\nC\nD1\n", "A\nX\nC\nD\n")
        self.assertEqual(r.text, "A\nX\nC\nD1\n")

    def test_insertions_at_same_point_conflict_unless_identical(self):
        self.assertFalse(merge3("A\nB\n", "A\nL\nB\n", "A\nR\nB\n").clean)
        self.assertEqual(merge3("A\nB\n", "A\nS\nB\n", "A\nS\nB\n").text, "A\nS\nB\n")

    def test_one_side_unchanged_takes_the_other(self):
        self.assertEqual(merge3("A\n", "A\n", "Z\n").text, "Z\n")
        self.assertEqual(merge3("A\n", "Z\n", "A\n").text, "Z\n")

    def test_deterministic(self):
        args = ("h\n" * 5 + "x\n", "h\nH\n" + "h\n" * 3 + "x\n", "h\n" * 5 + "y\n")
        self.assertEqual(merge3(*args), merge3(*args))

    def test_code_fence_guard(self):
        from cortex import merge as merge_module
        self.assertTrue(merge_module._fences_balanced("a\n```\nx\n```\n"))
        self.assertFalse(merge_module._fences_balanced("a\n```\nx\n"))
        base = "intro\n\n```\ncode\n```\n\ntail\n"
        r = merge3(base, base.replace("intro", "Intro!"), base.replace("tail", "Tail!"))
        self.assertTrue(r.clean)
        self.assertEqual(r.text, "Intro!\n\n```\ncode\n```\n\nTail!\n")
        original = merge_module._fences_balanced
        calls = iter([True, True, True, False])  # inputs balanced, merged output not
        merge_module._fences_balanced = lambda text: next(calls)
        try:
            guarded = merge3(base, base.replace("intro", "I"), base.replace("tail", "T"))
        finally:
            merge_module._fences_balanced = original
        self.assertFalse(guarded.clean)
        self.assertIn("code-fence", guarded.reason)


class DbTests(unittest.TestCase):
    def test_schema_default_users_and_idempotent_open(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.db")
            db = CortexDB(path)
            self.assertEqual({u["user_id"]: u["role"] for u in users.list_users(db)},
                             {"local": "admin", "system": "system"})
            db.close()
            db = CortexDB(path)
            self.assertEqual(len(users.list_users(db)), 2)
            db.close()

    def test_newer_schema_refused(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.db")
            CortexDB(path).close()
            con = sqlite3.connect(path)
            con.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
            con.commit()
            con.close()
            with self.assertRaises(CortexError):
                CortexDB(path)


class IdentityMigrationTests(CortexFixture):
    def test_old_state_is_migrated_to_documents_with_default_owner(self):
        self.source = self.ingest()
        registered = self.engine.register("local")
        self.assertEqual(len(registered), 1)
        doc = identity.get_document(self.db, registered[0])
        self.assertEqual(doc["owner_id"], "local")
        self.assertEqual(doc["source_path"], str(self.source.resolve()))
        self.assertTrue(doc["vault_path"].startswith("Knowledge Cortex/Managed/plan-"))
        self.assertEqual(doc["reverse_writable"], 1)
        base = versions.get_revision(self.db, doc["base_revision_id"])
        self.assertEqual(base["origin"], "system")
        self.assertEqual(base["content"], "A\nB\nC\nD\nE\n")
        self.assertEqual(base["metadata"]["source_prefix"], "---\ntitle: Plan\nproject: kc\n---\n\n")
        self.assertEqual(self.engine.register("local"), [])  # idempotent

    def test_identity_is_uuid_and_stamped_into_note_on_first_sync(self):
        self.seed()
        import uuid
        uuid.UUID(self.doc_id)
        self.assertIn(f"cortex_document_id: {self.doc_id}", self.note_path.read_text())
        self.assertEqual(self.note_body(), "A\nB\nC\nD\nE\n")  # body untouched

    def test_non_admin_cannot_register(self):
        self.ingest()
        users.add_user(self.db, "local", "bob", "member")
        with self.assertRaises(AuthorizationError):
            self.engine.register("bob")

    def test_optimistic_row_versioning(self):
        self.seed()
        doc = self.doc()
        with self.db.transaction():
            identity.update_document(self.db, self.doc_id, doc["version"], project="x")
        with self.assertRaises(CortexError) as ctx:
            with self.db.transaction():
                identity.update_document(self.db, self.doc_id, doc["version"], project="y")
        self.assertEqual(ctx.exception.code, "concurrent_modification")
        self.assertEqual(self.doc()["project"], "x")


class VersionTests(CortexFixture):
    def test_first_and_next_revision_with_parents(self):
        self.seed()
        first = self.doc()["current_revision_id"]
        self.edit_source("A\n", "A1\n")
        self.sync()
        second = versions.get_revision(self.db, self.doc()["current_revision_id"])
        self.assertEqual(second["parent_ids"], [first])
        self.assertEqual(second["origin"], "source")
        self.assertEqual(second["actor_id"], "local")

    def test_merge_revision_has_two_parents(self):
        self.seed()
        self.edit_source("A\n", "A1\n")
        self.edit_note("E\n", "E1\n")
        self.sync()
        merge = versions.get_revision(self.db, self.doc()["current_revision_id"])
        self.assertEqual(merge["origin"], "merge")
        self.assertEqual(len(merge["parent_ids"]), 2)
        origins = sorted(versions.get_revision(self.db, p)["origin"] for p in merge["parent_ids"])
        self.assertEqual(origins, ["source", "vault"])

    def test_history_is_immutable(self):
        self.seed()
        rid = self.doc()["current_revision_id"]
        for sql in ("UPDATE revisions SET content = 'x'", "DELETE FROM revisions",
                    "UPDATE heads SET revision_id = NULL", "DELETE FROM events"):
            with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                self.db.conn.execute(sql)
        self.assertEqual(versions.get_revision(self.db, rid)["content"], "A\nB\nC\nD\nE\n")

    def test_revision_id_and_hash_are_deterministic(self):
        args = ("doc", ["p"], "h", "source", "local", "2026-01-01T00:00:00.000000Z", "r", {"a": 1})
        self.assertEqual(versions.compute_revision_id(*args), versions.compute_revision_id(*args))
        self.assertNotEqual(versions.compute_revision_id(*args),
                            versions.compute_revision_id(*args[:-1], {"a": 2}))
        self.assertEqual(versions.content_hash("x"), versions.content_hash("x"))

    def test_diff_and_restore_creates_new_revision(self):
        self.seed()
        first = self.doc()["current_revision_id"]
        self.edit_source("C\n", "C1\n")
        self.sync()
        second = self.doc()["current_revision_id"]
        diff = versions.diff_revisions(self.db, first, second)
        self.assertIn("-C", diff["diff"])
        self.assertIn("+C1", diff["diff"])
        import asyncio
        result = asyncio.run(self.engine.restore_revision("local", first))
        restored = versions.get_revision(self.db, result["revision_id"])
        self.assertEqual(restored["origin"], "restore")
        self.assertEqual(restored["parent_ids"], [second])
        self.assertEqual(restored["content"], "A\nB\nC\nD\nE\n")
        self.assertNotIn(restored["revision_id"], (first, second))
        self.assertIn("\nC\n", self.source.read_text())
        self.assertEqual(self.note_body(), "A\nB\nC\nD\nE\n")
        self.assertEqual(len(versions.history(self.db, self.doc_id)), 3)
        self.assertEqual(self.result()["state"], "IN_SYNC")

    def test_graph_consistency(self):
        self.seed()
        self.edit_source("A\n", "A1\n")
        self.sync()
        self.assertEqual(versions.verify_graph(self.db), [])


class UserTests(CortexFixture):
    def setUp(self):
        super().setUp()
        self.seed()
        users.add_user(self.db, "local", "alice", "member")
        users.add_user(self.db, "local", "vic", "viewer")

    def test_default_local_admin_owns_migrated_data(self):
        self.assertEqual(self.doc()["owner_id"], "local")
        self.assertTrue(users.can(self.db, "local", "write", self.doc()))

    def test_private_document_is_invisible_to_other_members(self):
        self.assertFalse(users.can(self.db, "alice", "read", self.doc()))
        users.set_visibility(self.db, "local", self.doc_id, "shared")
        self.assertTrue(users.can(self.db, "alice", "read", self.doc()))
        self.assertFalse(users.can(self.db, "alice", "write", self.doc()))

    def test_grants_and_viewer_limits(self):
        users.set_visibility(self.db, "local", self.doc_id, "shared")
        users.grant(self.db, "local", self.doc_id, "alice", "write")
        self.assertTrue(users.can(self.db, "alice", "write", self.doc()))
        self.assertFalse(users.can(self.db, "alice", "apply", self.doc()))
        users.grant(self.db, "local", self.doc_id, "vic", "write")
        self.assertFalse(users.can(self.db, "vic", "write", self.doc()))  # viewers never mutate
        self.assertTrue(users.can(self.db, "vic", "read", self.doc()))

    def test_owner_member_controls_own_document(self):
        with self.db.transaction():
            identity.update_document(self.db, self.doc_id, self.doc()["version"], visibility="private")
            self.db.conn.execute("UPDATE documents SET owner_id = 'alice' WHERE document_id = ?", (self.doc_id,))
        for cap in ("read", "write", "approve", "apply", "delete"):
            self.assertTrue(users.can(self.db, "alice", cap, self.doc()))
        self.assertFalse(users.can(self.db, "alice", "admin", self.doc()))

    def test_system_visibility_is_admin_mutable_only(self):
        users.set_visibility(self.db, "local", self.doc_id, "system")
        users.grant(self.db, "local", self.doc_id, "alice", "write")
        self.assertTrue(users.can(self.db, "alice", "read", self.doc()))
        self.assertFalse(users.can(self.db, "alice", "write", self.doc()))
        with self.assertRaises(AuthorizationError):
            users.set_visibility(self.db, "alice", self.doc_id, "system")

    def test_invalid_and_system_actors_refused(self):
        for bad in ("", "Bob", "../x", "nobody", "system", None, 5):
            with self.subTest(actor=bad), self.assertRaises(AuthorizationError):
                users.resolve_actor(self.db, bad)
        with self.assertRaises(AuthorizationError):
            users.add_user(self.db, "alice", "eve", "admin")  # members cannot add users
        with self.assertRaises(CortexError):
            users.add_user(self.db, "local", "eve", "root")

    def test_unregistered_content_is_admin_mutable_only(self):
        self.assertTrue(users.can(self.db, "alice", "read", None))
        self.assertFalse(users.can(self.db, "alice", "apply", None))
        self.assertTrue(users.can(self.db, "local", "apply", None))

    def test_unauthorized_sync_skips_document_and_writes_nothing(self):
        self.edit_source("A\n", "A1\n")
        before = self.note_path.read_text()
        report = self.sync("alice")
        self.assertEqual(report["documents"][0]["status"], "skipped_unauthorized")
        self.assertEqual(self.note_path.read_text(), before)

    def test_actor_recorded_in_revisions_and_journal(self):
        users.set_visibility(self.db, "local", self.doc_id, "shared")
        users.grant(self.db, "local", self.doc_id, "alice", "write")
        self.edit_source("A\n", "A1\n")
        self.sync("alice")
        rev = versions.get_revision(self.db, self.doc()["current_revision_id"])
        self.assertEqual(rev["actor_id"], "alice")
        events = [e for e in self.engine.journal.events(self.doc_id) if e["op_type"] == "source_to_vault"]
        self.assertEqual({e["actor_id"] for e in events}, {"alice"})
        self.assertEqual([e["phase"] for e in events], ["begin", "commit"])


if __name__ == "__main__":
    unittest.main()
