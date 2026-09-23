"""Safety audit regressions: path traversal, symlink escapes, malformed state,
tampering, authorization of restores/purges, and no silent overwrite."""
import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cortex import users
from cortex.db import CortexError
from cortex.vault_fs import FilesystemVault, VaultPathError
from tests.cortex_fixtures import CortexFixture

run = asyncio.run


class VaultPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "vault").mkdir()
        self.vault = FilesystemVault(self.root / "vault")

    def test_traversal_and_absolute_paths_refused(self):
        for bad in ("../outside.md", "a/../../x.md", "/etc/passwd", "a\\b.md", "x\x00.md", ""):
            with self.subTest(path=bad), self.assertRaises(VaultPathError):
                self.vault.resolve(bad)

    def test_symlink_escape_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "vault" / "link").symlink_to(outside)
        with self.assertRaises(VaultPathError):
            run(self.vault.write_note("link/evil.md", "x"))
        self.assertEqual(list(outside.iterdir()), [])
        self.assertNotIn("link/", run(self.vault.list_dir("")))

    def test_create_and_move_never_clobber(self):
        run(self.vault.write_note("a.md", "A"))
        run(self.vault.write_note("b.md", "B"))
        with self.assertRaises(FileExistsError):
            run(self.vault.create_note("a.md", "new"))
        with self.assertRaises(FileExistsError):
            run(self.vault.move_note("a.md", "b.md"))
        self.assertEqual(((self.root / "vault" / "a.md").read_text(), (self.root / "vault" / "b.md").read_text()),
                         ("A", "B"))


class EngineSafetyTests(CortexFixture):
    def setUp(self):
        super().setUp()
        self.seed()

    def test_symlinked_parent_directory_blocks_source_writes(self):
        real = self.root / "real_src"
        self.src_dir.rename(real)
        self.src_dir.symlink_to(real)  # the recorded source path now goes through a symlink
        self.edit_source("A\n", "A1\n")
        self.edit_note("E\n", "E1\n")
        result = self.result()
        self.assertEqual(result["status"], "unsafe_path")
        self.assertIn("E1", self.note_body())
        self.assertNotIn("E1", (real / "plan.md").read_text())

    def test_malformed_sync_state_fails_closed(self):
        self.state.write_text("{not json")
        self.edit_source("A\n", "A1\n")
        note = self.note_path.read_text()
        with self.assertRaises(CortexError) as ctx:  # the whole run refuses up front
            self.sync()
        self.assertEqual(ctx.exception.code, "malformed_state")
        self.assertEqual(self.note_path.read_text(), note)

    def test_restoring_a_deleted_source_needs_apply(self):
        users.add_user(self.db, "local", "bob", "member")
        users.set_visibility(self.db, "local", self.doc_id, "shared")
        users.grant(self.db, "local", self.doc_id, "bob", "write")
        self.source.unlink()
        tid = self.result()["tombstone_id"]
        with self.assertRaises(users.AuthorizationError):
            run(self.engine.restore_tombstone("bob", tid))
        self.assertFalse(self.source.exists())

    def test_purge_refuses_tampered_trash_paths(self):
        self.note_path.unlink()
        tid = self.result()["tombstone_id"]
        run(self.engine.approve_tombstone("local", tid))
        victim = self.root / "precious.txt"
        victim.write_text("keep me")
        with self.db.transaction():
            self.db.conn.execute("UPDATE tombstones SET trash = ? WHERE tombstone_id = ?",
                                 (json.dumps({"source": str(victim)}), tid))
        with self.assertRaises(CortexError) as ctx:
            run(self.engine.purge_tombstone("local", tid))
        self.assertEqual(ctx.exception.code, "unsafe_path")
        self.assertEqual(victim.read_text(), "keep me")

    def test_unmanaged_note_at_managed_path_is_never_overwritten(self):
        self.note_path.write_text("a human wrote this, no cortex frontmatter\n")
        self.edit_source("A\n", "A1\n")
        result = self.result()
        self.assertEqual(result["state"], "UNMANAGED")
        self.assertEqual(self.note_path.read_text(), "a human wrote this, no cortex frontmatter\n")

    def test_foreign_identity_note_is_invalid(self):
        content = self.note_path.read_text().replace(self.doc_id, "00000000-0000-0000-0000-000000000000")
        self.note_path.write_text(content)
        self.assertEqual(self.result()["state"], "INVALID")

    def test_backups_are_never_overwritten(self):
        self.edit_source("A\n", "A1\n")
        self.edit_note("E\n", "E1\n")
        backups = self.result()["backups"]
        self.assertEqual(len(backups), len(set(backups)))
        for b in backups:
            self.assertTrue(Path(b).exists())

    def test_recover_requires_admin(self):
        users.add_user(self.db, "local", "bob", "member")
        with self.assertRaises(users.AuthorizationError):
            run(self.engine.recover("bob"))

    def test_cli_reports_a_corrupt_store_instead_of_crashing(self):
        import io
        from contextlib import redirect_stdout
        import cortex_cli
        bad = self.root / "corrupt.db"
        bad.write_bytes(b"this is not sqlite" * 100)
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cortex_cli.run(["--cortex-db", str(bad), "--no-index", "--state", str(self.state), "sync-status"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(buf.getvalue())["code"], "store_error")
