import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sync_cli import load_state, save_state


class NamespacedStateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "sync_state.json"

    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(load_state(self.path), {})
        self.assertEqual(load_state(self.path, namespace="local"), {})

    def test_vault_and_local_namespaces_do_not_collide(self):
        save_state(self.path, {"note.md": "hash1"}, namespace="vault")
        save_state(self.path, {"/abs/file.pdf": "hash2"}, namespace="local")
        self.assertEqual(load_state(self.path, namespace="vault"), {"note.md": "hash1"})
        self.assertEqual(load_state(self.path, namespace="local"), {"/abs/file.pdf": "hash2"})

    def test_saving_one_namespace_preserves_the_other(self):
        save_state(self.path, {"a.md": "h1"}, namespace="vault")
        save_state(self.path, {"a.md": "h1", "b.md": "h2"}, namespace="vault")
        save_state(self.path, {"/x.pdf": "h3"}, namespace="local")
        self.assertEqual(load_state(self.path, namespace="vault"), {"a.md": "h1", "b.md": "h2"})
        self.assertEqual(load_state(self.path, namespace="local"), {"/x.pdf": "h3"})

    def test_legacy_flat_state_file_is_read_as_vault_namespace(self):
        self.path.write_text(json.dumps({"note.md": "legacyhash"}))
        self.assertEqual(load_state(self.path), {"note.md": "legacyhash"})
        self.assertEqual(load_state(self.path, namespace="local"), {})

    def test_writing_after_legacy_format_migrates_without_losing_data(self):
        self.path.write_text(json.dumps({"note.md": "legacyhash"}))
        save_state(self.path, {"/x.pdf": "h1"}, namespace="local")
        self.assertEqual(load_state(self.path, namespace="vault"), {"note.md": "legacyhash"})
        self.assertEqual(load_state(self.path, namespace="local"), {"/x.pdf": "h1"})

    def test_default_namespace_is_vault(self):
        save_state(self.path, {"a.md": "h1"})
        self.assertEqual(load_state(self.path), {"a.md": "h1"})


if __name__ == "__main__":
    unittest.main()
