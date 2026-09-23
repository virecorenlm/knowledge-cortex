import asyncio
import json
import unittest

from ingest.obsidian_client import ObsidianClient


class FakeVault:
    """In-memory directory tree keyed by directory path -> list of entry names
    (dirs end with '/'), mimicking vault_list's shape."""

    def __init__(self, tree):
        self.tree = tree

    async def call(self, tool_name, arguments):
        assert tool_name == "vault_list"
        path = arguments.get("path", "")
        return json.dumps({"files": self.tree.get(path, [])})


def run(coro):
    return asyncio.run(coro)


class IterMarkdownPathsTests(unittest.TestCase):
    def test_walks_nested_directories_and_yields_only_markdown(self):
        vault = FakeVault({
            "": ["notes/", "readme.md", "image.png"],
            "notes": ["a.md", "sub/"],
            "notes/sub": ["b.md"],
        })
        client = ObsidianClient(url="http://fake", authorization=None)
        client._call = vault.call
        paths = run(_collect(client))
        self.assertEqual(set(paths), {"readme.md", "notes/a.md", "notes/sub/b.md"})

    def test_excludes_paths_matching_substrings_case_insensitively(self):
        vault = FakeVault({
            "": ["Keep.md", "SKIP_ME/", "keep2.md"],
            "SKIP_ME": ["hidden.md"],
        })
        client = ObsidianClient(url="http://fake", authorization=None)
        client._call = vault.call
        paths = run(_collect(client, exclude=("skip_me",)))
        self.assertEqual(set(paths), {"Keep.md", "keep2.md"})

    def test_does_not_infinite_loop_on_repeated_directory_names(self):
        vault = FakeVault({"": ["a.md"]})
        client = ObsidianClient(url="http://fake", authorization=None)
        client._call = vault.call
        paths = run(_collect(client))
        self.assertEqual(paths, ["a.md"])


async def _collect(client, root="", exclude=()):
    return [p async for p in client.iter_markdown_paths(root=root, exclude_substrings=exclude)]


if __name__ == "__main__":
    unittest.main()


class ObsidianMoveDeleteTests(unittest.IsolatedAsyncioTestCase):
    """vault_move / vault_delete wrappers, against the verified server schemas
    (no live server: _call is recorded)."""

    def client(self, allow_moves):
        from ingest.obsidian_client import ObsidianClient
        c = ObsidianClient(url="http://x", authorization="", allow_moves=allow_moves)
        c.calls = []

        async def fake_call(tool, args):
            c.calls.append((tool, args))
            return "ok"
        c._call = fake_call
        return c

    async def test_moves_are_opt_in_and_never_overwrite(self):
        off = self.client(False)
        self.assertFalse(off.supports_moves)
        with self.assertRaises(RuntimeError):
            await off.move_note("a.md", "b.md")
        self.assertEqual(off.calls, [])
        on = self.client(True)
        await on.move_note("K/a.md", "K/sub/b.md")
        self.assertEqual(on.calls, [("vault_move", {"path": "K/a.md", "destination": "K/sub/b.md",
                                                    "allowOverwrite": False})])

    async def test_env_flag_enables_moves(self):
        import os
        from unittest.mock import patch
        from ingest.obsidian_client import ObsidianClient
        with patch.dict(os.environ, {"OBSIDIAN_ALLOW_MOVES": "1"}):
            self.assertTrue(ObsidianClient(url="http://x").supports_moves)
        with patch.dict(os.environ, {"OBSIDIAN_ALLOW_MOVES": ""}):
            self.assertFalse(ObsidianClient(url="http://x").supports_moves)

    async def test_remove_goes_to_trash_not_permanent(self):
        c = self.client(False)
        await c.remove_note("K/_cortex-trash/t/a.md")
        self.assertEqual(c.calls, [("vault_delete", {"path": "K/_cortex-trash/t/a.md", "permanent": False})])

    async def test_unsafe_paths_rejected_client_side(self):
        c = self.client(True)
        for bad in ("../x.md", "/abs.md", "a//b.md", "a/./b.md", "", "a\x00.md", "a\\b.md"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                await c.move_note(bad, "ok.md")
        self.assertEqual(c.calls, [])


class ObsidianErrorUnwrapTests(unittest.TestCase):
    def test_nested_exception_groups_unwrap_to_the_real_error(self):
        from ingest.obsidian_client import _first_leaf
        err = RuntimeError("Destination already exists")
        self.assertIs(_first_leaf(BaseExceptionGroup("a", [BaseExceptionGroup("b", [err])])), err)
