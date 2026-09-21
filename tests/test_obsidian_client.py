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
