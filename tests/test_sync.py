import asyncio
import json
import unittest

from ingest.sync import sync_vault_to_index
from graph.store import VectorStore
from qdrant_client import QdrantClient


class FakeOllama:
    def __init__(self, dim=6):
        self.dim = dim

    def post(self, url, json, timeout):
        texts = json["input"]
        vectors = [[float((hash(t) >> (i * 4)) % 97) for i in range(self.dim)] for t in texts]
        return type("R", (), {"status_code": 200, "raise_for_status": lambda self: None,
                               "json": lambda self: {"embeddings": vectors}})()


class FakeObsidian:
    def __init__(self, notes):
        self.notes = notes  # {path: content}
        self.written = {}

    async def iter_markdown_paths(self, root="", exclude_substrings=()):
        for path in self.notes:
            yield path

    async def read_note(self, path):
        return {"content": self.notes[path], "path": path}

    async def write_note(self, path, content):
        self.written[path] = content


def make_store():
    return VectorStore(
        ollama_client=FakeOllama(),
        qdrant_client=QdrantClient(location=":memory:"),
        collection="sync_test",
        embedding_model="fake-model",
    )


def run(coro):
    return asyncio.run(coro)


class SyncVaultToIndexTests(unittest.TestCase):
    def test_indexes_every_note_on_first_run(self):
        obsidian = FakeObsidian({"a.md": "alpha content", "b.md": "beta content"})
        store = make_store()
        report, state = run(sync_vault_to_index(obsidian, store))
        self.assertEqual(len(report["indexed"]), 2)
        self.assertEqual(report["skipped_unchanged"], [])
        self.assertEqual(len(state), 2)

    def test_second_run_with_no_changes_skips_everything(self):
        obsidian = FakeObsidian({"a.md": "alpha content"})
        store = make_store()
        _, state = run(sync_vault_to_index(obsidian, store))
        report2, _ = run(sync_vault_to_index(obsidian, store, state=state))
        self.assertEqual(report2["indexed"], [])
        self.assertEqual(report2["skipped_unchanged"], ["a.md"])

    def test_changed_note_is_reindexed_not_skipped(self):
        obsidian = FakeObsidian({"a.md": "version one"})
        store = make_store()
        _, state = run(sync_vault_to_index(obsidian, store))
        original_hash = state["a.md"]
        obsidian.notes["a.md"] = "version two, changed"
        report2, state2 = run(sync_vault_to_index(obsidian, store, state=state))
        self.assertEqual(len(report2["indexed"]), 1)
        self.assertIs(state2, state)
        self.assertNotEqual(original_hash, state2["a.md"])

    def test_read_failure_is_reported_not_fatal(self):
        obsidian = FakeObsidian({"a.md": "ok content"})

        async def broken_read(path):
            raise RuntimeError("network error")
        obsidian.read_note = broken_read
        store = make_store()
        report, _ = run(sync_vault_to_index(obsidian, store))
        self.assertEqual(len(report["errors"]), 1)
        self.assertEqual(report["errors"][0]["path"], "a.md")


if __name__ == "__main__":
    unittest.main()
