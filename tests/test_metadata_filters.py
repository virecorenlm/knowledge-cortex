import asyncio
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient

from graph.store import VectorStore, build_search_filter
from ingest.metadata import extract_filter_metadata, extract_tags, parse_frontmatter, to_rfc3339
from ingest.sync import index_local_path, sync_vault_to_index


class FakeOllama:
    def __init__(self, dim=6):
        self.dim = dim
        self.calls = 0

    def post(self, url, json, timeout):
        self.calls += 1
        texts = json["input"]
        vectors = [[float((hash(t) >> (i * 4)) % 97) + 1.0 for i in range(self.dim)] for t in texts]
        return type("R", (), {"status_code": 200, "raise_for_status": lambda self: None,
                               "json": lambda self: {"embeddings": vectors}})()


class FakeObsidian:
    def __init__(self, notes):
        self.notes = notes

    async def iter_markdown_paths(self, root="", exclude_substrings=()):
        for path in self.notes:
            yield path

    async def read_note(self, path):
        return {"content": self.notes[path], "path": path}


def make_store():
    return VectorStore(
        ollama_client=FakeOllama(),
        qdrant_client=QdrantClient(location=":memory:"),
        collection="filter_test",
        embedding_model="fake-model",
    )


def paths(results):
    return sorted({r["path"] for r in results})


class FrontmatterAndTagTests(unittest.TestCase):
    def test_parses_scalars_inline_and_block_lists(self):
        fm = parse_frontmatter("---\ntitle: \"Hi\"\ntags: [a, 'b']\naliases:\n  - x\n  - y\n---\nbody")
        self.assertEqual(fm, {"title": "Hi", "tags": ["a", "b"], "aliases": ["x", "y"]})

    def test_no_frontmatter_is_empty(self):
        self.assertEqual(parse_frontmatter("just text\n---\nnot: fm\n---"), {})

    def test_tags_from_frontmatter_and_inline_are_normalized_and_expanded(self):
        text = "---\ntags: Project/Alpha, #Idea\n---\nSee #todo and #Area/Health/Sleep.\n"
        self.assertEqual(extract_tags(text), sorted(
            ["project", "project/alpha", "idea", "todo", "area", "area/health", "area/health/sleep"]))

    def test_non_tags_are_ignored(self):
        text = ("# Heading\nissue #123 and url http://x.com/#anchor and a#b\n"
                "`#inline` code\n```\n#fenced\n```\n#real\n")
        self.assertEqual(extract_tags(text), ["real"])


class FilterMetadataTests(unittest.TestCase):
    def test_vault_project_defaults_to_top_level_folder(self):
        meta = extract_filter_metadata("x", path="Vire_Realm/notes/a.md", folder_as_project=True)
        self.assertEqual(meta["project"], "Vire_Realm")

    def test_root_level_note_has_no_project(self):
        self.assertIsNone(extract_filter_metadata("x", path="a.md", folder_as_project=True)["project"])

    def test_frontmatter_project_beats_folder_and_explicit_beats_frontmatter(self):
        text = "---\nproject: Beta\n---\nx"
        self.assertEqual(extract_filter_metadata(text, path="Alpha/a.md", folder_as_project=True)["project"], "Beta")
        self.assertEqual(extract_filter_metadata(text, project="Gamma")["project"], "Gamma")

    def test_date_precedence_frontmatter_then_filename_then_fallback(self):
        fm = "---\ncreated: 2023-02-03T10:00:00+02:00\n---\nx"
        self.assertEqual(extract_filter_metadata(fm, path="2024-01-01.md")["doc_date"], "2023-02-03T08:00:00Z")
        self.assertEqual(extract_filter_metadata("x", path="daily/2024-01-01 Mon.md")["doc_date"],
                         "2024-01-01T00:00:00Z")
        self.assertEqual(extract_filter_metadata("x", path="a.md", fallback_date="2022-06-01")["doc_date"],
                         "2022-06-01T00:00:00Z")
        self.assertIsNone(extract_filter_metadata("x", path="a.md")["doc_date"])

    def test_unparseable_date_is_ignored(self):
        self.assertIsNone(extract_filter_metadata("---\ndate: someday\n---\nx")["doc_date"])
        self.assertIsNone(to_rfc3339("2024-13-45"))


class BuildSearchFilterTests(unittest.TestCase):
    def test_empty_filters_mean_no_filter(self):
        self.assertIsNone(build_search_filter(None))
        self.assertIsNone(build_search_filter({"source": None, "tags": [], "tag_mode": "all"}))

    def test_invalid_filters_raise(self):
        for bad in ({"colour": "red"}, {"tag_mode": "some"}, {"date_from": "nope"},
                    {"date_to": "2024-99-01T00:00"}, {"tags": ["bad tag"]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                build_search_filter(bad)


class FilteredSearchTests(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        notes = {
            "Alpha/2024-05-01.md": "---\ntags: [health]\n---\nsleep notes #sleep/deep",
            "Alpha/plan.md": "---\ndate: 2024-06-15\n---\nplanning #work",
            "Beta/idea.md": "---\ndate: 2025-01-10\ntags: work, health\n---\nidea",
            "loose.md": "no metadata at all",
        }
        asyncio.run(sync_vault_to_index(FakeObsidian(notes), self.store))

    def search(self, **filters):
        return paths(self.store.search("anything", limit=50, filters=filters))

    def test_unfiltered_returns_everything(self):
        self.assertEqual(len(self.search()), 4)

    def test_payload_carries_metadata(self):
        hit = self.store.search("x", limit=50, filters={"project": "Beta"})[0]
        self.assertEqual((hit["project"], hit["tags"], hit["doc_date"]),
                         ("Beta", ["health", "work"], "2025-01-10T00:00:00Z"))

    def test_source_filter(self):
        self.assertEqual(len(self.search(source="obsidian")), 4)
        self.assertEqual(self.search(source="local_ingest"), [])

    def test_project_filter_single_and_any_of(self):
        self.assertEqual(self.search(project="Alpha"), ["Alpha/2024-05-01.md", "Alpha/plan.md"])
        self.assertEqual(len(self.search(project=["Alpha", "Beta"])), 3)

    def test_tags_all_vs_any_and_nested_parent_match(self):
        self.assertEqual(self.search(tags=["health", "work"]), ["Beta/idea.md"])
        self.assertEqual(len(self.search(tags=["health", "work"], tag_mode="any")), 3)
        self.assertEqual(self.search(tags="sleep"), ["Alpha/2024-05-01.md"])
        self.assertEqual(self.search(tags="#Sleep/Deep"), ["Alpha/2024-05-01.md"])

    def test_date_range_is_inclusive_and_excludes_undated(self):
        self.assertEqual(self.search(date_from="2024-05-01", date_to="2024-06-15"),
                         ["Alpha/2024-05-01.md", "Alpha/plan.md"])
        self.assertEqual(self.search(date_from="2024-12-31"), ["Beta/idea.md"])
        self.assertEqual(self.search(date_to="2024-05-01T00:00:00Z"), ["Alpha/2024-05-01.md"])

    def test_filters_combine_with_and(self):
        self.assertEqual(self.search(project="Alpha", tags="work", date_from="2024-06-01"), ["Alpha/plan.md"])
        self.assertEqual(self.search(project="Beta", date_to="2024-12-31"), [])


class LocalIngestMetadataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = make_store()

    def write(self, name, content, mtime=None):
        p = self.root / name
        p.write_text(content, encoding="utf-8")
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    def test_local_chunks_get_project_tags_and_mtime_date(self):
        f = self.write("note.txt", "remember #groceries", mtime=1700000000)  # 2023-11-14T22:13:20Z
        index_local_path(self.store, str(f), state={}, project="Home")
        hits = self.store.search("x", filters={"source": "local_ingest", "project": "Home", "tags": "groceries",
                                               "date_from": "2023-11-14", "date_to": "2023-11-14"})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["doc_date"], "2023-11-14T22:13:20Z")

    def test_ingestion_timestamp_is_not_the_document_date(self):
        f = self.write("note.txt", "text", mtime=946684800)  # 2000-01-01
        index_local_path(self.store, str(f), state={})
        self.assertEqual(self.store.search("x")[0]["doc_date"], "2000-01-01T00:00:00Z")

    def test_unchanged_file_backfills_metadata_without_reembedding(self):
        f = self.write("note.txt", "body #alpha")
        _, state = index_local_path(self.store, str(f), state={})
        # Simulate state written before filter metadata existed.
        del state[str(f.resolve())]["filter_metadata"]
        self.store.set_metadata("local_ingest/note.txt", {"project": None, "tags": [], "doc_date": None})
        calls_before = self.store._http.calls

        report, state = index_local_path(self.store, str(f), state=state, project="P")

        self.assertEqual(report["skipped_unchanged"], [str(f.resolve())])
        self.assertEqual(report["metadata_updated"], [str(f.resolve())])
        self.assertEqual(self.store._http.calls, calls_before)  # no embedding call
        self.assertEqual(paths(self.store.search("x", filters={"project": "P", "tags": "alpha"})),
                         ["local_ingest/note.txt"])

        report, _ = index_local_path(self.store, str(f), state=state, project="P")
        self.assertEqual(report["metadata_updated"], [])  # nothing changed -> no Qdrant write

    def test_backfill_failure_is_reported_and_retried(self):
        f = self.write("note.txt", "body")
        _, state = index_local_path(self.store, str(f), state={})
        original = self.store.set_metadata

        def broken(path, fields):
            raise RuntimeError("qdrant down")
        self.store.set_metadata = broken
        report, state = index_local_path(self.store, str(f), state=state, project="P")
        self.assertEqual(len(report["errors"]), 1)
        self.store.set_metadata = original
        report, _ = index_local_path(self.store, str(f), state=state, project="P")
        self.assertEqual(report["metadata_updated"], [str(f.resolve())])


class McpSemanticSearchTests(unittest.TestCase):
    def test_filters_pass_through_and_invalid_filters_return_error_json(self):
        import mcp_server

        store = make_store()
        asyncio.run(sync_vault_to_index(FakeObsidian({"A/x.md": "#t one", "B/y.md": "two"}), store))
        original = mcp_server._store
        mcp_server._store = lambda: store
        self.addCleanup(setattr, mcp_server, "_store", original)

        results = json.loads(mcp_server.semantic_search("q", project=["A"], tags=["t"]))
        self.assertEqual([r["path"] for r in results], ["A/x.md"])
        error = json.loads(mcp_server.semantic_search("q", tag_mode="sometimes"))
        self.assertIn("tag_mode", error["error"])


if __name__ == "__main__":
    unittest.main()
