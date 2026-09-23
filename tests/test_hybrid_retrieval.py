"""Hybrid retrieval tests. In-memory Qdrant plus a fake embedder that maps
query text to fixed vectors, so every cosine score is known exactly."""
import asyncio
import importlib
import io
import json
import math
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from graph import retrieval
from graph.retrieval import DEFAULT_RANKING, RankingConfig, hybrid_search, to_json
from graph.store import VectorStore

REPO = Path(__file__).resolve().parent.parent


def unit(cos):
    """3-d unit vector whose cosine with (1, 0, 0) is exactly `cos`."""
    return [cos, math.sqrt(max(0.0, 1 - cos * cos)), 0.0]


class FixedEmbedder:
    def __init__(self):
        self.calls = 0
        self.inputs = []

    def post(self, url, json, timeout):
        self.calls += 1
        self.inputs.extend(json["input"])
        vectors = [[1.0, 0.0, 0.0] for _ in json["input"]]
        return type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: {"embeddings": vectors}})()


def make_store(distance=models.Distance.COSINE, create=True):
    store = VectorStore(ollama_client=FixedEmbedder(), qdrant_client=QdrantClient(location=":memory:"),
                        collection="hybrid_test", embedding_model="fake")
    if create:
        store.client.create_collection("hybrid_test",
                                       vectors_config=models.VectorParams(size=3, distance=distance))
    return store


def add(store, pid, cos, **payload):
    payload.setdefault("text", f"text of point {pid}")
    store.client.upsert("hybrid_test", points=[models.PointStruct(id=pid, vector=unit(cos), payload=payload)],
                        wait=True)


def paths(response):
    return [r["path"] for r in response["results"]]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        add(self.store, 1, 0.90, path="Net/a.md", source="obsidian", project="other", tags=["networking"],
            chunk_index=0, doc_date="2023-01-01T00:00:00Z")
        add(self.store, 2, 0.87, path="KC/b.md", source="obsidian", project="knowledge-cortex",
            tags=["architecture"], chunk_index=0, doc_date="2024-06-01T00:00:00Z")
        add(self.store, 3, 0.40, path="KC/c.md", source="obsidian", project="knowledge-cortex",
            tags=["architecture", "qdrant"], chunk_index=0, doc_date="2024-07-01T00:00:00Z")
        add(self.store, 4, 0.70, path="local_ingest/d.txt", source="local_ingest", project="knowledge-cortex",
            tags=["qdrant"], chunk_index=0, source_file="/src/d.txt", markdown_path="local_ingest/d.txt",
            ai_structured=True, source_sha256="s" * 64, doc_date=None)
        add(self.store, 5, 0.60, path="local_ingest/e.md", source="local_ingest", project=None, tags=[],
            chunk_index=0, source_file="/src/e.md", markdown_path="local_ingest/e.md", ai_structured=False,
            doc_date="2024-01-15T00:00:00Z")

    def search(self, query="q", **kwargs):
        kwargs.setdefault("limit", 10)
        return hybrid_search(self.store, query, **kwargs)


class SemanticOnlyTests(Fixture):
    def test_semantic_only_equals_existing_search(self):
        legacy = self.store.search("q", limit=3, instruct="task")
        hybrid = self.search(limit=3, instruct="task")
        self.assertEqual(hybrid["mode"], "semantic")
        self.assertEqual(hybrid["candidate_limit"], 3)
        self.assertEqual([r["path"] for r in legacy], paths(hybrid))
        self.assertEqual([r["score"] for r in legacy], [r["scores"]["semantic"] for r in hybrid["results"]])
        self.assertTrue(all(r["scores"]["metadata_boost"] == 0 for r in hybrid["results"]))
        self.assertEqual(self.store._http.inputs[-1], self.store._http.inputs[-2])  # same instruct text

    def test_legacy_store_search_is_unchanged(self):
        results = self.store.search("q", limit=2, filters={"project": "knowledge-cortex"})
        self.assertEqual([r["path"] for r in results], ["KC/b.md", "local_ingest/d.txt"])
        self.assertIn("score", results[0])
        self.assertEqual(results[0]["tags"], ["architecture"])


class HardFilterTests(Fixture):
    def test_equality(self):
        self.assertEqual(paths(self.search(filters={"project": "other"})), ["Net/a.md"])
        self.assertEqual(paths(self.search(filters={"project": {"eq": "other"}})), ["Net/a.md"])
        self.assertEqual(paths(self.search(filters={"source_file": "/src/e.md"})), ["local_ingest/e.md"])
        self.assertEqual(paths(self.search(filters={"path": "KC/c.md"})), ["KC/c.md"])

    def test_one_of(self):
        self.assertEqual(paths(self.search(filters={"project": ["other", "knowledge-cortex"]})),
                         ["Net/a.md", "KC/b.md", "local_ingest/d.txt", "KC/c.md"])
        self.assertEqual(paths(self.search(filters={"source": {"in": ["local_ingest"]}})),
                         ["local_ingest/d.txt", "local_ingest/e.md"])

    def test_tags_contains_and_contains_any(self):
        self.assertEqual(paths(self.search(filters={"tags": "qdrant"})), ["local_ingest/d.txt", "KC/c.md"])
        self.assertEqual(paths(self.search(filters={"tags": ["networking", "qdrant"]})),
                         ["Net/a.md", "local_ingest/d.txt", "KC/c.md"])
        self.assertEqual(paths(self.search(filters={"tags": {"contains_any": ["#Networking"]}})), ["Net/a.md"])

    def test_tags_contains_all(self):
        self.assertEqual(paths(self.search(filters={"tags": {"contains_all": ["architecture", "qdrant"]}})),
                         ["KC/c.md"])

    def test_boolean_with_missing_counting_as_false(self):
        self.assertEqual(paths(self.search(filters={"ai_structured": True})), ["local_ingest/d.txt"])
        self.assertEqual(paths(self.search(filters={"ai_structured": False})),
                         ["Net/a.md", "KC/b.md", "local_ingest/e.md", "KC/c.md"])

    def test_date_range_with_inclusive_days(self):
        self.assertEqual(paths(self.search(filters={"doc_date": {"gte": "2024-06-01", "lte": "2024-07-01"}})),
                         ["KC/b.md", "KC/c.md"])
        self.assertEqual(paths(self.search(filters={"doc_date": {"gt": "2024-06-01"}})), ["KC/c.md"])

    def test_hard_filter_removes_semantically_strongest(self):
        response = self.search(filters={"project": "knowledge-cortex"})
        self.assertNotIn("Net/a.md", paths(response))
        self.assertEqual(response["results"][0]["matched_filters"],
                         [{"field": "project", "op": "eq", "value": "knowledge-cortex"}])

    def test_combined_filters_are_anded(self):
        self.assertEqual(paths(self.search(filters={"project": "knowledge-cortex", "tags": "qdrant",
                                                    "source": "obsidian"})), ["KC/c.md"])

    def test_invalid_fields_operators_and_values_are_rejected_before_embedding(self):
        bad = [
            {"source_type": "markdown"}, {"text": "x"}, {"project": {"startswith": "k"}},
            {"project": {"eq": "a", "in": ["b"]}}, {"tags": {"eq": "a"}}, {"ai_structured": "false"},
            {"project": ""}, {"project": []}, {"tags": "not a tag!"}, {"doc_date": "2024-01-01"},
            {"doc_date": {"after": "2024"}}, {"doc_date": {"gte": "yesterday"}}, {"project": 5},
        ]
        for filters in bad:
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                self.search(filters=filters)
        for prefer in ({"source_type": "md"}, {"tags": {"value": "a", "weight": "high"}},
                       {"tags": {"value": "a", "boost": 1}}):
            with self.subTest(prefer=prefer), self.assertRaises(ValueError):
                self.search(prefer=prefer)
        with self.assertRaises(ValueError):
            self.search(filters="project=x")
        self.assertEqual(self.store._http.calls, 0)


class SoftPreferenceTests(Fixture):
    def test_preference_promotes_a_near_tie(self):
        self.assertEqual(paths(self.search(limit=2)), ["Net/a.md", "KC/b.md"])
        response = self.search(limit=2, prefer={"project": "knowledge-cortex"})
        self.assertEqual(paths(response), ["KC/b.md", "Net/a.md"])  # 0.87 + 0.05 > 0.90
        top = response["results"][0]
        self.assertEqual(top["matched_preferences"], [{"field": "project", "op": "eq", "value": "knowledge-cortex",
                                                       "weight": 0.05, "match": 1.0, "contribution": 0.05}])

    def test_boost_is_bounded_so_a_large_semantic_gap_wins(self):
        prefer = {"project": {"value": "knowledge-cortex", "weight": 1.0},
                  "tags": {"value": ["architecture", "qdrant"], "weight": 1.0},
                  "source": {"value": "obsidian", "weight": 1.0},
                  "doc_date": {"value": {"gte": "2024-07-01"}, "weight": 1.0}}
        response = self.search(prefer=prefer)
        c = next(r for r in response["results"] if r["path"] == "KC/c.md")
        self.assertAlmostEqual(c["scores"]["metadata_boost"], DEFAULT_RANKING.max_total_boost)
        self.assertLess(paths(response).index("Net/a.md"), paths(response).index("KC/c.md"))
        for p in c["matched_preferences"]:
            self.assertLessEqual(p["weight"], DEFAULT_RANKING.max_preference_weight)

    def test_score_components_compose_by_the_documented_rule(self):
        response = self.search(prefer={"tags": ["architecture", "qdrant"], "project": "knowledge-cortex"})
        for r in response["results"]:
            s = r["scores"]
            raw = sum(p["contribution"] for p in r["matched_preferences"])
            self.assertAlmostEqual(s["metadata_boost"], min(raw, DEFAULT_RANKING.max_total_boost))
            self.assertAlmostEqual(s["final"], s["semantic"] + s["metadata_boost"])
            self.assertEqual(r["score"], s["final"])
        b = next(r for r in response["results"] if r["path"] == "KC/b.md")
        tag_pref = next(p for p in b["matched_preferences"] if p["field"] == "tags")
        self.assertEqual(tag_pref["match"], 0.5)  # 1 of 2 preferred tags
        self.assertAlmostEqual(tag_pref["contribution"], 0.025)
        finals = [r["scores"]["final"] for r in response["results"]]
        self.assertEqual(finals, sorted(finals, reverse=True))

    def test_bool_and_date_preferences(self):
        response = self.search(prefer={"ai_structured": False, "doc_date": {"gte": "2024-01-01"}})
        e = next(r for r in response["results"] if r["path"] == "local_ingest/e.md")
        self.assertAlmostEqual(e["scores"]["metadata_boost"], 0.10)
        d = next(r for r in response["results"] if r["path"] == "local_ingest/d.txt")
        self.assertEqual(d["scores"]["metadata_boost"], 0.0)  # ai_structured true, no date

    def test_candidate_pool_lets_a_preferred_chunk_outside_top_n_rise(self):
        store = make_store()
        for i in range(1, 6):
            add(store, i, 0.90 - i * 0.001, path=f"other/{i}.md", project="other", chunk_index=0)
        add(store, 99, 0.88, path="kc/target.md", project="knowledge-cortex", chunk_index=0)
        plain = hybrid_search(store, "q", limit=3)
        self.assertNotIn("kc/target.md", paths(plain))
        boosted = hybrid_search(store, "q", limit=3, prefer={"project": "knowledge-cortex"})
        self.assertEqual(boosted["candidate_limit"], 20)
        self.assertEqual(paths(boosted)[0], "kc/target.md")

    def test_candidate_limit_rule(self):
        cfg = DEFAULT_RANKING
        self.assertEqual(retrieval.candidate_limit(3, {}, None, cfg), 3)
        self.assertEqual(retrieval.candidate_limit(3, {"x": 1}, None, cfg), 20)
        self.assertEqual(retrieval.candidate_limit(10, {"x": 1}, None, cfg), 40)
        self.assertEqual(retrieval.candidate_limit(100, {}, 2, cfg), 200)
        self.assertEqual(retrieval.candidate_limit(200, {"x": 1}, None, cfg), 200)

    def test_preferences_require_cosine(self):
        store = make_store(distance=models.Distance.DOT)
        add(store, 1, 0.5, path="a.md")
        self.assertEqual(paths(hybrid_search(store, "q")), ["a.md"])
        with self.assertRaises(ValueError):
            hybrid_search(store, "q", prefer={"project": "x"})

    def test_custom_ranking_config_is_the_only_tuning_surface(self):
        cfg = RankingConfig(default_preference_weight=0.2, max_preference_weight=0.5, max_total_boost=0.5)
        response = self.search(prefer={"project": "knowledge-cortex"}, config=cfg)
        self.assertEqual(paths(response)[0], "KC/b.md")
        self.assertAlmostEqual(response["results"][0]["scores"]["metadata_boost"], 0.2)
        self.assertEqual(response["ranking"]["max_total_boost"], 0.5)


class DeterminismTests(Fixture):
    def test_identical_inputs_give_identical_output(self):
        kwargs = dict(filters={"project": "knowledge-cortex"}, prefer={"tags": ["qdrant"]}, max_per_source=1)
        first = to_json(self.search(**kwargs))
        for _ in range(5):
            self.assertEqual(to_json(self.search(**kwargs)), first)

    def test_ties_break_by_identity_then_chunk_index_then_id(self):
        store = make_store()
        add(store, 30, 0.5, path="b.md", chunk_index=1)
        add(store, 10, 0.5, path="b.md", chunk_index=0)
        add(store, 20, 0.5, path="a.md", chunk_index=0)
        add(store, 40, 0.5, path="a.md", chunk_index=0)
        response = hybrid_search(store, "q", limit=10, prefer={"project": "none"})
        self.assertEqual([r["id"] for r in response["results"]], ["20", "40", "10", "30"])


class ResultSchemaTests(Fixture):
    def test_payload_metadata_and_provenance(self):
        [d] = self.search(filters={"source_file": "/src/d.txt"})["results"]
        self.assertEqual(d["id"], "4")
        self.assertEqual(d["source"], "local_ingest")
        self.assertEqual(d["path"], "local_ingest/d.txt")
        self.assertEqual(d["source_file"], "/src/d.txt")
        self.assertEqual(d["source_identity"], "/src/d.txt")
        self.assertEqual(d["chunk_index"], 0)
        self.assertEqual(d["text"], "text of point 4")
        self.assertEqual(d["metadata"], {"path": "local_ingest/d.txt", "source": "local_ingest",
                                         "project": "knowledge-cortex", "tags": ["qdrant"], "chunk_index": 0,
                                         "source_file": "/src/d.txt", "markdown_path": "local_ingest/d.txt",
                                         "ai_structured": True, "source_sha256": "s" * 64, "doc_date": None})
        self.assertEqual(d["rank"], 1)

    def test_json_is_stable_and_has_no_timings_by_default(self):
        text = to_json(self.search(prefer={"project": "knowledge-cortex"}))
        data = json.loads(text)
        self.assertNotIn("timings_ms", data)
        self.assertEqual(text, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
        timed = self.search(with_timings=True)
        self.assertEqual(set(timed["timings_ms"]), {"embed_ms", "qdrant_ms", "rerank_ms", "total_ms"})


class DiversityTests(Fixture):
    def setUp(self):
        super().setUp()
        for i, cos in enumerate((0.95, 0.94, 0.93)):
            add(self.store, 100 + i, cos, path="Deep/doc.md", project="deep", chunk_index=i)

    def test_cap_per_source(self):
        response = self.search(limit=4, max_per_source=1)
        self.assertEqual(paths(response), ["Deep/doc.md", "Net/a.md", "KC/b.md", "local_ingest/d.txt"])
        self.assertEqual(response["candidate_limit"], 20)

    def test_off_by_default_keeps_normal_ranking(self):
        response = self.search(limit=4)
        self.assertEqual(paths(response), ["Deep/doc.md", "Deep/doc.md", "Deep/doc.md", "Net/a.md"])
        self.assertEqual([r["chunk_index"] for r in response["results"][:3]], [0, 1, 2])

    def test_invalid_cap_rejected(self):
        for cap in (0, -1, True, "2"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.search(max_per_source=cap)


class RobustnessTests(unittest.TestCase):
    def test_missing_and_legacy_metadata_do_not_crash(self):
        store = make_store()
        add(store, 1, 0.9, source_path="Vire_Realm/mongodb.md", model="old", chunk_index=0, sha256="x")
        add(store, 2, 0.8)
        response = hybrid_search(store, "q", prefer={"project": "x", "tags": ["a"], "ai_structured": False,
                                                     "doc_date": {"gte": "2020-01-01"}}, max_per_source=1)
        first, second = response["results"]
        self.assertEqual(first["path"], "Vire_Realm/mongodb.md")
        self.assertEqual(first["source_identity"], "Vire_Realm/mongodb.md")
        self.assertEqual(second["source_identity"], "2")
        self.assertEqual(first["scores"]["metadata_boost"], 0.05)  # only ai_structured (missing = false)

    def test_malformed_payload_values_are_ignored_safely(self):
        store = make_store()
        add(store, 1, 0.9, path=7, project=["not", "a", "string"], tags="architecture", doc_date="garbage",
            chunk_index="zero", ai_structured="yes", text=None)
        add(store, 2, 0.9, path="ok.md", tags=[1, None, "architecture"], doc_date=12345, chunk_index=True)
        response = hybrid_search(store, "q", prefer={"project": "not", "tags": "architecture",
                                                     "doc_date": {"gte": "2000-01-01"}, "ai_structured": True})
        by_id = {r["id"]: r for r in response["results"]}
        self.assertEqual(by_id["1"]["scores"]["metadata_boost"], 0.0)
        self.assertEqual(by_id["1"]["text"], "")
        self.assertEqual(by_id["2"]["scores"]["metadata_boost"], 0.05)  # only the valid tag

    def test_missing_collection_returns_cleanly_without_embedding(self):
        store = make_store(create=False)
        response = hybrid_search(store, "q", prefer={"project": "x"})
        self.assertEqual(response["results"], [])
        self.assertEqual(store._http.calls, 0)
        self.assertFalse(store.client.collection_exists("hybrid_test"))

    def test_empty_collection_returns_cleanly(self):
        self.assertEqual(hybrid_search(make_store(), "q", prefer={"project": "x"})["results"], [])

    def test_limit_handling(self):
        store = make_store()
        for i in range(1, 8):
            add(store, i, 0.5 + i / 100, path=f"{i}.md")
        self.assertEqual(len(hybrid_search(store, "q", limit=3)["results"]), 3)
        self.assertEqual(len(hybrid_search(store, "q", limit=50)["results"]), 7)
        for bad in (0, -1, 201, 2.5, True, "10"):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                hybrid_search(store, "q", limit=bad)


class MetadataOnlyTests(Fixture):
    def test_requires_a_hard_filter(self):
        for query in ("", "   ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.search(query)

    def test_lists_matching_chunks_without_embedding(self):
        response = self.search("", filters={"project": "knowledge-cortex"}, prefer={"tags": "qdrant"})
        self.assertEqual(response["mode"], "metadata_only")
        self.assertEqual(self.store._http.calls, 0)
        # c and d tie on boost; identity breaks it ("/src/d.txt" < "KC/c.md").
        self.assertEqual(paths(response), ["local_ingest/d.txt", "KC/c.md", "KC/b.md"])
        self.assertIsNone(response["results"][0]["scores"]["semantic"])
        self.assertEqual(response["results"][0]["scores"]["final"], 0.05)
        self.assertFalse(response["truncated"])

    def test_truncation_is_reported(self):
        cfg = RankingConfig(metadata_only_max=2)
        response = self.search("", filters={"source": "obsidian"}, config=cfg, limit=10)
        self.assertTrue(response["truncated"])
        self.assertEqual(len(response["results"]), 2)


class IndexAndMutationSafetyTests(unittest.TestCase):
    def test_retrieval_index_creation_is_explicit_and_idempotent(self):
        class FakeClient:
            def __init__(self):
                self.schema, self.created = {"path": object()}, []

            def collection_exists(self, name):
                return True

            def get_collection(self, name):
                return type("C", (), {"payload_schema": dict(self.schema)})()

            def create_payload_index(self, collection_name, field_name, field_schema, wait):
                self.created.append(field_name)
                self.schema[field_name] = field_schema

        client = FakeClient()
        store = VectorStore(ollama_client=FixedEmbedder(), qdrant_client=client, collection="c")
        first = store.ensure_retrieval_indexes()
        self.assertEqual(first["existing"], ["path"])
        self.assertEqual(set(first["created"]),
                         {"source", "project", "tags", "doc_date", "source_file", "ai_structured"})
        second = store.ensure_retrieval_indexes()
        self.assertEqual(second["created"], [])
        self.assertEqual(len(client.created), 6)

    def test_index_helper_never_creates_a_collection(self):
        store = make_store(create=False)
        with self.assertRaises(RuntimeError):
            store.ensure_retrieval_indexes()
        self.assertFalse(store.client.collection_exists("hybrid_test"))

    def test_importing_modules_creates_no_clients(self):
        with patch("qdrant_client.QdrantClient.__init__", side_effect=AssertionError("client created")):
            for name in ("graph.store", "graph.retrieval", "mcp_server", "main"):
                importlib.reload(importlib.import_module(name))

    def test_search_only_uses_read_operations(self):
        store = make_store()
        add(store, 1, 0.9, path="a.md", project="p", tags=["t"])
        allowed = {"collection_exists", "get_collection", "query_points", "scroll"}
        real = store.client

        class Spy:
            def __getattr__(self, name):
                if name not in allowed:
                    raise AssertionError(f"retrieval called client.{name}")
                return getattr(real, name)

        store.client = Spy()
        hybrid_search(store, "q", filters={"project": "p"}, prefer={"tags": "t"}, max_per_source=1)
        hybrid_search(store, "", filters={"project": "p"})


class McpTests(Fixture):
    def setUp(self):
        super().setUp()
        import mcp_server
        self.mcp = mcp_server
        original = mcp_server._store
        mcp_server._store = lambda: self.store
        self.addCleanup(setattr, mcp_server, "_store", original)

    def test_semantic_search_tool_still_works(self):
        results = json.loads(self.mcp.semantic_search("q", limit=2, project=["knowledge-cortex"]))
        self.assertEqual([r["path"] for r in results], ["KC/b.md", "local_ingest/d.txt"])

    def test_hybrid_search_tool(self):
        data = json.loads(self.mcp.hybrid_search("q", limit=2, filters={"tags": ["architecture", "networking"]},
                                                 prefer={"project": "knowledge-cortex"}, max_per_source=1))
        self.assertEqual(paths(data), ["KC/b.md", "Net/a.md"])
        self.assertIn("Instruct:", self.store._http.inputs[-1])
        lean = json.loads(self.mcp.hybrid_search("q", limit=1, include_scores=False))
        self.assertNotIn("scores", lean["results"][0])
        self.assertIn("score", lean["results"][0])
        error = json.loads(self.mcp.hybrid_search("q", filters={"nope": 1}))
        self.assertIn("unknown metadata field", error["error"])

    def test_tools_are_registered_with_object_parameters(self):
        tools = {t.name: t for t in asyncio.run(self.mcp.server.list_tools())}
        self.assertIn("semantic_search", tools)
        schema = tools["hybrid_search"].input_schema["properties"]
        self.assertTrue({"query", "limit", "filters", "prefer", "max_per_source", "include_scores"} <= set(schema))


class CliTests(Fixture):
    def run_cli(self, query, **kwargs):
        from main import search_cli
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = search_cli(query, store=self.store, **kwargs)
        return code, buf.getvalue()

    def test_json_and_human_output(self):
        code, out = self.run_cli("q", limit=2, filters={"project": "knowledge-cortex"}, json_output=True)
        self.assertEqual(code, 0)
        self.assertEqual(paths(json.loads(out)), ["KC/b.md", "local_ingest/d.txt"])
        code, out = self.run_cli("q", limit=1, prefer={"tags": ["architecture"]})
        self.assertIn("KC/b.md#0", out)
        self.assertIn("preferred tags", out)
        code, out = self.run_cli("q", filters={"bad": 1}, json_output=True)
        self.assertEqual(code, 1)
        self.assertIn("error", json.loads(out))

    def test_parser_shortcuts_and_validation(self):
        from main import _search_arguments
        import argparse
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(filter_json='{"source": "obsidian"}', prefer_json=None,
                                  filter_project=["knowledge-cortex"], filter_source=None, tag=["a", "b"],
                                  prefer_project=None, prefer_tag=["x"])
        filters, prefer = _search_arguments(args, parser)
        self.assertEqual(filters, {"source": "obsidian", "project": "knowledge-cortex",
                                   "tags": {"contains_all": ["a", "b"]}})
        self.assertEqual(prefer, {"tags": ["x"]})
        for argv in (["--tag", "x"], ["--search", "q", "--filter", "[1]"],
                     ["--search", "q", "--filter", '{"project": "a"}', "--filter-project", "b"]):
            proc = subprocess.run([sys.executable, str(REPO / "main.py"), *argv], capture_output=True, text=True,
                                  cwd=REPO, env={"PATH": "", "QDRANT_URL": "http://127.0.0.1:9"})
            self.assertEqual(proc.returncode, 2, proc.stderr)


if __name__ == "__main__":
    unittest.main()
