"""Phase H interfaces (CLI + MCP) and compatibility: existing retrieval and
proposal behavior, actor recording, and no mutation on import."""
import asyncio
import io
import json
import os
from contextlib import redirect_stdout
from unittest.mock import patch

import cortex_cli
from cortex import users
from cortex.api import CortexService
from graph.retrieval import hybrid_search
from main import _decision_cli, apply_proposal_cli
from tests.cortex_fixtures import CortexFixture, FakeModel


class ServiceFixture(CortexFixture):
    def setUp(self):
        super().setUp()
        self.db.close()
        self.svc = CortexService(db_path=self.root / "state" / "cortex.db", state_path=self.state, vault=self.vault,
                                 proposals_dir=self.root / "state" / "proposals", store=self.store,
                                 model=FakeModel({"items": [{"statement": "Plan uses A to E.", "evidence": ["E1"]}]}),
                                 clock=self.clock)
        self.addCleanup(self.svc.close)
        self.db, self.engine = self.svc.db, self.svc.engine
        self.seed()

    def cli(self, *argv, user="local"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cortex_cli.run(["--user", user, *argv], service=self.svc)
        return code, json.loads(buf.getvalue())


class CliTests(ServiceFixture):
    def test_status_history_revision_and_diff(self):
        code, out = self.cli("sync-status")
        self.assertEqual((code, out["documents"][0]["state"]), (0, "IN_SYNC"))
        self.edit_source("A\n", "A1\n")
        code, out = self.cli("sync")
        self.assertEqual(out["documents"][0]["status"], "ok")
        code, hist = self.cli("history", self.doc_id)
        ids = [r["revision_id"] for r in hist["revisions"]]
        self.assertEqual(len(ids), 2)
        code, rev = self.cli("show-revision", ids[1])
        self.assertEqual(rev["origin"], "source")
        code, diff = self.cli("diff-revisions", ids[0], ids[1])
        self.assertIn("+A1", diff["diff"])
        code, verify = self.cli("verify")
        self.assertEqual(verify, {"journal": [], "pending_operations": [], "revision_graph": []})

    def test_dry_run_edit_restore_and_conflict_commands(self):
        code, out = self.cli("sync", "--dry-run")
        self.assertIn("planned_action", out["documents"][0])
        body = self.root / "body.md"
        body.write_text("A\nB\nC-cli\nD\nE\n")
        r0 = self.doc()["current_revision_id"]
        code, out = self.cli("edit", self.doc_id, "--file", str(body), "--expected-revision", r0)
        self.assertEqual(out["status"], "applied")
        body.write_text("A\nB\nC-stale\nD\nE\n")
        code, out = self.cli("edit", self.doc_id, "--file", str(body), "--expected-revision", r0)
        self.assertEqual(out["status"], "conflict")
        cid = out["conflict_id"]
        code, listed = self.cli("conflicts", "--status", "open")
        self.assertEqual([c["conflict_id"] for c in listed["conflicts"]], [cid])
        code, out = self.cli("resolve-conflict", cid, "--action", "accept_right")
        self.assertEqual(out["status"], "resolved")
        code, out = self.cli("restore-revision", r0)
        self.assertEqual(out["status"], "restored")
        self.assertEqual(self.note_body(), "A\nB\nC\nD\nE\n")

    def test_synthesis_temporal_and_user_commands(self):
        code, out = self.cli("synthesize", "plan", "--type", "summary")
        sid = out["synthesis"]["synthesis_id"]
        code, listed = self.cli("list-syntheses")
        self.assertEqual([s["synthesis_id"] for s in listed["syntheses"]], [sid])
        code, check = self.cli("check-synthesis", sid)
        self.assertFalse(check["stale"])
        code, out = self.cli("temporal-search", "plan", "--as-of", self.db.now())
        self.assertTrue(out["results"])
        code, out = self.cli("changes")
        self.assertEqual(len(out["changes"]), 1)
        code, out = self.cli("add-user", "bob", "--role", "member")
        self.assertEqual(out["role"], "member")
        code, out = self.cli("history", self.doc_id, user="bob")
        self.assertEqual((code, out["code"]), (1, "unauthorized"))
        code, out = self.cli("add-user", "eve", user="bob")
        self.assertEqual(code, 1)
        code, out = self.cli("set-visibility", self.doc_id, "shared")
        code, out = self.cli("history", self.doc_id, user="bob")
        self.assertEqual(code, 0)
        code, out = self.cli("sync-status", user="nobody")
        self.assertEqual((code, out["code"]), (1, "invalid_actor"))

    def test_tombstone_commands(self):
        self.note_path.unlink()
        code, out = self.cli("sync")
        tid = out["documents"][0]["tombstone_id"]
        code, out = self.cli("tombstones")
        self.assertEqual(out["tombstones"][0]["status"], "detected")
        code, out = self.cli("restore-tombstone", tid)
        self.assertEqual(out["status"], "restored")


class McpTests(ServiceFixture):
    def setUp(self):
        super().setUp()
        import mcp_server
        self.mcp = mcp_server
        patcher = patch.object(mcp_server, "_cortex", lambda: (self.svc, False))
        patcher.start()
        self.addCleanup(patcher.stop)

    def call(self, name, **kw):
        return json.loads(getattr(self.mcp, name)(**kw))

    def test_read_tools(self):
        self.assertEqual(self.call("sync_status")["documents"][0]["state"], "IN_SYNC")
        hist = self.call("history", document_id=self.doc_id)
        rid = hist["revisions"][0]["revision_id"]
        self.assertEqual(self.call("get_revision", revision_id=rid)["document_id"], self.doc_id)
        self.assertTrue(self.call("temporal_search", query="plan", as_of=self.db.now())["results"])
        self.assertIn("error", self.call("get_revision", revision_id="nope"))

    def test_mutating_tools_follow_the_same_rules(self):
        self.edit_source("A\n", "A1\n")
        self.assertEqual(self.call("sync_document", document_id=self.doc_id)["status"], "ok")
        self.edit_source("C\n", "C-src\n")
        self.edit_note("C\n", "C-vault\n")
        cid = self.call("sync_document", document_id=self.doc_id)["conflict_id"]
        self.assertEqual([c["conflict_id"] for c in self.call("list_conflicts")], [cid])
        self.assertEqual(self.call("get_conflict", conflict_id=cid)["status"], "open")
        bad = self.call("resolve_conflict", conflict_id=cid, action="manual", body="")
        self.assertEqual(bad["code"], "invalid_body")
        self.assertEqual(self.call("resolve_conflict", conflict_id=cid, action="accept_vault")["status"], "resolved")
        created = self.call("synthesize", query="plan")
        sid = created["synthesis"]["synthesis_id"]
        self.assertEqual(self.call("get_synthesis", synthesis_id=sid)["synthesis_id"], sid)
        self.assertEqual(len(self.call("list_syntheses")), 1)

    def test_actor_is_server_configured_and_enforced(self):
        users.add_user(self.db, "local", "bob", "member")
        self.svc.user = "bob"
        self.assertEqual(self.call("sync_status")["documents"], [])
        self.assertEqual(self.call("history", document_id=self.doc_id)["code"], "unauthorized")
        self.assertEqual(self.call("sync_document", document_id=self.doc_id)["status"], "skipped_unauthorized")
        tools = {t.name: t for t in asyncio.run(self.mcp.server.list_tools())}
        for name in ("sync_status", "sync_document", "history", "get_revision", "temporal_search", "synthesize",
                     "list_conflicts", "get_conflict", "resolve_conflict", "list_syntheses", "get_synthesis",
                     "semantic_search", "hybrid_search"):
            self.assertIn(name, tools)
            self.assertNotIn("user", tools[name].input_schema["properties"])


class CompatibilityTests(ServiceFixture):
    def test_hybrid_search_unchanged_and_visibility_hook(self):
        plain = hybrid_search(self.store, "plan", limit=5)
        self.assertEqual(plain["mode"], "semantic")
        self.assertEqual(plain["results"][0]["metadata"]["document_id"], self.doc_id)
        hidden = hybrid_search(self.store, "plan", limit=5, exclude_document_ids=[self.doc_id])
        self.assertEqual(hidden["results"], [])
        filtered = hybrid_search(self.store, "plan", filters={"document_id": self.doc_id})
        self.assertEqual(len(filtered["results"]), len(plain["results"]))
        with self.assertRaises(ValueError):
            hybrid_search(self.store, "plan", filters={"document_id": {"startswith": "x"}})

    def test_semantic_search_store_api_unchanged(self):
        results = self.store.search("plan", limit=2)
        self.assertIn("score", results[0])
        self.assertIn("text", results[0])

    def test_proposal_actions_record_actor_and_enforce_users(self):
        users.add_user(self.db, "local", "bob", "member")
        self.edit_note("E\n", "E-vault\n")
        pid = self.result()["proposal_id"]
        pdir = self.root / "state" / "proposals"
        with redirect_stdout(io.StringIO()) as buf:
            code = _decision_cli(pdir, pid, "approve", None, obsidian=self.vault, state_path=self.state, user="bob",
                                 cortex_db=self.db.path)
        self.assertEqual(code, 1)
        self.assertIn("may not approve", buf.getvalue())
        proposal = json.loads((pdir / f"{pid}.json").read_text())
        self.assertEqual(proposal["status"], "pending")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(_decision_cli(pdir, pid, "approve", None, obsidian=self.vault, state_path=self.state,
                                           user="local", cortex_db=self.db.path), 0)
            self.assertEqual(apply_proposal_cli(pid, proposals_dir=pdir, state_path=self.state, obsidian=self.vault,
                                                user="bob", cortex_db=self.db.path), 1)
        self.assertNotIn("E-vault", self.source.read_text())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(apply_proposal_cli(pid, proposals_dir=pdir, state_path=self.state, obsidian=self.vault,
                                                user="local", cortex_db=self.db.path), 0)
        proposal = json.loads((pdir / f"{pid}.json").read_text())
        self.assertEqual((proposal["decision"]["actor"], proposal["apply"]["actor"]), ("local", "local"))

    def test_old_proposal_records_without_actor_still_load(self):
        from ingest.proposals import load_proposal
        self.edit_note("E\n", "E-vault\n")
        pid = self.result()["proposal_id"]
        pdir = self.root / "state" / "proposals"
        data = json.loads((pdir / f"{pid}.json").read_text())
        self.assertNotIn("actor", data["decision"])
        proposal, error = load_proposal(pdir, pid)
        self.assertIsNone(error)

    def test_default_user_without_store_is_unchanged_single_user_behavior(self):
        from main import authorize_proposal_actor
        missing = self.root / "no" / "cortex.db"
        self.assertIsNone(authorize_proposal_actor(self.root, "x", "local", "approve", missing))
        self.assertFalse(missing.exists())
        self.assertIn("unknown user", authorize_proposal_actor(self.root, "x", "bob", "approve", missing))


class NoMutationOnImportTests(ServiceFixture):
    def test_importing_modules_opens_no_database_and_no_clients(self):
        # Fresh interpreter: reloading modules in-process would swap class
        # objects (e.g. CortexError) under other test modules.
        import subprocess
        import sys
        code = (
            "import sqlite3, qdrant_client\n"
            "def boom(*a, **k): raise AssertionError('opened on import')\n"
            "sqlite3.connect = boom\n"
            "qdrant_client.QdrantClient.__init__ = boom\n"
            "import cortex.db, cortex.users, cortex.versions, cortex.identity, cortex.merge, cortex.journal\n"
            "import cortex.vault_fs, cortex.sync_engine, cortex.conflicts, cortex.temporal, cortex.synthesis\n"
            "import cortex.api, cortex_cli, mcp_server, main\n"
            "print('clean')\n")
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True)
        self.assertEqual(proc.stdout.strip(), "clean", proc.stderr)

    def test_tests_never_touch_the_default_production_store(self):
        from cortex.db import DEFAULT_DB_PATH
        self.assertNotEqual(os.path.realpath(self.db.path), os.path.realpath(DEFAULT_DB_PATH))
