"""Shared disposable environment for Cortex tests: temp source dir, a
directory-backed vault, in-memory Qdrant with a fake embedder, a temp Cortex
store with a deterministic clock, and the real ingestion pipeline."""
import asyncio
import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient

from cortex.db import CortexDB
from cortex.sync_engine import SyncEngine
from cortex.vault_fs import FilesystemVault
from graph.store import VectorStore
from ingest.markdown import FRONTMATTER_RE
from main import index_to_qdrant
from tests.test_main_cli import FakeOllama

MD = "---\ntitle: Plan\nproject: kc\n---\n\nA\nB\nC\nD\nE\n"


class Clock:
    def __init__(self, start="2026-01-01T00:00:00"):
        self.t = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FakeModel:
    """Deterministic stand-in for an LLM: returns a canned response (or the
    result of a function of the prompt) and records calls."""
    provider = "fake"

    def __init__(self, response=None, name="fake-model"):
        self.name = name
        self.response = response
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        if callable(self.response):
            return self.response(system, user)
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


class CortexFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.src_dir = self.root / "src"
        self.src_dir.mkdir()
        (self.root / "vault").mkdir()
        self.vault = FilesystemVault(self.root / "vault")
        self.state = self.root / "sync_state.json"
        self.store = VectorStore(ollama_client=FakeOllama(), qdrant_client=QdrantClient(location=":memory:"),
                                 collection="cortex_test", embedding_model="fake")
        self.clock = Clock()
        self.db = CortexDB(self.root / "state" / "cortex.db", clock=self.clock)
        self.addCleanup(self.db.close)
        self.engine = SyncEngine(self.db, self.vault, self.state, store=self.store,
                                 proposals_dir=self.root / "state" / "proposals")

    # -- helpers -------------------------------------------------------------
    def ingest(self, name="plan.md", content=MD):
        path = self.src_dir / name
        path.write_text(content, encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            index_to_qdrant(str(path), None, self.state, store=self.store, write_vault=True, obsidian=self.vault)
        return path

    def seed(self, name="plan.md", content=MD):
        self.source = self.ingest(name, content)
        report = self.sync()
        self.doc_id = self.db.one("SELECT document_id FROM documents WHERE source_path = ?",
                                  (str(self.source.resolve()),))["document_id"]
        return report

    def sync(self, actor="local", **kw):
        return asyncio.run(self.engine.sync(actor, **kw))

    def result(self, report=None):
        report = report or self.sync()
        return next(d for d in report["documents"] if d["document_id"] == self.doc_id)

    def doc(self):
        return dict(self.db.one("SELECT * FROM documents WHERE document_id = ?", (self.doc_id,)))

    @property
    def note_path(self):
        return self.root / "vault" / self.doc()["vault_path"]

    def note_body(self):
        content = self.note_path.read_text(encoding="utf-8")
        return content[FRONTMATTER_RE.match(content).end():]

    def edit_note(self, old, new):
        content = self.note_path.read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(content)
        body = content[match.end():]
        assert old in body, (old, body)
        self.note_path.write_text(content[:match.end()] + body.replace(old, new, 1), encoding="utf-8")

    def edit_source(self, old, new):
        text = self.source.read_text(encoding="utf-8")
        assert old in text
        self.source.write_text(text.replace(old, new, 1), encoding="utf-8")

    def points(self):
        if not self.store.client.collection_exists(self.store.collection):
            return []
        pts, _ = self.store.client.scroll(self.store.collection, limit=10000, with_payload=True)
        return pts
