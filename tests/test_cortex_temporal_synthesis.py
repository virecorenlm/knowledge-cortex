"""Phase D (temporal context) and Phase E (knowledge synthesis). Models are
fakes; Qdrant is in-memory; no live service is used."""
import hashlib
import json
import sqlite3

from cortex import synthesis, temporal, users, versions
from cortex.db import CortexError, to_bound
from graph.retrieval import hybrid_search
from tests.cortex_fixtures import MD, CortexFixture, FakeModel


class TemporalTests(CortexFixture):
    def setUp(self):
        super().setUp()
        self.seed()
        self.t0 = self.db.now()                       # after the first revision became current
        self.rev0 = self.doc()["current_revision_id"]
        self.edit_source("B\n", "B-second\n")
        self.sync()
        self.rev1 = self.doc()["current_revision_id"]
        self.t1 = self.db.now()
        self.edit_source("C\n", "C-third\n")
        self.sync()
        self.rev2 = self.doc()["current_revision_id"]

    def test_revision_selection_as_of(self):
        self.assertEqual(versions.revisions_current_at(self.db, self.t0), {self.doc_id: self.rev0})
        self.assertEqual(versions.revisions_current_at(self.db, self.t1), {self.doc_id: self.rev1})
        self.assertEqual(versions.revisions_current_at(self.db, self.db.now()), {self.doc_id: self.rev2})
        self.assertEqual(versions.revisions_current_at(self.db, "2000-01-01T00:00:00.000000Z"), {})

    def test_as_of_search_returns_the_historical_state(self):
        r = temporal.temporal_search(self.db, self.store, "local", "B second", as_of=self.t1)
        self.assertEqual(r["temporal"]["mode"], "historical")
        self.assertEqual({x["metadata"]["revision_id"] for x in r["results"]}, {self.rev1})
        self.assertIn("B-second", r["results"][0]["text"])
        self.assertNotIn("C-third", r["results"][0]["text"])
        old = temporal.temporal_search(self.db, self.store, "local", "x", as_of=self.t0)
        self.assertEqual({x["metadata"]["revision_id"] for x in old["results"]}, {self.rev0})

    def test_current_search_is_plain_hybrid_search(self):
        current = temporal.temporal_search(self.db, self.store, "local", "plan", limit=3)
        plain = hybrid_search(self.store, "plan", limit=3)
        self.assertEqual([r["id"] for r in current["results"]], [r["id"] for r in plain["results"]])
        for a, b in zip(current["results"], plain["results"]):  # float32 noise in local Qdrant ~1e-7
            self.assertAlmostEqual(a["score"], b["score"], places=5)
        self.assertEqual(current["temporal"], {"mode": "current"})

    def test_before_after_between_windows(self):
        everything = temporal.changes(self.db, "local")
        self.assertEqual([c["revision_id"] for c in everything], [self.rev0, self.rev1, self.rev2])
        self.assertEqual([c["revision_id"] for c in temporal.changes(self.db, "local", after=self.t1)], [self.rev2])
        self.assertEqual([c["revision_id"] for c in temporal.changes(self.db, "local", before=self.t0)], [self.rev0])
        between = temporal.changes(self.db, "local", after=self.t0, before=self.t1)
        self.assertEqual([c["revision_id"] for c in between], [self.rev1])
        self.assertEqual((between[0]["lines_added"], between[0]["lines_removed"]), (1, 1))
        r = temporal.temporal_search(self.db, self.store, "local", "third", changed_after=self.t1)
        self.assertEqual({x["metadata"]["revision_id"] for x in r["results"]}, {self.rev2})

    def test_date_only_bounds_cover_whole_days(self):
        self.assertEqual(to_bound("2026-01-01", end_of_day=True), "2026-01-01T23:59:59.999999Z")
        self.assertEqual(len(temporal.changes(self.db, "local", after="2026-01-01", before="2026-01-01")), 3)
        self.assertEqual(temporal.changes(self.db, "local", after="2026-01-02"), [])
        with self.assertRaises(CortexError):
            temporal.changes(self.db, "local", after="last tuesday")

    def test_missing_valid_time_is_unknown_never_guessed(self):
        r = temporal.temporal_search(self.db, self.store, "local", "x", valid_at="2026-01-01")
        self.assertEqual(r["results"], [])
        self.assertEqual(r["temporal"]["unknown_validity_revisions"], 1)
        r = temporal.temporal_search(self.db, self.store, "local", "x", valid_at="2026-01-01",
                                     include_unknown_validity=True)
        self.assertTrue(r["results"])

    def test_declared_valid_time_is_used(self):
        self.edit_source("---\ntitle: Plan\nproject: kc\n---\n",
                         "---\ntitle: Plan\nproject: kc\nvalid_from: 2025-01-01\nvalid_to: 2025-12-31\n---\n")
        self.sync()
        rev = versions.get_revision(self.db, self.doc()["current_revision_id"])
        self.assertEqual((rev["valid_from"], rev["valid_to"]),
                         ("2025-01-01T00:00:00.000000Z", "2025-12-31T23:59:59.999999Z"))
        inside = temporal.temporal_search(self.db, self.store, "local", "x", valid_at="2025-06-01")
        self.assertEqual({x["metadata"]["revision_id"] for x in inside["results"]}, {rev["revision_id"]})
        outside = temporal.temporal_search(self.db, self.store, "local", "x", valid_at="2026-06-01")
        self.assertEqual(outside["results"], [])

    def test_deleted_documents_drop_out_of_as_of_after_deletion(self):
        import asyncio
        self.source.unlink()
        tid = self.result()["tombstone_id"]
        asyncio.run(self.engine.approve_tombstone("local", tid))
        self.assertEqual(versions.revisions_current_at(self.db, self.db.now()), {})
        self.assertEqual(versions.revisions_current_at(self.db, self.t1), {self.doc_id: self.rev1})

    def test_visibility_applies_to_current_and_historical_search(self):
        users.add_user(self.db, "local", "bob", "member")
        self.assertEqual(temporal.temporal_search(self.db, self.store, "bob", "x")["results"], [])
        self.assertEqual(temporal.temporal_search(self.db, self.store, "bob", "x", as_of=self.t1)["results"], [])
        self.assertEqual(temporal.changes(self.db, "bob"), [])
        users.set_visibility(self.db, "local", self.doc_id, "shared")
        self.assertTrue(temporal.temporal_search(self.db, self.store, "bob", "x")["results"])


def _items(*statements):
    return {"items": [{"statement": s, "evidence": ev} for s, ev in statements]}


class SynthesisTests(CortexFixture):
    def setUp(self):
        super().setUp()
        self.seed(content="---\ntitle: Plan\nproject: kc\n---\n\nWe decided to use Qdrant.\nOpen: backups?\n")

    def synth(self, model, query="plan", stype="SUMMARY", actor="local", **kw):
        return synthesis.synthesize(self.db, self.store, actor, query, stype, model=model, **kw)

    def test_synthesis_retains_exact_provenance_and_model_metadata(self):
        model = FakeModel(_items(("The team chose Qdrant.", ["E1"])))
        result = self.synth(model, stype="DECISION")
        self.assertEqual(result["status"], "created")
        s = result["synthesis"]
        self.assertEqual((s["type"], s["model"], s["provider"], s["prompt_version"], s["status"]),
                         ("DECISION", "fake-model", "fake", synthesis.PROMPT_VERSION, "current"))
        e = s["evidence"][0]
        self.assertEqual(e["document_id"], self.doc_id)
        self.assertEqual(e["revision_id"], self.doc()["current_revision_id"])
        point = self.store.client.retrieve(self.store.collection, ids=[e["point_id"]], with_payload=True)[0]
        self.assertEqual(e["content_hash"], hashlib.sha256(point.payload["text"].encode()).hexdigest())
        self.assertIsNotNone(e["score"])
        self.assertIn("[E1]", s["body"])
        self.assertIn("not source evidence", s["body"])
        self.assertEqual(s["items"], [{"statement": "The team chose Qdrant.", "evidence": ["E1"]}])

    def test_no_evidence_is_a_safe_refusal(self):
        model = FakeModel(_items(("x", ["E1"])))
        result = self.synth(model, filters={"project": "nothing-here"})
        self.assertEqual(result["status"], "no_evidence")
        self.assertEqual(model.calls, [])
        self.assertEqual(synthesis.list_syntheses(self.db, "local"), [])

    def test_uncited_or_unknown_statements_are_rejected(self):
        model = FakeModel({"items": [{"statement": "Supported", "evidence": ["E1"]},
                                     {"statement": "Invented", "evidence": ["E9"]},
                                     {"statement": "Uncited", "evidence": []}]})
        result = self.synth(model)
        self.assertEqual([i["statement"] for i in result["synthesis"]["items"]], ["Supported"])
        self.assertEqual(len(result["rejected"]), 2)
        nothing = self.synth(FakeModel({"items": [{"statement": "Invented", "evidence": ["E9"]}]}), query="other")
        self.assertEqual(nothing["status"], "no_supported_statements")
        self.assertEqual(self.synth(FakeModel("not json at all"), query="third")["status"], "no_supported_statements")

    def test_open_question_and_consensus_artifacts(self):
        q = self.synth(FakeModel(_items(("How are backups handled?", ["E1"]))), stype="OPEN_QUESTION")
        self.assertEqual(q["synthesis"]["type"], "OPEN_QUESTION")
        self.assertIn("backups", q["synthesis"]["body"])
        c = self.synth(FakeModel(_items(("Qdrant is the store.", ["E1"]))), stype="consensus")
        self.assertEqual(c["synthesis"]["type"], "CONSENSUS")
        with self.assertRaises(CortexError):
            self.synth(FakeModel({}), stype="POEM")

    def test_contradiction_artifact_defaults_to_unresolved(self):
        model = FakeModel({"items": [{"claim_a": "Qdrant was chosen", "claim_b": "Backups are open",
                                      "evidence_a": ["E1"], "evidence_b": ["E1"]}]})
        s = self.synth(model, stype="CONTRADICTION")["synthesis"]
        full = synthesis.read_synthesis(self.db, "local", s["synthesis_id"])
        [c] = full["contradictions"]
        self.assertEqual((c["claim_a"], c["status"]), ("Qdrant was chosen", "unresolved"))
        self.assertEqual(c["evidence_a"][0]["document_id"], self.doc_id)
        resolved = synthesis.resolve_contradiction(self.db, "local", c["contradiction_id"], "not_conflicting",
                                                   note="different topics")
        self.assertEqual((resolved["status"], resolved["resolved_by"]), ("not_conflicting", "local"))
        with self.assertRaises(sqlite3.DatabaseError):
            self.db.conn.execute("UPDATE contradictions SET claim_a = 'tampered'")

    def test_contradiction_with_non_overlapping_validity_is_time_dependent(self):
        status, basis = synthesis._contradiction_status(
            self.db, [{"valid_from": "2024-01-01T00:00:00.000000Z", "valid_to": "2024-06-30T23:59:59.999999Z"}],
            [{"valid_from": "2025-01-01T00:00:00.000000Z", "valid_to": "2025-12-31T23:59:59.999999Z"}])
        self.assertEqual(status, "time_dependent")
        self.edit_source("Open: backups?", "Open: none")
        self.sync()
        old, new = versions.history(self.db, self.doc_id)[0], versions.history(self.db, self.doc_id)[-1]
        status, _ = synthesis._contradiction_status(self.db, [{"revision_id": old["revision_id"]}],
                                                    [{"revision_id": new["revision_id"]}])
        self.assertEqual(status, "superseded")

    def test_identity_is_deterministic_and_idempotent(self):
        model = FakeModel(_items(("Qdrant.", ["E1"])))
        first = self.synth(model)
        second = self.synth(model)
        self.assertEqual(second["status"], "existing")
        self.assertEqual(second["synthesis"]["synthesis_id"], first["synthesis"]["synthesis_id"])
        self.assertEqual(len(model.calls), 1)
        other_model = FakeModel(_items(("Qdrant.", ["E1"])), name="other-model")
        self.assertNotEqual(self.synth(other_model)["synthesis"]["synthesis_id"], first["synthesis"]["synthesis_id"])

    def test_changed_evidence_marks_stale_and_resynthesis_supersedes(self):
        model = FakeModel(_items(("Qdrant.", ["E1"])))
        sid = self.synth(model)["synthesis"]["synthesis_id"]
        self.assertFalse(synthesis.check_synthesis(self.db, self.store, sid)["stale"])
        self.edit_source("We decided to use Qdrant.", "We decided to use Qdrant with snapshots.")
        self.sync()
        check = synthesis.check_synthesis(self.db, self.store, sid)
        self.assertTrue(check["stale"])
        self.assertEqual(synthesis.get_synthesis(self.db, sid)["status"], "stale")
        new = synthesis.resynthesize(self.db, self.store, "local", sid, model=model)
        self.assertNotEqual(new["synthesis"]["synthesis_id"], sid)
        old = synthesis.get_synthesis(self.db, sid)
        self.assertEqual((old["status"], old["superseded_by"]), ("superseded", new["synthesis"]["synthesis_id"]))

    def test_synthesis_content_and_provenance_are_immutable(self):
        sid = self.synth(FakeModel(_items(("Qdrant.", ["E1"]))))["synthesis"]["synthesis_id"]
        for sql in ("UPDATE syntheses SET body = 'x'", "UPDATE syntheses SET evidence = '[]'",
                    "DELETE FROM syntheses"):
            with self.subTest(sql=sql), self.assertRaises(sqlite3.DatabaseError):
                self.db.conn.execute(sql)
        self.assertEqual(synthesis.get_synthesis(self.db, sid)["status"], "current")

    def test_synthesis_never_modifies_evidence(self):
        before_files = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file() and "cortex.db" not in p.name}
        before_points = sorted((str(p.id), json.dumps(p.payload, sort_keys=True)) for p in self.points())
        self.synth(FakeModel(_items(("Qdrant.", ["E1"]))), stype="PROJECT_STATE")
        self.assertEqual({p: p.read_bytes() for p in before_files}, before_files)
        self.assertEqual(sorted((str(p.id), json.dumps(p.payload, sort_keys=True)) for p in self.points()),
                         before_points)

    def test_change_summary_uses_revision_diffs_as_evidence(self):
        self.edit_source("Open: backups?", "Open: backups are nightly")
        self.sync()
        model = FakeModel(_items(("Backups became nightly.", ["E2"])))
        s = self.synth(model, query="", stype="CHANGE_SUMMARY")["synthesis"]
        diff_ev = [e for e in s["evidence"] if e["kind"] == "revision_diff"]
        self.assertEqual(len(diff_ev), 2)
        self.assertIn("+Open: backups are nightly", diff_ev[1]["text"])
        self.assertEqual(diff_ev[1]["revision_id"], self.doc()["current_revision_id"])

    def test_private_evidence_limits_who_can_read_a_synthesis(self):
        users.add_user(self.db, "local", "bob", "member")
        sid = self.synth(FakeModel(_items(("Qdrant.", ["E1"]))))["synthesis"]["synthesis_id"]
        self.assertEqual(synthesis.list_syntheses(self.db, "bob"), [])
        with self.assertRaises(users.AuthorizationError):
            synthesis.read_synthesis(self.db, "bob", sid)
        self.assertEqual(self.synth(FakeModel(_items(("x", ["E1"]))), actor="bob")["status"], "no_evidence")
