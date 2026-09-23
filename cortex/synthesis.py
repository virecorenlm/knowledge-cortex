"""Knowledge synthesis over retrieved evidence -- never a replacement for it.

    query/topic -> temporal_search/hybrid retrieval (visibility-filtered)
                -> evidence set E1..En (exact provenance per item)
                -> model (JSON items, each citing evidence refs)
                -> validation (uncited/unknown-ref items are dropped)
                -> stored artifact (immutable content, status current|stale|superseded)

Types: SUMMARY, DECISION, CONCEPT, RELATIONSHIP, CONTRADICTION, OPEN_QUESTION,
CONSENSUS, CHANGE_SUMMARY (evidence = revision diffs in a time window),
PROJECT_STATE.

Provenance: every evidence item records document_id, revision_id (when the
chunk belongs to a Cortex document), the Qdrant point id, path, chunk index,
sha256 of the exact chunk text, and the retrieval scores. Every stored
statement cites at least one evidence ref; statements that cite nothing or an
unknown ref are rejected, never stored as if supported. The body is marked as
model-generated and lists its evidence.

Identity: synthesis_id = sha256(type, query, params, evidence fingerprint,
model, provider, prompt_version)[:24]. The same question over the same
evidence with the same model/prompt returns the existing artifact instead of
creating another.

Staleness: check_synthesis() re-verifies each evidence item (current-mode:
the document's current revision is still the cited one and the point still
holds the same text; as_of mode: the cited revisions are immutable). Any
drift marks the synthesis stale; resynthesize() creates a new artifact and
marks the old one superseded. Synthesis never writes sources, notes, or the
evidence index.

Contradictions: CONTRADICTION items become contradiction records (claims +
evidence). Status is only decided deterministically when metadata proves it
(non-overlapping declared validity -> time_dependent; one claim's revision is
an ancestor of the other's -> superseded); otherwise unresolved, for a human.

The model is pluggable (anything with .name, .provider, .generate(system,
user)); OllamaChatModel reuses the project's Ollama chat setup. Unit tests
use fakes; no live model is required.
"""

import hashlib
import json
import os
import re

import requests

from cortex import identity, versions
from cortex.db import CortexError, NotFound, canonical_json, row_dict
from cortex.users import AuthorizationError, can, resolve_actor

PROMPT_VERSION = 1
TYPES = {
    "SUMMARY": "Summarize what the evidence says about the topic.",
    "DECISION": "List decisions that the evidence records (what was decided, and when if stated).",
    "CONCEPT": "Explain the key concepts the evidence defines or uses.",
    "RELATIONSHIP": "Describe relationships between entities/ideas stated in the evidence.",
    "CONTRADICTION": "Find pairs of claims in the evidence that conflict. For each pair give claim_a, claim_b, "
                     "evidence_a and evidence_b. Report only real, explicit conflicts.",
    "OPEN_QUESTION": "List questions the evidence raises or leaves unanswered.",
    "CONSENSUS": "State points on which several evidence items agree.",
    "CHANGE_SUMMARY": "Summarize what changed, based on the revision diffs given as evidence.",
    "PROJECT_STATE": "Describe the current state of the project: status, decisions, open work, risks.",
}
SYSTEM_PROMPT = """You synthesize knowledge strictly from the numbered evidence you are given.
Rules: use only facts present in the evidence; never invent names, numbers, dates or claims; every item
must cite the evidence refs it relies on (like "E1"); if the evidence does not support an item, omit it.
Respond with JSON only: {"items": [{"statement": "...", "evidence": ["E1", "E2"]}]}
For contradictions respond: {"items": [{"claim_a": "...", "claim_b": "...", "evidence_a": ["E1"],
"evidence_b": ["E2"]}]}
Task: """
_JSON = ("params", "evidence", "items")


class OllamaChatModel:
    provider = "ollama"

    def __init__(self, name=None, url=None, timeout=300, http=None):
        from ingest.ai_struct import DEFAULT_MODEL, DEFAULT_OLLAMA_URL
        self.name = name or os.getenv("CORTEX_SYNTHESIS_MODEL") or os.getenv("AI_STRUCTURE_MODEL", DEFAULT_MODEL)
        self.url = (url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip("/")
        self.timeout = timeout
        self.http = http or requests

    def generate(self, system, user):
        response = self.http.post(f"{self.url}/api/chat", json={
            "model": self.name, "stream": False, "format": "json",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def gather_evidence(db, store, actor, query, synthesis_type, filters=None, prefer=None, limit=8, as_of=None,
                    changed_after=None, changed_before=None, instruct=None):
    from cortex import temporal
    if synthesis_type == "CHANGE_SUMMARY":
        evidence = []
        for i, change in enumerate(temporal.changes(db, actor, changed_after, changed_before,
                                                    project=(filters or {}).get("project")
                                                    if isinstance((filters or {}).get("project"), str) else None)[:limit], 1):
            rev = versions.get_revision(db, change["revision_id"])
            parent = versions.get_revision(db, change["parent_revision_id"]) if change["parent_revision_id"] else None
            diff = versions.diff_revisions(db, parent["revision_id"], rev["revision_id"])["diff"] if parent else \
                ["(first revision)"] + rev["content"].splitlines()
            text = "\n".join(diff)
            evidence.append({"ref": f"E{i}", "kind": "revision_diff", "document_id": rev["document_id"],
                             "revision_id": rev["revision_id"], "parent_revision_id": change["parent_revision_id"],
                             "point_id": None, "path": rev["source_path"], "chunk_index": None,
                             "content_hash": _sha(text), "revision_content_hash": rev["content_hash"],
                             "score": None, "text": text, "valid_from": rev["valid_from"], "valid_to": rev["valid_to"],
                             "became_current_at": change["became_current_at"]})
        return evidence, "changes"
    response = temporal.temporal_search(db, store, actor, query, as_of=as_of, changed_after=changed_after,
                                        changed_before=changed_before, filters=filters, prefer=prefer, limit=limit,
                                        instruct=instruct)
    mode = response.get("temporal", {}).get("mode", "current")
    evidence = []
    for i, r in enumerate(response["results"], 1):
        meta = r["metadata"]
        evidence.append({"ref": f"E{i}", "kind": "chunk", "document_id": meta.get("document_id"),
                         "revision_id": meta.get("revision_id"), "point_id": r["id"], "path": r["path"],
                         "source_file": r.get("source_file"), "chunk_index": r["chunk_index"],
                         "content_hash": _sha(r["text"]), "score": r["scores"]["final"],
                         "semantic_score": r["scores"]["semantic"], "text": r["text"],
                         "doc_date": meta.get("doc_date"), "valid_from": meta.get("valid_from"),
                         "valid_to": meta.get("valid_to"), "collection": response.get("collection")})
    return evidence, mode


def _fingerprint(evidence):
    return _sha(canonical_json([[e["ref"], e.get("point_id"), e.get("revision_id"), e["content_hash"]]
                                for e in evidence]))


def _parse_items(text, synthesis_type, refs):
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except ValueError:
        data = None
    raw = data.get("items") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return [], [{"reason": "model output was not the required JSON object"}]
    accepted, rejected = [], []
    for item in raw:
        if not isinstance(item, dict):
            rejected.append({"item": item, "reason": "not an object"})
            continue
        if synthesis_type == "CONTRADICTION":
            a, b = item.get("evidence_a"), item.get("evidence_b")
            ok = all(isinstance(x, str) and x.strip() for x in (item.get("claim_a"), item.get("claim_b"))) and \
                all(isinstance(ev, list) and ev and set(ev) <= refs for ev in (a, b))
            if ok:
                accepted.append({"claim_a": item["claim_a"].strip(), "claim_b": item["claim_b"].strip(),
                                 "evidence_a": sorted(set(a)), "evidence_b": sorted(set(b))})
            else:
                rejected.append({"item": item, "reason": "claims need text and known evidence refs on both sides"})
            continue
        ev = item.get("evidence")
        if isinstance(item.get("statement"), str) and item["statement"].strip() and isinstance(ev, list) and ev \
                and all(isinstance(x, str) for x in ev) and set(ev) <= refs:
            accepted.append({"statement": item["statement"].strip(), "evidence": sorted(set(ev))})
        else:
            rejected.append({"item": item, "reason": "statement must cite at least one known evidence ref"})
    return accepted, rejected


def _render(synthesis_type, query, items, evidence, model):
    lines = [f"# {synthesis_type.replace('_', ' ').title()}: {query}", "",
             f"> Synthesized by {model.provider}:{model.name} from {len(evidence)} evidence item(s). "
             "This is derived knowledge, not source evidence; every point cites its evidence.", ""]
    for item in items:
        if synthesis_type == "CONTRADICTION":
            lines.append(f"- **A:** {item['claim_a']} [{', '.join(item['evidence_a'])}]")
            lines.append(f"  **B:** {item['claim_b']} [{', '.join(item['evidence_b'])}]")
        else:
            lines.append(f"- {item['statement']} [{', '.join(item['evidence'])}]")
    lines += ["", "## Evidence", ""]
    for e in evidence:
        where = f"{e['path']}#{e['chunk_index']}" if e.get("chunk_index") is not None else e["path"]
        lines.append(f"- {e['ref']}: {where} (document {e.get('document_id')}, revision {e.get('revision_id')}, "
                     f"sha256 {e['content_hash'][:12]})")
    return "\n".join(lines) + "\n"


def synthesize(db, store, actor, query, synthesis_type="SUMMARY", model=None, filters=None, prefer=None, limit=8,
               as_of=None, changed_after=None, changed_before=None, instruct=None):
    resolve_actor(db, actor)
    synthesis_type = (synthesis_type or "").upper()
    if synthesis_type not in TYPES:
        raise CortexError(f"unknown synthesis type {synthesis_type!r}; one of {sorted(TYPES)}", code="invalid_type")
    if synthesis_type != "CHANGE_SUMMARY" and not (query or "").strip():
        raise CortexError("a query/topic is required", code="invalid_query")
    model = model or OllamaChatModel()
    evidence, mode = gather_evidence(db, store, actor, query or "", synthesis_type, filters, prefer, limit, as_of,
                                     changed_after, changed_before, instruct)
    if not evidence:
        return {"status": "no_evidence", "reason": "no readable evidence matched; nothing was synthesized"}
    params = {"filters": filters, "prefer": prefer, "limit": limit, "as_of": as_of, "changed_after": changed_after,
              "changed_before": changed_before, "mode": mode}
    fingerprint = _fingerprint(evidence)
    synthesis_id = _sha(canonical_json({"type": synthesis_type, "query": query or "", "params": params,
                                        "evidence": fingerprint, "model": model.name, "provider": model.provider,
                                        "prompt_version": PROMPT_VERSION}))[:24]
    existing = get_synthesis(db, synthesis_id, missing_ok=True)
    if existing:
        return {"status": "existing", "synthesis": existing}
    user = f"Topic: {query or '(changes in the window)'}\n\nEvidence:\n" + "\n\n".join(
        f"[{e['ref']}] ({e['path']}, revision {e.get('revision_id')})\n{e['text']}" for e in evidence)
    try:
        raw = model.generate(SYSTEM_PROMPT + TYPES[synthesis_type], user)
    except Exception as exc:  # noqa: BLE001 - a model failure produces no artifact
        return {"status": "model_error", "reason": str(exc)}
    items, rejected = _parse_items(raw, synthesis_type, {e["ref"] for e in evidence})
    if not items:
        return {"status": "no_supported_statements", "rejected": rejected,
                "reason": "the model produced no statement backed by the evidence; nothing was stored"}
    body = _render(synthesis_type, query or "changes", items, evidence, model)
    now = db.now()
    record = {"synthesis_id": synthesis_id, "type": synthesis_type, "query": query or "", "params": params,
              "created_at": now, "actor_id": actor, "model": model.name, "provider": model.provider,
              "prompt_version": PROMPT_VERSION, "evidence": evidence, "evidence_fingerprint": fingerprint,
              "items": items, "body": body, "content_hash": _sha(body), "status": "current"}
    with db.transaction():
        db.conn.execute(
            "INSERT INTO syntheses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'current', NULL, NULL, ?)",
            (synthesis_id, synthesis_type, record["query"], canonical_json(params), now, actor, model.name,
             model.provider, PROMPT_VERSION, canonical_json(evidence), fingerprint, canonical_json(items), body,
             record["content_hash"], now))
        if synthesis_type == "CONTRADICTION":
            by_ref = {e["ref"]: e for e in evidence}
            for n, item in enumerate(items):
                status, basis = _contradiction_status(db, [by_ref[r] for r in item["evidence_a"]],
                                                      [by_ref[r] for r in item["evidence_b"]])
                db.conn.execute(
                    "INSERT INTO contradictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)",
                    (f"{synthesis_id}-{n}", synthesis_id, item["claim_a"], item["claim_b"],
                     canonical_json([_ev_ref(by_ref[r]) for r in item["evidence_a"]]),
                     canonical_json([_ev_ref(by_ref[r]) for r in item["evidence_b"]]),
                     canonical_json({"a": [(by_ref[r].get("valid_from"), by_ref[r].get("valid_to"))
                                           for r in item["evidence_a"]],
                                     "b": [(by_ref[r].get("valid_from"), by_ref[r].get("valid_to"))
                                           for r in item["evidence_b"]]}),
                     canonical_json({"a": sorted({by_ref[r]["path"] or "" for r in item["evidence_a"]}),
                                     "b": sorted({by_ref[r]["path"] or "" for r in item["evidence_b"]})}),
                     status, basis, now))
    return {"status": "created", "synthesis": get_synthesis(db, synthesis_id), "rejected": rejected}


def _ev_ref(e):
    return {k: e.get(k) for k in ("ref", "document_id", "revision_id", "point_id", "path", "content_hash")}


def _contradiction_status(db, side_a, side_b):
    def interval(items):
        vf = [e.get("valid_from") for e in items]
        vt = [e.get("valid_to") for e in items]
        if any(v is None for v in vf + vt):
            return None
        return min(vf), max(vt)
    ia, ib = interval(side_a), interval(side_b)
    if ia and ib and (ia[1] < ib[0] or ib[1] < ia[0]):
        return "time_dependent", "declared validity intervals do not overlap"
    ra = {e.get("revision_id") for e in side_a} - {None}
    rb = {e.get("revision_id") for e in side_b} - {None}
    if len(ra) == 1 and len(rb) == 1 and ra != rb:
        a, b = next(iter(ra)), next(iter(rb))
        if versions.is_ancestor(db, a, b):
            return "superseded", "claim A comes from an ancestor revision of claim B's revision"
        if versions.is_ancestor(db, b, a):
            return "superseded", "claim B comes from an ancestor revision of claim A's revision"
    return "unresolved", "no metadata establishes precedence; needs human review"


def get_synthesis(db, synthesis_id, missing_ok=False):
    s = row_dict(db.one("SELECT * FROM syntheses WHERE synthesis_id = ?", (synthesis_id,)), _JSON)
    if s is None and not missing_ok:
        raise NotFound(f"no synthesis {synthesis_id}")
    return s


def can_read_synthesis(db, actor, synthesis):
    if can(db, actor, "admin", None) or synthesis["actor_id"] == actor:
        return True
    for e in synthesis["evidence"]:
        doc = identity.get_document(db, e["document_id"], missing_ok=True) if e.get("document_id") else None
        if not can(db, actor, "read", doc):
            return False
    return True


def read_synthesis(db, actor, synthesis_id):
    s = get_synthesis(db, synthesis_id)
    if not can_read_synthesis(db, actor, s):
        raise AuthorizationError(f"{actor} may not read synthesis {synthesis_id} (it cites unreadable evidence)")
    s["contradictions"] = [row_dict(r, ("evidence_a", "evidence_b", "time_context", "source_context")) for r in
                           db.all("SELECT * FROM contradictions WHERE synthesis_id = ? ORDER BY contradiction_id",
                                  (synthesis_id,))]
    return s


def list_syntheses(db, actor, status=None):
    sql, params = "SELECT * FROM syntheses", ()
    if status:
        sql, params = sql + " WHERE status = ?", (status,)
    out = []
    for row in db.all(sql + " ORDER BY created_at", params):
        s = row_dict(row, _JSON)
        if can_read_synthesis(db, actor, s):
            out.append({k: s[k] for k in ("synthesis_id", "type", "query", "created_at", "actor_id", "model",
                                          "status", "status_reason", "superseded_by")})
    return out


def check_synthesis(db, store, synthesis_id, actor=None):
    """Re-verify evidence; marks the synthesis stale on proven drift."""
    s = get_synthesis(db, synthesis_id)
    problems, unverifiable = [], []
    historical = s["params"].get("mode") in ("historical", "changes")
    for e in s["evidence"]:
        if e.get("revision_id"):
            rev = versions.get_revision(db, e["revision_id"], missing_ok=True)
            if rev is None:
                problems.append(f"{e['ref']}: cited revision no longer exists")
                continue
            if not historical:
                doc = identity.get_document(db, e["document_id"], missing_ok=True)
                if doc is None or doc["status"] != "active":
                    problems.append(f"{e['ref']}: document was deleted")
                elif doc["current_revision_id"] != e["revision_id"]:
                    problems.append(f"{e['ref']}: document moved on from revision {e['revision_id']}")
        if e["kind"] == "chunk" and e.get("point_id") and not historical:
            if store is None:
                unverifiable.append(e["ref"])
                continue
            try:
                points = store.client.retrieve(e.get("collection") or store.collection, ids=[e["point_id"]],
                                               with_payload=True)
            except Exception:  # noqa: BLE001
                points = []
            if not points or _sha((points[0].payload or {}).get("text") or "") != e["content_hash"]:
                problems.append(f"{e['ref']}: evidence chunk changed or disappeared")
    if problems and s["status"] == "current":
        with db.transaction():
            db.conn.execute("UPDATE syntheses SET status = 'stale', status_reason = ?, updated_at = ? "
                            "WHERE synthesis_id = ? AND status = 'current'",
                            ("; ".join(problems), db.now(), synthesis_id))
    return {"synthesis_id": synthesis_id, "stale": bool(problems), "problems": problems,
            "unverifiable": unverifiable, "status": get_synthesis(db, synthesis_id)["status"]}


def resynthesize(db, store, actor, synthesis_id, model=None):
    old = read_synthesis(db, actor, synthesis_id)
    p = old["params"]
    result = synthesize(db, store, actor, old["query"], old["type"], model=model, filters=p.get("filters"),
                        prefer=p.get("prefer"), limit=p.get("limit") or 8, as_of=p.get("as_of"),
                        changed_after=p.get("changed_after"), changed_before=p.get("changed_before"))
    new = result.get("synthesis")
    if new and new["synthesis_id"] != synthesis_id and old["status"] != "superseded":
        with db.transaction():
            db.conn.execute("UPDATE syntheses SET status = 'superseded', superseded_by = ?, updated_at = ? "
                            "WHERE synthesis_id = ?", (new["synthesis_id"], db.now(), synthesis_id))
    return result


def resolve_contradiction(db, actor, contradiction_id, status, note=None):
    if status not in ("resolved", "superseded", "time_dependent", "not_conflicting", "unresolved"):
        raise CortexError("invalid contradiction status", code="invalid_status")
    row = row_dict(db.one("SELECT * FROM contradictions WHERE contradiction_id = ?", (contradiction_id,)),
                   ("evidence_a", "evidence_b"))
    if row is None:
        raise NotFound(f"no contradiction {contradiction_id}")
    user = resolve_actor(db, actor)
    if user["role"] == "viewer":
        raise AuthorizationError("viewers cannot resolve contradictions")
    for e in row["evidence_a"] + row["evidence_b"]:
        doc = identity.get_document(db, e["document_id"], missing_ok=True) if e.get("document_id") else None
        if not can(db, actor, "read", doc):
            raise AuthorizationError(f"{actor} cannot read all evidence of this contradiction")
    with db.transaction():
        db.conn.execute("UPDATE contradictions SET status = ?, resolved_by = ?, resolved_at = ?, note = ? "
                        "WHERE contradiction_id = ?", (status, actor, db.now(), note, contradiction_id))
    return row_dict(db.one("SELECT * FROM contradictions WHERE contradiction_id = ?", (contradiction_id,)))
