"""Durable, metadata-only review proposals. No apply layer exists.

Only proposal JSON files are written. Source files, live Obsidian notes,
Qdrant, and sync state are never written by this module.

Identity: sha256(canonical JSON of resolved state_path, exact source_path,
managed_note_path, classification, proposed_action, source_exists, and the
three fingerprints)[:16]. Timestamps, decisions, and rendered diffs are not
identity inputs. Identical facts reproduce the same ID without resetting a
decision. Existing unreadable records are never repaired or overwritten.

Approval first validates internal consistency: schema, filename/embedded/
recomputed identity, field types, SHA256 shapes, source presence, vault
presence, and both captured bodies against their respective fingerprints.
Invalid proposals are left byte-for-byte unchanged, without external reads.
Then approval re-reads all three authorities: source presence/extracted-text
hash; vault presence/managed-body hash; and the cached generated_body under
the EXACT source key in the ORIGINAL resolved state file. Missing/malformed
state or drift makes the proposal stale. An unrelated --state is refused,
not accepted merely because its content matches. Source absence is a frozen
presence fact, never replaced with a historical source hash.

Snapshots are exact text, but managed-body fingerprints deliberately reuse
vault_writer.hash_managed_body (outer whitespace stripped, frontmatter
excluded); source hashes reuse reverse_analyzer's extracted-text helper.
For HUMAN_MODIFIED the reviewed live body is a future candidate. For MISSING
the captured generated body is a future recreation candidate, including when
the analyzer returned no generated hash before its early MISSING return.
SOURCE_CHANGED and investigation/source-missing findings are diagnostic:
approving them does NOT authorize newly regenerated content. No proposal
here supplies a generic, executable mutation plan.

Approved/rejected decisions are terminal; stale cannot later be approved.
Rejection does not require live facts to match. Local JSON consistency is
not a signature/authentication system; coordinated edits can forge records.
There is no distributed lock, compare-and-swap, or atomic cross-authority
snapshot; a future apply layer must revalidate everything and define its own
format/ownership/authorization policy before making any writes.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

from ingest.vault_writer import hash_managed_body

SCHEMA_VERSION = 1

DEFAULT_PROPOSALS_DIR = Path(__file__).resolve().parent.parent / "state" / "proposals"
DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "sync_state.json"

# Classifications that create no proposal at all -- either there's nothing
# to review (IN_SYNC) or there isn't enough information to propose
# anything meaningful without guessing (ANALYSIS_INSUFFICIENT_STATE,
# ERROR).
NO_PROPOSAL_CLASSIFICATIONS = ("IN_SYNC", "ANALYSIS_INSUFFICIENT_STATE", "ERROR")


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_proposal_id(source_path, managed_note_path, classification, proposed_action,
                         source_sha256, expected_generated_sha256, live_vault_sha256, state_path=None, source_exists=None):
    """Deterministic id from MATERIAL FACTS only -- see module docstring's
    identity contract. Never include timestamps, status, or diff text."""
    material = {
        "source_exists": source_sha256 is not None if source_exists is None else source_exists,
        "state_path": state_path,
        "source_path": source_path,
        "managed_note_path": managed_note_path,
        "classification": classification,
        "proposed_action": proposed_action,
        "source_sha256": source_sha256,
        "expected_generated_sha256": expected_generated_sha256,
        "live_vault_sha256": live_vault_sha256,
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()[:16]


def _proposal_path(proposals_dir, proposal_id):
    return Path(proposals_dir) / f"{proposal_id}.json"


def _atomic_write_json(path, data):
    """Write JSON atomically (temp file + os.replace) so a crash mid-write
    never leaves a truncated/corrupt proposal file, and a concurrent
    reader never observes a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_proposal(proposals_dir, proposal_id):
    """Load one proposal by id. Returns (proposal_dict_or_None, error_or_None).
    Never raises for a missing or malformed file -- callers get a clear
    error string instead, matching this project's fail-safely convention."""
    if not isinstance(proposal_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", proposal_id):
        return None, "invalid proposal id (path components are forbidden)"
    path = _proposal_path(proposals_dir, proposal_id)
    if path.is_symlink():
        return None, "proposal symlinks are not supported"
    if not path.exists():
        return None, f"no proposal found with id {proposal_id!r} at {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return None, f"proposal file {path} is unreadable or not valid JSON: {exc}"
    if not isinstance(data, dict) or "schema_version" not in data:
        return None, f"proposal file {path} is missing required 'schema_version' field"
    if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
        return None, (f"proposal file {path} has unsupported schema_version "
                       f"{data['schema_version']!r} (this build supports {SCHEMA_VERSION})")
    required = ("proposal_id", "status", "source_path", "fingerprints")
    missing = [f for f in required if f not in data]
    if missing:
        return None, f"proposal file {path} is missing required field(s): {', '.join(missing)}"
    if data["proposal_id"] != proposal_id:
        return None, "proposal_id does not match its own material fields or filename id"
    return data, None


def list_proposals(proposals_dir):
    """Return [(proposal_dict, error_or_None), ...] for every *.json file
    in proposals_dir, sorted by filename. A malformed file is reported
    with its error rather than raising or silently skipping -- so a
    corrupted proposal is visible, not invisible."""
    proposals_dir = Path(proposals_dir)
    if not proposals_dir.exists():
        return []
    results = []
    for path in sorted(proposals_dir.glob("*.json")):
        proposal, error = load_proposal(proposals_dir, path.stem)
        results.append((proposal, error))
    return results


def create_proposals(analyzer_results, local_state=None, proposals_dir=DEFAULT_PROPOSALS_DIR, state_path=None):
    """Turn reverse_analyzer results into durable proposal files.

    analyzer_results: the list returned by
    ingest.reverse_analyzer.analyze_managed_notes(...) (read-only; this
    function does not call the analyzer itself, so callers control
    exactly what was analyzed -- e.g. main.py's --propose-vault-changes
    wires the two together).

    local_state: the same "local" namespace dict passed to the analyzer
    (read-only here too). Used only to capture "expected_generated_body"
    -- the exact cached Cortex-generated body text for this source at
    proposal-creation time -- so a future apply layer has an exact
    candidate payload to work with for classifications where the managed
    note doesn't itself hold the needed content (e.g. MISSING). Optional;
    if omitted, expected_generated_body is simply not captured.

    Only creates a proposal for classifications where a human decision is
    meaningful (everything except NO_PROPOSAL_CLASSIFICATIONS). Idempotent:
    if an identical proposal (same deterministic id) already exists on
    disk, it is NOT rewritten or duplicated -- its existing status is
    reported as-is via "already_existed": True in the returned entry
    (this preserves an already-approved/rejected decision rather than
    resetting it back to "pending" just because the same divergence was
    re-analyzed).

    Returns a list of {"proposal_id", "classification", "status",
    "already_existed", "created": bool} for every analyzer result that
    was eligible for a proposal (IN_SYNC/ANALYSIS_INSUFFICIENT_STATE/ERROR
    results are skipped and NOT included in the return value at all).
    """
    local_state = local_state or {}
    state_path = str(Path(state_path or DEFAULT_STATE_PATH).resolve())
    created = []
    for result in analyzer_results:
        classification = result.get("classification")
        if classification in NO_PROPOSAL_CLASSIFICATIONS:
            continue

        state_entry = local_state.get(result.get("source_path"))
        expected_generated_body = state_entry.get("generated_body") if isinstance(state_entry, dict) else None
        expected_hash = result.get("expected_generated_sha256")
        if isinstance(expected_generated_body, str) and expected_hash is None:
            expected_hash = hash_managed_body(expected_generated_body)
        if expected_generated_body is not None and (not isinstance(expected_generated_body, str)
                or hash_managed_body(expected_generated_body) != expected_hash):
            raise ValueError("cached generated body does not match analyzer's generated fingerprint")
        fingerprints = {
            "source_sha256": result.get("current_source_sha256"),
            "source_exists": not (result.get("flags") or {}).get("source_missing", False),
            "expected_generated_sha256": expected_hash,
            "live_vault_sha256": result.get("current_vault_sha256"),
        }
        proposal_id = compute_proposal_id(
            result.get("source_path"), result.get("managed_note_path"),
            classification, result.get("proposed_action"),
            fingerprints["source_sha256"], fingerprints["expected_generated_sha256"],
            fingerprints["live_vault_sha256"], state_path=state_path,
            source_exists=fingerprints["source_exists"],
        )

        existing, error = load_proposal(proposals_dir, proposal_id)
        if existing is None and _proposal_path(proposals_dir, proposal_id).exists():
            raise ValueError(f"refusing to repair existing proposal: {error}")
        if existing is not None:
            created.append({"proposal_id": proposal_id, "classification": classification,
                             "status": existing["status"], "already_existed": True, "created": False})
            continue

        proposal = {
            "state_path": state_path,
            "schema_version": SCHEMA_VERSION,
            "proposal_id": proposal_id,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "classification": classification,
            "proposed_action": result.get("proposed_action"),
            "source_path": result.get("source_path"),
            "managed_note_path": result.get("managed_note_path"),
            "fingerprints": fingerprints,
            "ai_structured": bool(result.get("ai_structured")),
            "structure_model": result.get("structure_model"),
            "diff": result.get("diff", []),
            "diff_truncated": bool(result.get("diff_truncated")),
            "diff_total_lines": result.get("diff_total_lines", 0),
            "reason": result.get("reason"),
            # The EXACT live vault body reviewed at proposal time. A future
            # apply layer must be able to act on precisely what was
            # reviewed, not something regenerated later that might differ
            # (e.g. if the analyzer's diff/hash logic changes, or the
            # cached generated_body in sync_state.json is later
            # overwritten by a subsequent run) -- so the full text is
            # captured here, not just its hash. Absent when the analyzer
            # never read a live body (e.g. MISSING).
            "reviewed_live_vault_body": result.get("_live_vault_body"),
            # The EXACT cached Cortex-generated body at proposal-creation
            # time (from sync_state.json's "generated_body" for this
            # source, if local_state was provided). This is what a future
            # apply layer would need to (re)write for classifications
            # like MISSING, where the live vault note doesn't hold the
            # needed content at all -- so it never has to regenerate or
            # guess it later.
            "expected_generated_body": expected_generated_body,
            "decision": {"status": None, "decided_at": None, "note": None},
        }
        _atomic_write_json(_proposal_path(proposals_dir, proposal_id), proposal)
        created.append({"proposal_id": proposal_id, "classification": classification,
                         "status": "pending", "already_existed": False, "created": True})
    return created


async def _current_fingerprints(source_path, managed_note_path, obsidian):
    """Re-read the CURRENT live source/vault state, purely for staleness
    comparison -- performs no writes anywhere. Mirrors
    ingest.reverse_analyzer's own hashing exactly (reuses the same
    helpers) rather than reimplementing it."""
    from ingest.reverse_analyzer import _current_source_hash
    from ingest.vault_writer import parse_frontmatter, hash_managed_body, note_exists

    current = {"source_exists": False, "source_sha256": None,
               "note_exists": False, "live_vault_sha256": None}

    current["source_exists"] = os.path.exists(source_path)
    if current["source_exists"]:
        current["source_sha256"], _ = _current_source_hash(source_path)

    if managed_note_path:
        current["note_exists"] = await note_exists(obsidian, managed_note_path)
        if current["note_exists"]:
            doc = await obsidian.read_note(managed_note_path)
            _, body = parse_frontmatter(doc["content"])
            current["live_vault_sha256"] = hash_managed_body(body)

    return current


def _validate_proposal_internal_consistency(proposal):
    """Validate frozen facts before reading any external authority; never repair."""
    problems = []
    fp = proposal.get("fingerprints")
    if not isinstance(fp, dict):
        return ["fingerprints has an invalid type (expected object)"]
    for key in ("source_path", "managed_note_path", "state_path", "classification", "proposed_action"):
        value = proposal.get(key)
        if not isinstance(value, str) or not value or "\x00" in value:
            problems.append(f"{key} must be a nonempty string")
    if type(fp.get("source_exists")) is not bool:
        problems.append("source_exists must be a boolean presence fingerprint")
    elif fp["source_exists"] != (fp.get("source_sha256") is not None):
        problems.append("source presence and source_sha256 are inconsistent")
    for key in ("source_sha256", "expected_generated_sha256", "live_vault_sha256"):
        value = fp.get(key)
        if key not in fp:
            problems.append(f"missing fingerprints.{key}")
        elif value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
            problems.append(f"fingerprints.{key} has an invalid type or SHA256 shape")
    if fp.get("expected_generated_sha256") is None:
        problems.append("expected_generated_sha256 is required for approval")
    if (proposal.get("classification") == "MISSING") != (fp.get("live_vault_sha256") is None):
        problems.append("classification and live vault presence are inconsistent")
    recomputed_id = compute_proposal_id(
        proposal.get("source_path"), proposal.get("managed_note_path"),
        proposal.get("classification"), proposal.get("proposed_action"),
        fp.get("source_sha256"), fp.get("expected_generated_sha256"), fp.get("live_vault_sha256"),
        state_path=proposal.get("state_path"), source_exists=fp.get("source_exists"))
    if recomputed_id != proposal.get("proposal_id"):
        problems.append("proposal_id does not match its own material fields")
    for body_key, hash_key in (("reviewed_live_vault_body", "live_vault_sha256"),
                               ("expected_generated_body", "expected_generated_sha256")):
        body = proposal.get(body_key)
        if body is not None:
            if not isinstance(body, str) or hash_managed_body(body) != fp.get(hash_key):
                problems.append(f"{body_key} does not hash to fingerprints.{hash_key}")
        elif fp.get(hash_key) is not None:
            problems.append(f"{body_key} is required when its fingerprint exists")
    decision = proposal.get("decision")
    if not isinstance(decision, dict) or (proposal.get("status") == "pending" and
            (decision.get("status") is not None or decision.get("decided_at") is not None)):
        problems.append("decision metadata is inconsistent with status")
    if proposal.get("status") not in ("pending", "approved", "rejected", "stale"):
        problems.append("status is not a recognized value")
    return problems


async def _current_generated_baseline(source_path, local_state_loader):
    """Look up the CURRENT sync_state.json "local" namespace entry for
    the EXACT source_path recorded on the proposal, and return its
    cached generated_body's hash using the identical hash_managed_body
    contract vault_writer/reverse_analyzer already use. Read-only.

    Returns (hash_or_None, error_or_None). An error means the baseline
    cannot be verified at all (missing/malformed entry, wrong identity,
    no cached body) -- the caller must treat that as stale/unverifiable,
    never as "assume it's fine".

    local_state_loader: a zero-arg callable returning the current "local"
    namespace dict (injectable for tests; defaults to reading the real
    sync state file via a strict JSON reader in the caller).
    """
    try:
        state = local_state_loader()
    except (OSError, ValueError, TypeError) as exc:
        return None, f"cannot read sync state: {exc}"
    if not isinstance(state, dict):
        return None, "local state namespace is malformed (not an object)"
    entry = state.get(source_path)
    if entry is None:
        return None, f"no sync_state.json entry exists for source path {source_path!r}"
    if not isinstance(entry, dict):
        return None, f"sync_state.json entry for {source_path!r} is malformed (not an object)"
    if "generated_body" not in entry:
        return None, f"sync_state.json entry for {source_path!r} has no cached generated_body"
    body = entry["generated_body"]
    if not isinstance(body, str):
        return None, f"sync_state.json entry for {source_path!r} has a non-string generated_body"
    return hash_managed_body(body), None


async def approve_proposal(proposals_dir, proposal_id, obsidian, note=None, state_path=None):
    """Approve a pending proposal, IF AND ONLY IF:
      1. the proposal passes its own internal-consistency checks
         (proposal_id matches its material fields, reviewed_live_vault_body
         hashes to live_vault_sha256, fingerprint types are sane), AND
      2. ALL THREE fingerprinted authorities still match what was recorded
         at proposal-creation time: the source file, the live Obsidian
         note, AND Cortex's own last-generated baseline (verified against
         sync_state.json's cached generated_body for the EXACT source
         path recorded on the proposal -- never matched by hash alone).

    Approval means "approve this exact frozen three-way snapshot", not
    merely "source and vault happen to still match." Never modifies
    source, Obsidian, Qdrant, or sync_state.json -- only the proposal's
    own JSON file (status + decision metadata), and only for the
    staleness case, never for a failed internal-consistency check (see
    below).

    state_path: path to sync_state.json (default: the project's own
    sync_state.json next to main.py, matching every other entry point's
    default). Must match the resolved state_path frozen at creation. The
    "local" namespace is loaded strictly and read-only, never written back.

    Returns {"ok": bool, "status": "<new proposal status>", "reason": str|None}.
    """
    proposal, error = load_proposal(proposals_dir, proposal_id)
    if proposal is None:
        return {"ok": False, "status": None, "reason": error}

    if proposal["status"] in ("approved", "rejected"):
        # Terminal decisions are immutable -- see module docstring.
        return {"ok": False, "status": proposal["status"],
                "reason": f"proposal is already {proposal['status']}; decisions are terminal, "
                          f"regenerate a new proposal if the situation has changed"}

    if proposal["status"] == "stale":
        return {"ok": False, "status": "stale",
                "reason": "proposal was already marked stale by a previous approval attempt; "
                          "generate a fresh proposal"}

    # Internal consistency FIRST, before touching any external system or
    # even relabeling the proposal: a proposal that fails this check
    # cannot be trusted enough to safely reclassify as "stale" either --
    # it is left exactly as found (still "pending") for a human to
    # inspect directly, since we cannot be confident what part of it is
    # trustworthy.
    consistency_problems = _validate_proposal_internal_consistency(proposal)
    if consistency_problems:
        return {"ok": False, "status": proposal["status"],
                "reason": "proposal failed internal consistency validation, refusing to approve: "
                          + "; ".join(consistency_problems)}

    state_path = Path(state_path or DEFAULT_STATE_PATH).resolve()
    if str(state_path) != proposal.get("state_path"):
        return {"ok": False, "status": proposal["status"],
                "reason": "approval state path differs from the proposal's original state_path"}
    current = await _current_fingerprints(proposal["source_path"], proposal["managed_note_path"], obsidian)
    fp = proposal["fingerprints"]
    drift_reasons = []

    if current["source_exists"] != fp["source_exists"]:
        drift_reasons.append("source presence changed")
    if current["note_exists"] != (fp["live_vault_sha256"] is not None):
        drift_reasons.append("managed note presence changed")

    if fp.get("source_sha256") is not None:
        if not current["source_exists"]:
            drift_reasons.append("source file no longer exists")
        elif current["source_sha256"] != fp["source_sha256"]:
            drift_reasons.append(
                f"source_sha256 changed (was {fp['source_sha256']!r}, now {current['source_sha256']!r})")

    if proposal.get("managed_note_path"):
        if not current["note_exists"]:
            if fp.get("live_vault_sha256") is not None:
                drift_reasons.append("managed note no longer exists in the vault")
        elif fp.get("live_vault_sha256") is not None and current["live_vault_sha256"] != fp["live_vault_sha256"]:
            drift_reasons.append(
                f"live vault body changed (was {fp['live_vault_sha256']!r}, now {current['live_vault_sha256']!r})")

    # Third authority: Cortex's own generated baseline. Checked against
    # the CURRENT sync_state.json local-namespace entry for the exact
    # source_path recorded on the proposal -- identity is verified by
    # path, never by "some entry happens to have a matching hash".
    if fp.get("expected_generated_sha256") is not None:
        def _load_local_state():
            data = json.loads(Path(state_path).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("local"), dict):
                raise ValueError("missing or malformed local state namespace")
            return data["local"]

        current_baseline_hash, baseline_error = await _current_generated_baseline(
            proposal["source_path"], _load_local_state)
        if baseline_error:
            drift_reasons.append(f"Cortex generated baseline is unverifiable: {baseline_error}")
        elif current_baseline_hash != fp["expected_generated_sha256"]:
            drift_reasons.append(
                f"Cortex generated baseline changed (was {fp['expected_generated_sha256']!r}, "
                f"now {current_baseline_hash!r})")

    if drift_reasons:
        proposal["status"] = "stale"
        proposal["decision"] = {
            "status": None, "decided_at": None,
            "note": "auto-marked stale on approval attempt: " + "; ".join(drift_reasons),
        }
        _atomic_write_json(_proposal_path(proposals_dir, proposal_id), proposal)
        return {"ok": False, "status": "stale", "reason": "; ".join(drift_reasons)}

    proposal["status"] = "approved"
    proposal["decision"] = {"status": "approved", "decided_at": datetime.now().isoformat(), "note": note}
    _atomic_write_json(_proposal_path(proposals_dir, proposal_id), proposal)
    return {"ok": True, "status": "approved", "reason": None}


def reject_proposal(proposals_dir, proposal_id, note=None):
    """Reject a proposal. Unlike approval, rejection does NOT require live
    fingerprints to still match -- a human may reject an outdated proposal
    freely. Terminal decisions ("approved"/"rejected") are still immutable:
    an already-decided proposal cannot be silently flipped.

    Returns {"ok": bool, "status": "<new proposal status>", "reason": str|None}.
    """
    proposal, error = load_proposal(proposals_dir, proposal_id)
    if proposal is None:
        return {"ok": False, "status": None, "reason": error}

    if proposal["status"] in ("approved", "rejected"):
        return {"ok": False, "status": proposal["status"],
                "reason": f"proposal is already {proposal['status']}; decisions are terminal"}

    proposal["status"] = "rejected"
    proposal["decision"] = {"status": "rejected", "decided_at": datetime.now().isoformat(), "note": note}
    _atomic_write_json(_proposal_path(proposals_dir, proposal_id), proposal)
    return {"ok": True, "status": "rejected", "reason": None}
