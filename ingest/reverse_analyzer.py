"""READ-ONLY reverse-sync analyzer: detects and classifies divergence
between a cortex-managed Obsidian note and the last generated version
knowledge-cortex knows about, from cached local-ingestion state.

This module performs NO writes of any kind:
  - never writes to Obsidian (only list_dir/read_note are called)
  - never writes to sync_state.json
  - never touches Qdrant (VectorStore is never imported/used here)
  - never modifies source files (existing text may be re-extracted
    in-memory purely to compute a comparison hash; the file itself is
    only ever opened for reading)

It is preparation for a future, still-unbuilt, human-approved reverse
sync -- this module only detects and reports; it never applies anything.

Authority model:
  - the LIVE VAULT NOTE is authority for what currently exists in Obsidian
  - sync_state.json ("local" namespace) is the record of what cortex last
    generated/knew, via its cached "generated_body" (see ingest/sync.py) --
    reused here directly rather than duplicating a second content cache
  - the SOURCE FILE ON DISK is authority for whether the underlying source
    currently exists and whether it has changed since state was recorded

Classification model (mutually exclusive PRIMARY classification, computed
by first-match precedence below; independent boolean FLAGS surface any
secondary condition that isn't already the primary, so nothing is hidden):

    INVALID_MANAGED_NOTE     Note claims cortex_managed: true but its
                             provenance frontmatter is malformed, missing
                             required fields, or internally inconsistent
                             (e.g. cortex_source_id doesn't match the
                             source path we have on file). Checked first:
                             if we can't trust the note's own claims, no
                             other classification is meaningful.
    UNMANAGED_AT_TARGET      A note exists at the expected deterministic
                             path but is not marked cortex_managed: true.
    MISSING                  No note exists at the expected path at all.
    ANALYSIS_INSUFFICIENT_STATE
                             The note is valid and cortex-managed, but our
                             own cached state has no "generated_body" for
                             this source (e.g. state predates that field,
                             or was manually edited/stripped) -- we cannot
                             honestly claim IN_SYNC or HUMAN_MODIFIED
                             without a reliable expected body to compare
                             against. This is a genuinely distinct state
                             from the 7 requested; it exists because
                             collapsing it into HUMAN_MODIFIED or IN_SYNC
                             would be a guess, not a detection.
    HUMAN_MODIFIED           Note is valid/cortex-managed and we have an
                             expected body, the live body hash differs
                             from it, AND differs from the note's own
                             recorded cortex_generated_sha256 (something
                             edited it after cortex wrote it). Flag
                             vault_generated_from_older_source says the
                             note was generated from a different source
                             version than the current one.
    SOURCE_MISSING           The source file recorded for this entry no
                             longer exists on disk. (Vault note itself may
                             still be IN_SYNC; this only reports source
                             existence.)
    SOURCE_CHANGED           The source file's current content hash
                             differs from the hash recorded when the
                             managed note was last generated -- either
                             versus state (not yet re-ingested), or the
                             note is unmodified since cortex wrote it
                             (body still hashes to its own
                             cortex_generated_sha256) but holds an older
                             generated body than the re-ingested state.
                             The latter is never HUMAN_MODIFIED. Read-only:
                             this ONLY reports the detection: no
                             reprocessing is triggered.
    IN_SYNC                  Live vault body matches the last
                             cortex-generated body, source is unchanged
                             and present.

Precedence (first match wins as PRIMARY; ALL applicable flags are still
reported so nothing is hidden behind the primary label):
    INVALID_MANAGED_NOTE > UNMANAGED_AT_TARGET > MISSING >
    ANALYSIS_INSUFFICIENT_STATE > HUMAN_MODIFIED > SOURCE_MISSING >
    SOURCE_CHANGED > IN_SYNC
Rationale: provenance/ownership problems are checked before content
comparison (an untrustworthy or foreign note makes any further comparison
meaningless); among content-comparison outcomes, a human edit in the vault
is surfaced ahead of a source-side change because it is the more urgent,
harder-to-recover-from divergence (the vault is where a human is actively
working) -- but source_changed/source_missing are still reported as flags
even when HUMAN_MODIFIED is the primary classification.
"""

import difflib
import hashlib
import os
import re

from ingest.vault_writer import parse_frontmatter, hash_managed_body, note_exists, source_id_for

CLASSIFICATIONS = (
    "IN_SYNC", "HUMAN_MODIFIED", "MISSING", "UNMANAGED_AT_TARGET",
    "SOURCE_CHANGED", "SOURCE_MISSING", "INVALID_MANAGED_NOTE",
    "ANALYSIS_INSUFFICIENT_STATE",
)

PROPOSED_ACTIONS = {
    "IN_SYNC": "none",
    "HUMAN_MODIFIED": "review_human_changes",
    "MISSING": "recreate_missing_managed_note",
    "UNMANAGED_AT_TARGET": "investigate_provenance",
    "SOURCE_CHANGED": "source_changed_reprocess_required",
    "SOURCE_MISSING": "source_missing_review_required",
    "INVALID_MANAGED_NOTE": "investigate_provenance",
    "ANALYSIS_INSUFFICIENT_STATE": "reprocess_required",
    "ERROR": "investigate_analyzer_error",
}

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")

DEFAULT_MAX_DIFF_LINES = 200


def _current_source_hash(source_file):
    """Best-effort re-extraction of the CURRENT source file's text, purely
    in memory, to compute a comparison hash using the exact same
    extract-then-sha256 contract ingest/sync.py's index_local_path uses.
    This never writes anything and never re-runs AI structuring (which is
    the expensive step this analyzer is scoped to avoid); extraction alone
    is deterministic and cheap by comparison.

    Returns (hash_or_None, error_or_None).
    """
    from ingest.detect import detect_file_type
    from ingest.extract import extract_text

    try:
        ftype = detect_file_type(source_file)
        if ftype == "unknown":
            return None, f"unsupported file type: {source_file}"
        text = extract_text(source_file, ftype)
        return hashlib.sha256(text.encode("utf-8")).hexdigest(), None
    except Exception as exc:  # noqa: BLE001 - one bad source must not abort the batch
        return None, str(exc)


def _build_diff(expected_body, live_body, max_diff_lines):
    diff_lines = list(difflib.unified_diff(
        expected_body.strip().splitlines(), live_body.strip().splitlines(),
        fromfile="cortex_generated", tofile="vault_current", lineterm="",
    ))
    truncated = len(diff_lines) > max_diff_lines
    return {
        "diff": diff_lines[:max_diff_lines],
        "diff_truncated": truncated,
        "diff_total_lines": len(diff_lines),
    }


async def _analyze_one(obsidian, source_file, entry, max_diff_lines):
    dest_path = entry.get("vault_write", {}).get("dest_path")
    result = {
        "source_path": source_file,
        "managed_note_path": dest_path,
        "recorded_source_sha256": entry.get("source_sha256"),
        "current_source_sha256": None,
        "expected_generated_sha256": None,
        "current_vault_sha256": None,
        "ai_structured": bool(entry.get("ai_structure_succeeded")),
        "structure_model": entry.get("structure_model"),
        "flags": {"source_changed": False, "source_missing": False},
    }

    source_exists = os.path.exists(source_file)
    if not source_exists:
        result["flags"]["source_missing"] = True
    else:
        current_hash, extraction_error = _current_source_hash(source_file)
        result["current_source_sha256"] = current_hash
        if extraction_error:
            result["source_extraction_error"] = extraction_error
        elif current_hash != result["recorded_source_sha256"]:
            result["flags"]["source_changed"] = True

    if not dest_path:
        # Should not happen (caller only reaches here when vault_write is
        # present), but never crash the batch over one malformed entry.
        result["classification"] = "INVALID_MANAGED_NOTE"
        result["reason"] = "state entry has no recorded managed_note_path (dest_path)"
        result["proposed_action"] = PROPOSED_ACTIONS["INVALID_MANAGED_NOTE"]
        return result

    exists = await note_exists(obsidian, dest_path)
    if not exists:
        result["classification"] = "MISSING"
        result["reason"] = "no note exists at the expected managed path"
        result["proposed_action"] = PROPOSED_ACTIONS["MISSING"]
        return result

    current = await obsidian.read_note(dest_path)
    current_fm, current_body = parse_frontmatter(current["content"])
    result["current_vault_sha256"] = hash_managed_body(current_body)
    # Full live body text, kept out of the human-readable/JSON-default
    # rendering (main.py's analyze_vault only prints the summary fields)
    # but available to callers that need the EXACT reviewed content for a
    # durable record -- notably ingest.proposals.create_proposals, which
    # must capture precisely what a human would be approving, not just
    # its hash (see that module's docstring for why).
    result["_live_vault_body"] = current_body

    if current_fm.get("cortex_managed") != "true":
        result["classification"] = "UNMANAGED_AT_TARGET"
        result["reason"] = "a note exists at the expected path but is not cortex_managed"
        result["proposed_action"] = PROPOSED_ACTIONS["UNMANAGED_AT_TARGET"]
        return result

    required_fields = ("cortex_source_id", "cortex_source_path", "cortex_generated_sha256")
    missing_fields = [f for f in required_fields if not current_fm.get(f)]
    expected_source_id = source_id_for(source_file)
    actual_source_id = current_fm.get("cortex_source_id")
    recorded_hash_in_note = current_fm.get("cortex_generated_sha256", "")
    hash_format_valid = bool(_SHA256_RE.match(recorded_hash_in_note))

    invalid_reasons = []
    if missing_fields:
        invalid_reasons.append(f"missing required frontmatter field(s): {', '.join(missing_fields)}")
    if not missing_fields and actual_source_id != expected_source_id:
        invalid_reasons.append(
            f"cortex_source_id mismatch (note has {actual_source_id!r}, "
            f"expected {expected_source_id!r} for this source path)"
        )
    if not missing_fields and not hash_format_valid:
        invalid_reasons.append(f"cortex_generated_sha256 is not a valid sha256 hex digest: {recorded_hash_in_note!r}")

    if invalid_reasons:
        result["classification"] = "INVALID_MANAGED_NOTE"
        result["reason"] = "; ".join(invalid_reasons)
        result["proposed_action"] = PROPOSED_ACTIONS["INVALID_MANAGED_NOTE"]
        return result

    if "generated_body" not in entry:
        result["classification"] = "ANALYSIS_INSUFFICIENT_STATE"
        result["reason"] = ("cached generated_body is missing from state for this source; cannot "
                             "reliably determine IN_SYNC vs HUMAN_MODIFIED without reprocessing")
        result["proposed_action"] = PROPOSED_ACTIONS["ANALYSIS_INSUFFICIENT_STATE"]
        return result

    expected_body = entry["generated_body"]
    expected_hash = hash_managed_body(expected_body)
    result["expected_generated_sha256"] = expected_hash

    # The note's own recorded cortex_generated_sha256 says whether anyone
    # touched it since cortex wrote it. If not, a body that differs from the
    # current generated body is an OLDER cortex output (the source changed
    # and was re-ingested without --write-vault), never a human edit, so it
    # must not become an apply-eligible HUMAN_MODIFIED proposal that would
    # write stale generated text back over the newer source.
    vault_untouched = result["current_vault_sha256"] == recorded_hash_in_note
    result["flags"]["vault_generated_from_older_source"] = bool(
        result["current_source_sha256"] and current_fm.get("cortex_source_sha256")
        and current_fm["cortex_source_sha256"] != result["current_source_sha256"])

    if result["current_vault_sha256"] != expected_hash and vault_untouched:
        if result["flags"]["source_missing"]:
            result["classification"] = "SOURCE_MISSING"
            result["reason"] = "recorded source file no longer exists on disk"
            result["proposed_action"] = PROPOSED_ACTIONS["SOURCE_MISSING"]
            return result
        result["flags"]["source_changed"] = True
        result["classification"] = "SOURCE_CHANGED"
        result["reason"] = ("managed note is unmodified since cortex wrote it but holds an older generated "
                            "body; the source changed since (run --write-vault to update the note)")
        result["proposed_action"] = PROPOSED_ACTIONS["SOURCE_CHANGED"]
        result.update(_build_diff(expected_body, current_body, max_diff_lines))
        return result

    if result["current_vault_sha256"] != expected_hash:
        result["classification"] = "HUMAN_MODIFIED"
        result["reason"] = "live vault note body differs from the last cortex-generated body"
        result["proposed_action"] = PROPOSED_ACTIONS["HUMAN_MODIFIED"]
        result.update(_build_diff(expected_body, current_body, max_diff_lines))
        return result

    if result["flags"]["source_missing"]:
        result["classification"] = "SOURCE_MISSING"
        result["reason"] = "recorded source file no longer exists on disk"
        result["proposed_action"] = PROPOSED_ACTIONS["SOURCE_MISSING"]
        return result

    if result["flags"]["source_changed"]:
        result["classification"] = "SOURCE_CHANGED"
        result["reason"] = "source file content has changed since the managed note was last generated"
        result["proposed_action"] = PROPOSED_ACTIONS["SOURCE_CHANGED"]
        return result

    result["classification"] = "IN_SYNC"
    result["reason"] = "vault note matches the last cortex-generated body; source unchanged"
    result["proposed_action"] = PROPOSED_ACTIONS["IN_SYNC"]
    return result


async def analyze_managed_notes(obsidian, local_state, source_filter=None,
                                 max_diff_lines=DEFAULT_MAX_DIFF_LINES):
    """Read-only classification pass over every source in local_state that
    has previously been written via --write-vault (identified by the
    presence of a "vault_write" record -- entries only indexed, never
    write-vault'd, are not "managed notes" and are skipped).

    obsidian: an ObsidianClient (or compatible fake); only list_dir and
    read_note are ever called -- no write_note/append_note/anything else.
    local_state: the "local" namespace dict from sync_cli.load_state(...).
    This function NEVER mutates local_state and never persists anything;
    the caller is responsible for not saving it back after analysis.
    source_filter: if given, only the entry for this exact source path
    (absolute path, matching state's keys) is analyzed.

    Returns a list of per-entry result dicts (see module docstring for the
    classification model). One entry's unexpected failure is caught and
    reported as classification "ERROR" without aborting the rest of the
    batch.
    """
    results = []
    for source_file, entry in local_state.items():
        if source_filter and source_file != source_filter:
            continue
        if not isinstance(entry, dict) or not entry.get("vault_write"):
            continue  # never write-vault'd for this source; not a "managed note"
        try:
            result = await _analyze_one(obsidian, source_file, entry, max_diff_lines)
        except Exception as exc:  # noqa: BLE001 - one bad entry must not abort the batch
            result = {
                "source_path": source_file,
                "managed_note_path": entry.get("vault_write", {}).get("dest_path"),
                "classification": "ERROR",
                "reason": f"analysis failed: {exc}",
                "proposed_action": PROPOSED_ACTIONS["ERROR"],
            }
        results.append(result)
    return results
