"""Apply layer: write the exact reviewed vault body of an APPROVED
HUMAN_MODIFIED proposal back into its original .md/.txt source file.

This is the only code in the project that writes to a source file, and it
only ever runs when explicitly invoked (main.py --apply-proposal). It is NOT
bidirectional sync: no watcher, no merge, no rename/delete propagation, no
other classification is applied.

Eligibility (everything else is refused with zero writes):
  - status "approved" with an approved decision record
  - classification HUMAN_MODIFIED + proposed_action review_human_changes
  - not AI-structured (deliberate safety constraint, see below), source
    present at proposal time
  - source suffix .md or .txt, a regular non-symlink writable file, strict
    UTF-8
  - the proposal passes proposals._validate_proposal_internal_consistency
    (identity recomputes, reviewed_live_vault_body hashes to
    live_vault_sha256)

Frozen snapshot: approval authorizes one exact three-way snapshot. Right
before writing, proposals.detect_drift re-reads the source, the live
managed note, and the generated baseline in the ORIGINAL state file. Any
drift refuses the write and marks the proposal stale (dry-run: reported
only). The candidate content is always reviewed_live_vault_body from the
proposal -- never re-read from the vault, regenerated, or rebuilt.

Candidate construction:
  - .txt: the reviewed body, verbatim.
  - .md: the source's own leading frontmatter block (same regex as the
    rest of the project), preserved verbatim, followed by the reviewed body.
    Without source frontmatter, the reviewed body verbatim.
  The reviewed body never contains the Cortex ownership frontmatter
  (parse_frontmatter strips it at analysis time); a candidate whose body
  begins with a block carrying cortex_* keys is refused anyway. A UTF-8 BOM
  and consistent CRLF line endings in the source are carried over. The body
  must still hash (hash_managed_body) to live_vault_sha256 after line-ending
  normalization, and must not be empty. Round-trip guard: running the
  candidate bytes back through ingestion's own body contract
  (ingest.markdown.document_body) must reproduce a body with that same
  hash, otherwise the write is refused (not_round_trip_stable). This is what
  lets the next normal ingestion converge: its generated_body equals the
  vault body, so the analyzer reports IN_SYNC and --write-vault only
  re-baselines the note.

AI-structured proposals are ineligible by design, not by omission: their
generated body is model output, so source -> generated_body is neither
deterministic nor invertible, applying would replace source text with
model text, and the round trip could never converge.

Write sequence (real apply):
  1. backup: exact pre-apply bytes to <backups_dir>/<proposal_id>/
     <name>.pre-apply (then .1, .2, ... -- O_EXCL, never overwritten),
     fsynced and read back
  2. intent: "apply_intent" (pre/candidate hashes, backup path) is recorded
     atomically in the proposal file
  3. atomic replace: temp file in the source's directory, fsync, original
     mode (and owner, where permitted), a final compare of the source bytes
     against what was validated, os.replace, parent directory fsync
  4. verify: re-read; bytes must equal the candidate, the extracted-text
     hash must equal the expected post hash, the preserved frontmatter must
     be intact, and the body must hash to live_vault_sha256
  5. record: status "applied" plus an "apply" record; apply_intent removed
  Fingerprints and the reviewed body are never modified.

Recovery contract (source replace and proposal JSON cannot be atomic
together): if step 5 fails, the proposal stays "approved" with its
apply_intent. The next --apply-proposal then compares the source bytes:
  - equal to the intended candidate: the write already happened; the
    application is recorded (recovered: true) without writing the source
    again. Vault/baseline drift found at that point is recorded as a
    warning, since the source write cannot be undone automatically.
  - equal to the pre-apply bytes: the replace never happened; the normal
    fully-revalidated path runs again.
  - anything else: refused (manual_review_required), backup path reported.

Not touched by apply: the vault (no write calls), Qdrant (VectorStore is
never imported), and sync_state.json (read only, for the baseline check).
The next normal ingestion sees the changed source and reprocesses it.

There is no lock: the window between the final source-bytes compare and
os.replace, and vault edits after the drift check, are not covered.
"""

import difflib
import hashlib
import io
import os
import re
import stat
import tempfile
from datetime import datetime
from pathlib import Path

from ingest.markdown import FRONTMATTER_RE, document_body
from ingest.proposals import (
    DEFAULT_STATE_PATH, _atomic_write_json, _proposal_path,
    _validate_proposal_internal_consistency, detect_drift, load_proposal,
)
from ingest.vault_writer import hash_managed_body, parse_frontmatter

SUPPORTED_SOURCE_SUFFIXES = (".md", ".txt")
APPLICABLE = {("HUMAN_MODIFIED", "review_human_changes")}
BOM = b"\xef\xbb\xbf"
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class ApplyRefused(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status
        self.reason = reason


class _SourceChangedDuringApply(Exception):
    pass


def default_backups_dir(proposals_dir):
    return Path(proposals_dir).parent / "apply_backups"


def _bytes_sha256(data):
    return hashlib.sha256(data).hexdigest()


def _extracted_text(raw):
    """The text ingest.extract.extract_text returns for a .md/.txt file with
    these bytes (utf-8, errors ignored, universal newlines)."""
    return io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="ignore").read()


def extracted_sha256(raw):
    """The source_sha256 contract, computed from bytes already in hand so
    the hash and the backed-up bytes are the same read."""
    return hashlib.sha256(_extracted_text(raw).encode("utf-8")).hexdigest()


def _normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _uses_crlf(text):
    if "\r\n" not in text:
        return False
    rest = text.replace("\r\n", "")
    return "\n" not in rest and "\r" not in rest


def build_candidate(suffix, raw_before, reviewed_body, live_vault_sha256):
    """Return (candidate_bytes, info). Raises ApplyRefused. Pure: no I/O."""
    try:
        decoded = raw_before.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ApplyRefused("unsupported_source_encoding",
                           f"source is not valid UTF-8 ({exc}); refusing to rewrite it") from exc
    has_bom = raw_before.startswith(BOM)
    if has_bom:
        decoded = decoded[1:]
    crlf = _uses_crlf(decoded)
    source_text = _normalize_newlines(decoded)

    body = _normalize_newlines(reviewed_body)
    if not body.strip():
        raise ApplyRefused("empty_candidate", "reviewed body is empty; refusing to empty the source")
    if hash_managed_body(body) != live_vault_sha256:
        raise ApplyRefused("candidate_hash_mismatch",
                           "reviewed body does not hash to the approved live_vault_sha256")
    body_fm, _ = parse_frontmatter(body)
    leaked = sorted(k for k in body_fm if k.startswith("cortex_"))
    if leaked:
        raise ApplyRefused("cortex_metadata_in_candidate",
                           f"candidate body begins with Cortex ownership frontmatter ({', '.join(leaked)})")

    prefix = ""
    if suffix == ".md":
        match = FRONTMATTER_RE.match(source_text)
        if match:
            prefix = match.group(0)
    candidate_text = prefix + body
    if crlf:
        candidate_text = candidate_text.replace("\n", "\r\n")
    candidate = (BOM if has_bom else b"") + candidate_text.encode("utf-8")
    regenerated = document_body(_extracted_text(candidate), suffix.lstrip("."))
    if hash_managed_body(regenerated) != live_vault_sha256:
        raise ApplyRefused("not_round_trip_stable",
                           "re-ingesting the candidate source would not reproduce the reviewed body "
                           "(e.g. the body starts with a frontmatter-like block that would be read as "
                           "source frontmatter); refusing")
    info = {
        "frontmatter_preserved": bool(prefix),
        "preserved_frontmatter": prefix,
        "crlf": crlf,
        "bom": has_bom,
        "candidate_bytes_sha256": _bytes_sha256(candidate),
        "expected_post_apply_source_sha256": extracted_sha256(candidate),
    }
    return candidate, info


def verify_written(raw_after, candidate, info, live_vault_sha256):
    """Post-write checks; returns a list of problems (empty = verified)."""
    problems = []
    if raw_after != candidate:
        problems.append("source bytes differ from the intended candidate")
    if extracted_sha256(raw_after) != info["expected_post_apply_source_sha256"]:
        problems.append("source extracted-text hash differs from the expected post-apply hash")
    text = raw_after.decode("utf-8", errors="replace")
    text = _normalize_newlines(text[1:] if text.startswith("﻿") else text)
    prefix = info["preserved_frontmatter"]
    if not text.startswith(prefix):
        problems.append("preserved source frontmatter is not intact")
    elif hash_managed_body(text[len(prefix):]) != live_vault_sha256:
        problems.append("source body does not hash to the approved live_vault_sha256")
    return problems


def _fsync_dir(directory):
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _backup_candidates(backups_dir, proposal_id, source_path):
    base = Path(backups_dir) / proposal_id / f"{Path(source_path).name}.pre-apply"
    yield base
    for n in range(1, 1000):
        yield base.with_name(f"{base.name}.{n}")


def next_backup_path(backups_dir, proposal_id, source_path):
    for path in _backup_candidates(backups_dir, proposal_id, source_path):
        if not os.path.lexists(path):
            return path
    return None


def create_backup(backups_dir, proposal_id, source_path, raw_before):
    """Write raw_before to the first free backup path (O_EXCL: an existing
    backup is never overwritten), fsync, and read it back."""
    directory = Path(backups_dir) / proposal_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in _backup_candidates(backups_dir, proposal_id, source_path):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw_before)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        if path.read_bytes() != raw_before:
            raise OSError(f"backup {path} does not read back as the exact pre-apply bytes")
        _fsync_dir(directory)
        return path
    raise OSError(f"no free backup path under {directory}")


def atomic_replace(target, data, expected_current):
    """Replace target with data via a same-directory temp file + os.replace.
    Before any failure the original is untouched and the temp file removed.
    Raises _SourceChangedDuringApply if target no longer holds
    expected_current just before the swap."""
    target = Path(target)
    st = os.stat(target)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".cortex-apply.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, stat.S_IMODE(st.st_mode))
        if (st.st_uid, st.st_gid) != (os.getuid(), os.getgid()):
            try:
                os.chown(tmp, st.st_uid, st.st_gid)
            except PermissionError:
                pass
        if target.read_bytes() != expected_current:
            raise _SourceChangedDuringApply()
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    _fsync_dir(target.parent)


def _result(ok, status, reason=None, **details):
    return {"ok": ok, "status": status, "reason": reason, **details}


def _diff(before, after, max_lines):
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                      fromfile="source_current", tofile="source_after_apply", lineterm=""))
    return lines[:max_lines], len(lines) > max_lines, len(lines)


def _check_eligibility(proposal, state_path):
    """Static checks that need no external reads; raises ApplyRefused."""
    problems = _validate_proposal_internal_consistency(proposal)
    decision = proposal.get("decision")
    if not isinstance(decision, dict) or decision.get("status") != "approved" \
            or not isinstance(decision.get("decided_at"), str):
        problems.append("approved proposal has no approved decision record")
    if problems:
        raise ApplyRefused("invalid_proposal", "proposal failed internal consistency validation: "
                           + "; ".join(problems))
    pair = (proposal["classification"], proposal["proposed_action"])
    if pair not in APPLICABLE:
        raise ApplyRefused("ineligible_classification",
                           f"only HUMAN_MODIFIED/review_human_changes proposals can be applied "
                           f"(got {pair[0]}/{pair[1]})")
    if proposal.get("ai_structured") is not False:
        raise ApplyRefused("ineligible_classification",
                           "AI-structured proposals are not applied: the vault body is an LLM "
                           "rewrite of the source, not an edit of its text")
    if not isinstance(proposal.get("reviewed_live_vault_body"), str):
        raise ApplyRefused("invalid_proposal", "reviewed_live_vault_body is missing")
    if not proposal["fingerprints"]["source_exists"]:
        raise ApplyRefused("source_missing", "source did not exist when the proposal was created")
    suffix = Path(proposal["source_path"]).suffix.lower()
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        raise ApplyRefused("unsupported_source_type",
                           f"reverse writes are only supported for {', '.join(SUPPORTED_SOURCE_SUFFIXES)} "
                           f"sources (got {suffix or 'no extension'!r})")
    if str(Path(state_path).resolve()) != proposal["state_path"]:
        raise ApplyRefused("state_path_mismatch",
                           "state path differs from the proposal's original state_path")
    return suffix


def _read_source(source_path):
    """Read the source's bytes; raises ApplyRefused for anything that is not
    an existing, writable, regular, non-symlink file."""
    path = Path(source_path)
    if path.is_symlink():
        raise ApplyRefused("unsupported_source_file", "source is a symlink; refusing to replace it")
    if not path.exists():
        raise ApplyRefused("stale", "source file no longer exists")
    if not path.is_file():
        raise ApplyRefused("unsupported_source_file", "source is not a regular file")
    if not os.access(path, os.W_OK):
        raise ApplyRefused("source_not_writable", "source file is not writable; refusing to replace it")
    return path.read_bytes()


def _apply_record(proposal, raw_before, info, backup_path, recovered, warnings=()):
    return {
        "applied_at": datetime.now().isoformat(),
        "source_path": proposal["source_path"],
        "pre_apply_source_sha256": extracted_sha256(raw_before),
        "pre_apply_source_bytes_sha256": _bytes_sha256(raw_before),
        "post_apply_source_sha256": info["expected_post_apply_source_sha256"],
        "post_apply_source_bytes_sha256": info["candidate_bytes_sha256"],
        "backup_path": str(backup_path) if backup_path else None,
        "frontmatter_preserved": info["frontmatter_preserved"],
        "recovered": recovered,
        "warnings": list(warnings),
    }


async def apply_proposal(proposals_dir, proposal_id, obsidian, state_path=None, dry_run=False,
                         backups_dir=None, max_diff_lines=200, actor=None):
    """Apply one approved proposal (see module docstring). Returns a dict
    {"ok", "status", "reason", ...details}. Statuses: applied, would_apply
    (dry-run), recovered, would_recover (dry-run), already_applied (ok);
    not_approved, rejected, stale, invalid_proposal, ineligible_classification,
    source_missing, unsupported_source_type, unsupported_source_file,
    unsupported_source_encoding, source_not_writable, state_path_mismatch,
    empty_candidate, candidate_hash_mismatch, cortex_metadata_in_candidate, not_round_trip_stable,
    manual_review_required, error, proposal_write_failed, backup_failed,
    write_failed, verification_failed, partial_failure (not ok)."""
    proposal, error = load_proposal(proposals_dir, proposal_id)
    if proposal is None:
        return _result(False, "invalid_proposal", error)
    status = proposal["status"]
    if status == "applied":
        record = proposal.get("apply") if isinstance(proposal.get("apply"), dict) else None
        return _result(True, "already_applied", "proposal was already applied; the source was not written again",
                       apply=record)
    if status == "pending":
        return _result(False, "not_approved", "proposal is pending; approve it first (--approve-proposal)")
    if status == "rejected":
        return _result(False, "rejected", "proposal was rejected; rejected proposals are never applied")
    if status == "stale":
        return _result(False, "stale", "proposal is stale; generate a fresh proposal (--propose-vault-changes)")
    if status != "approved":
        return _result(False, "invalid_proposal", f"unrecognized proposal status {status!r}")

    state_path = Path(state_path or DEFAULT_STATE_PATH)
    backups_dir = Path(backups_dir) if backups_dir else default_backups_dir(proposals_dir)
    proposal_file = _proposal_path(proposals_dir, proposal_id)
    fp = proposal["fingerprints"]
    body = proposal.get("reviewed_live_vault_body")
    try:
        suffix = _check_eligibility(proposal, state_path)
        if "apply_intent" in proposal:
            recovery = await _recover(proposal, proposal_file, obsidian, state_path, dry_run)
            if recovery is not None:
                return recovery
            del proposal["apply_intent"]  # replace never happened: rerun the full path

        try:
            drift = await detect_drift(proposal, obsidian, state_path)
        except Exception as exc:  # noqa: BLE001 - unreadable authority: no write, no status change
            return _result(False, "error", f"could not revalidate authorities: {exc}")
        if drift:
            return _mark_stale(proposal, proposal_file, drift, dry_run)

        raw_before = _read_source(proposal["source_path"])
        if extracted_sha256(raw_before) != fp["source_sha256"]:
            return _mark_stale(proposal, proposal_file, ["source changed during revalidation"], dry_run)
        candidate, info = build_candidate(suffix, raw_before, body, fp["live_vault_sha256"])
    except ApplyRefused as refused:
        if refused.status == "stale":
            return _mark_stale(proposal, proposal_file, [refused.reason], dry_run)
        return _result(False, refused.status, refused.reason)

    details = {
        "source_path": proposal["source_path"],
        "pre_apply_source_sha256": fp["source_sha256"],
        "pre_apply_source_bytes_sha256": _bytes_sha256(raw_before),
        "post_apply_source_sha256": info["expected_post_apply_source_sha256"],
        "post_apply_source_bytes_sha256": info["candidate_bytes_sha256"],
        "frontmatter_preserved": info["frontmatter_preserved"],
    }
    if dry_run:
        before = _normalize_newlines(raw_before.decode("utf-8"))
        after = _normalize_newlines(candidate.decode("utf-8"))
        diff, truncated, total = _diff(before.lstrip("﻿"), after.lstrip("﻿"), max_diff_lines)
        return _result(True, "would_apply", "dry run: all checks passed; nothing was written",
                       backup_path=str(next_backup_path(backups_dir, proposal_id, proposal["source_path"])),
                       diff=diff, diff_truncated=truncated, diff_total_lines=total, **details)

    try:
        backup_path = create_backup(backups_dir, proposal_id, proposal["source_path"], raw_before)
    except OSError as exc:
        return _result(False, "backup_failed", f"could not create backup; source not written: {exc}", **details)
    details["backup_path"] = str(backup_path)

    proposal["apply_intent"] = {
        "started_at": datetime.now().isoformat(),
        "source_path": proposal["source_path"],
        "pre_apply_source_bytes_sha256": details["pre_apply_source_bytes_sha256"],
        "candidate_bytes_sha256": info["candidate_bytes_sha256"],
        "expected_post_apply_source_sha256": info["expected_post_apply_source_sha256"],
        "backup_path": str(backup_path),
        "frontmatter_preserved": info["frontmatter_preserved"],
    }
    try:
        _atomic_write_json(proposal_file, proposal)
    except OSError as exc:
        return _result(False, "proposal_write_failed",
                       f"could not record apply intent; source not written (backup kept): {exc}", **details)

    try:
        atomic_replace(proposal["source_path"], candidate, raw_before)
    except _SourceChangedDuringApply:
        result = _mark_stale(proposal, proposal_file, ["source changed during apply"], dry_run=False)
        result.update(details)
        return result
    except OSError as exc:
        _clear_intent(proposal, proposal_file)
        return _result(False, "write_failed", f"atomic source write failed; original untouched: {exc}", **details)

    problems = verify_written(Path(proposal["source_path"]).read_bytes(), candidate, info, fp["live_vault_sha256"])
    if problems:
        return _result(False, "verification_failed",
                       "SOURCE WAS REPLACED BUT POST-WRITE VERIFICATION FAILED: " + "; ".join(problems)
                       + f". Restore from the backup if needed: {backup_path}", **details)

    record = _apply_record(proposal, raw_before, info, backup_path, False)
    if actor:
        record["actor"] = actor
    return _finalize(proposal, proposal_file, record, "applied", details)


def _finalize(proposal, proposal_file, record, status, details):
    proposal["status"] = "applied"
    proposal["apply"] = record
    intent = proposal.pop("apply_intent", None)
    try:
        _atomic_write_json(proposal_file, proposal)
    except OSError as exc:
        proposal["status"] = "approved"
        proposal.pop("apply", None)
        if intent is not None:
            proposal["apply_intent"] = intent
        return _result(False, "partial_failure",
                       f"PARTIAL TRANSACTION: the source now holds the approved content, but the proposal "
                       f"could not be marked applied ({exc}). Rerun --apply-proposal: it will recognize the "
                       f"written content and record the application without rewriting the source.",
                       apply=record, **details)
    return _result(True, status, None, apply=record, **details)


def _clear_intent(proposal, proposal_file):
    proposal.pop("apply_intent", None)
    try:
        _atomic_write_json(proposal_file, proposal)
    except OSError:
        pass  # a leftover intent is handled by _recover (source == pre-apply bytes)


def _mark_stale(proposal, proposal_file, reasons, dry_run):
    reason = "; ".join(reasons)
    if not dry_run:
        proposal["status"] = "stale"
        proposal.pop("apply_intent", None)
        proposal["stale_on_apply"] = {"at": datetime.now().isoformat(), "reasons": list(reasons)}
        try:
            _atomic_write_json(proposal_file, proposal)
        except OSError as exc:
            reason += f" (could not persist stale status: {exc})"
    return _result(False, "stale", f"{reason}. Nothing was written; generate a fresh proposal")


async def _recover(proposal, proposal_file, obsidian, state_path, dry_run):
    """Resolve a leftover apply_intent. Returns a result, or None when the
    source still holds the pre-apply bytes (caller reruns the full path)."""
    intent = proposal["apply_intent"]
    keys = ("pre_apply_source_bytes_sha256", "candidate_bytes_sha256", "expected_post_apply_source_sha256")
    if not isinstance(intent, dict) or intent.get("source_path") != proposal["source_path"] \
            or not all(isinstance(intent.get(k), str) and _SHA256_RE.match(intent[k]) for k in keys):
        raise ApplyRefused("invalid_proposal", "apply_intent is malformed")
    backup = intent.get("backup_path")
    try:
        raw_now = Path(proposal["source_path"]).read_bytes()
    except OSError as exc:
        raise ApplyRefused("manual_review_required",
                           f"interrupted apply: source unreadable ({exc}); backup: {backup}") from exc
    now = _bytes_sha256(raw_now)
    if now == intent["pre_apply_source_bytes_sha256"]:
        return None
    if now != intent["candidate_bytes_sha256"]:
        raise ApplyRefused("manual_review_required",
                           f"interrupted apply: source matches neither the pre-apply nor the approved "
                           f"content; nothing written. Backup: {backup}")

    # The approved content is already in the source. Check the other two
    # authorities against the snapshot with the source fingerprint swapped
    # for the approved post-apply hash; drift is a warning, not a refusal,
    # because the write already happened and must be recorded truthfully.
    expected_world = dict(proposal, fingerprints=dict(
        proposal["fingerprints"], source_sha256=intent["expected_post_apply_source_sha256"]))
    try:
        warnings = await detect_drift(expected_world, obsidian, state_path)
    except Exception as exc:  # noqa: BLE001
        warnings = [f"could not revalidate authorities: {exc}"]
    raw_backup = None
    if backup and os.path.isfile(backup):
        raw_backup = Path(backup).read_bytes()
    if raw_backup is None or _bytes_sha256(raw_backup) != intent["pre_apply_source_bytes_sha256"]:
        warnings.append(f"backup {backup} is missing or does not match the pre-apply bytes")
    info = {"expected_post_apply_source_sha256": intent["expected_post_apply_source_sha256"],
            "candidate_bytes_sha256": intent["candidate_bytes_sha256"],
            "frontmatter_preserved": bool(intent.get("frontmatter_preserved"))}
    details = {"source_path": proposal["source_path"], "backup_path": backup, "warnings": warnings}
    if dry_run:
        return _result(True, "would_recover",
                       "dry run: the source already holds the approved content; the application would "
                       "be recorded without writing the source", **details)
    record = {
        "applied_at": datetime.now().isoformat(),
        "source_path": proposal["source_path"],
        "pre_apply_source_sha256": proposal["fingerprints"]["source_sha256"],
        "pre_apply_source_bytes_sha256": intent["pre_apply_source_bytes_sha256"],
        "post_apply_source_sha256": info["expected_post_apply_source_sha256"],
        "post_apply_source_bytes_sha256": info["candidate_bytes_sha256"],
        "backup_path": backup,
        "frontmatter_preserved": info["frontmatter_preserved"],
        "recovered": True,
        "warnings": warnings,
    }
    return _finalize(proposal, proposal_file, record, "recovered", details)
