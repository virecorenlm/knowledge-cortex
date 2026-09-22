import argparse

from ingest.detect import detect_file_type
from ingest.extract import extract_text
from ingest.markdown import to_markdown
from utils.fs import iter_files, safe_write


def extract_to_markdown(input_path, out_dir):
    """Original behavior: extract every supported file under input_path and
    write plain Markdown into out_dir. No embedding, no Qdrant. Used when
    --index is not passed, so existing workflows are unaffected."""
    for file in iter_files(input_path):
        print("Processing:", file)

        ftype = detect_file_type(file)
        text = extract_text(file, ftype)

        if not text.strip():
            continue

        md = to_markdown(text, source=file)

        print("Writing markdown for:", file)
        safe_write(md, file, out_dir)


def index_to_qdrant(input_path, out_dir, state_path, store=None, ai_structure=False, structure_fn=None,
                     write_vault=False, managed_vault_dir=None, obsidian=None, project=None):
    """New behavior (--index): extract + chunk + embed + upsert into the
    same Qdrant collection VectorStore/sync_cli use, via
    ingest.sync.index_local_path — no duplicated extraction, chunking, or
    embedding logic. Also writes the generated Markdown into out_dir (when
    out_dir is given) so the vault copy and the semantic index stay in sync
    — UNLESS a file already exists at that destination path, in which case
    the write is skipped and reported rather than silently overwriting what
    may be a human-authored vault note (the file is still indexed either
    way; only the Markdown write is skipped).
    Incremental via sha256 state stored under the "local" namespace of the
    same sync_state.json vault sync already uses.

    ai_structure: when True, extracted Markdown is passed through
    ingest.ai_struct.structure_markdown before indexing (opt-in; see
    ingest/ai_struct.py and ingest/sync.py:index_local_path for the
    fallback contract and incremental-reprocessing rules). Default False:
    pipeline behavior is identical to before this flag existed.

    structure_fn: optional override for the structuring call, forwarded to
    index_local_path (for tests / alternate providers); defaults to
    ingest.ai_struct.structure_markdown.

    write_vault: when True (opt-in, independent of --out — --out writes
    plain local Markdown files with no conflict protection; write_vault
    writes into the LIVE Obsidian vault via the managed, conflict-safe path
    in ingest/vault_writer.py), every successfully indexed entry from this
    run is also written to a dedicated managed subtree of the vault
    (managed_vault_dir). A note is only ever created/updated there if it
    can be proven cortex-owned and unmodified since cortex's last write;
    otherwise the attempt is reported as a conflict and the vault note is
    left untouched. Requires network access to the Obsidian MCP server;
    genuine connection/auth failures are reported per-entry like any other
    error and do not abort the rest of the batch.

    managed_vault_dir: destination subtree for write_vault (default:
    "Knowledge Cortex/Managed"). Only used when write_vault is True.

    obsidian: optional pre-built ObsidianClient (for tests); defaults to
    ObsidianClient() reading OBSIDIAN_MCP_URL/OBSIDIAN_MCP_AUTHORIZATION
    from the environment. Only used when write_vault is True.

    project: optional project name stored as filterable metadata on every
    indexed chunk (see ingest/metadata.py).

    store: optional pre-built VectorStore (for tests); defaults to
    VectorStore() reading OLLAMA_URL/EMBEDDING_MODEL/QDRANT_URL/QDRANT_COLLECTION
    from the environment, same as sync_cli.py.
    """
    from pathlib import Path
    from graph.store import VectorStore
    from ingest.sync import index_local_path
    from sync_cli import load_state, save_state

    store = store or VectorStore()
    state = load_state(state_path, namespace="local")
    report, state = index_local_path(store, input_path, state=state, ai_structure=ai_structure,
                                      structure_fn=structure_fn, project=project)

    vault_results = []
    if write_vault:
        import asyncio
        from ingest.vault_writer import write_managed_note, source_id_for
        from ingest.obsidian_client import ObsidianClient

        managed_vault_dir = (managed_vault_dir or "Knowledge Cortex/Managed").rstrip("/")
        client = obsidian or ObsidianClient()

        # Vault write-back is decoupled from Qdrant's own unchanged-skip:
        # a source that Qdrant skipped (no re-extraction, no re-structuring,
        # no re-embedding) still gets its managed vault note verified/
        # written when --write-vault is explicitly requested, using the
        # generated_body cached in state by index_local_path — never by
        # re-running extraction or AI structuring. This is what makes the
        # "human edited the vault note; source never changed" conflict
        # detectable without forcing an unnecessary Qdrant reindex.
        vault_candidates = list(report["indexed"])
        for source_file in report["skipped_unchanged"]:
            cached = state.get(source_file, {})
            cached_body = cached.get("generated_body")
            if cached_body is None:
                # Should not happen (index_local_path only skips when a
                # cached body is present), but never attempt a vault write
                # with no known content -- skip resiliently instead.
                continue
            vault_candidates.append({
                "source_file": source_file,
                "markdown": cached_body,
                "document_body": cached_body,
                "structure_model": cached.get("structure_model"),
            })

        async def _write_all():
            results = []
            for entry in vault_candidates:
                source_id = source_id_for(entry["source_file"])
                basename = Path(entry["source_file"]).stem
                dest_path = f"{managed_vault_dir}/{basename}-{source_id}.md"
                state_entry = state.get(entry["source_file"], {})
                metadata = {
                    "source_path": entry["source_file"],
                    "source_sha256": state_entry.get("source_sha256", ""),
                    "prompt_version": state_entry.get("prompt_version"),
                    "structure_model": entry.get("structure_model"),
                }
                body = entry.get("document_body", entry["markdown"])
                try:
                    result = await write_managed_note(client, dest_path, body, metadata)
                except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
                    result = {"status": "error", "path": dest_path, "reason": str(exc), "generated_sha256": None}
                result["source_file"] = entry["source_file"]
                # Record the outcome in local state for observability/debugging only —
                # the vault note itself remains the sole authority for conflict
                # detection on the NEXT run (see ingest/vault_writer.py's module
                # docstring); state is never consulted to decide whether a write is
                # safe, so a stale or deleted state file cannot cause an unsafe write.
                state.setdefault(entry["source_file"], {})["vault_write"] = {
                    "dest_path": dest_path, "status": result["status"],
                    "generated_sha256": result["generated_sha256"],
                }
                results.append(result)
            return results

        vault_results = asyncio.run(_write_all())

    save_state(state_path, state, namespace="local")

    skipped_existing_notes = []
    for entry in report["indexed"]:
        print("Processing:", entry["source_file"])
        if entry.get("structure_reason"):
            print(f"  Structuring: {'used' if entry['ai_structured'] else 'fell back to raw Markdown'} "
                  f"({entry['structure_reason']})")
        elif ai_structure:
            print(f"  Structuring: {'used' if entry['ai_structured'] else 'fell back to raw Markdown'}")
        if out_dir:
            dest = Path(out_dir) / Path(entry["source_file"]).with_suffix(".md").name
            if ai_structure and dest.exists():
                skipped_existing_notes.append(str(dest))
                print(f"  NOT writing Markdown: {dest} already exists (refusing to overwrite an existing note)")
            else:
                print("Writing markdown for:", entry["source_file"])
                safe_write(entry["markdown"], entry["source_file"], out_dir)
    for result in vault_results:
        label = {"created": "Vault: created", "updated": "Vault: updated",
                  "skipped": "Vault: unchanged, skipped", "conflict": "Vault CONFLICT (not written)",
                  "error": "Vault ERROR"}[result["status"]]
        suffix = f" ({result['reason']})" if result.get("reason") else ""
        print(f"  {label}: {result['path']}{suffix}")
    for path in report["skipped_unchanged"]:
        print("Unchanged, skipped:", path)
    for path in report["metadata_updated"]:
        print("  Filter metadata updated (no re-embedding):", path)
    for path in report["skipped_unsupported"]:
        print("Unsupported, skipped:", path)
    for path in report["skipped_empty"]:
        print("Empty, skipped:", path)
    for err in report["errors"]:
        print("ERROR:", err["path"], "-", err["error"])

    conflict_count = sum(1 for r in vault_results if r["status"] == "conflict")
    vault_error_count = sum(1 for r in vault_results if r["status"] == "error")
    print()
    print(f"Indexed {len(report['indexed'])} file(s) into Qdrant collection "
          f"'{store.collection}' using {store.embedding_model} "
          f"({len(report['skipped_unchanged'])} unchanged, "
          f"{len(report['errors'])} error(s), "
          f"{len(skipped_existing_notes)} existing note(s) not overwritten"
          + (f", {conflict_count} vault conflict(s), {vault_error_count} vault error(s)"
             if write_vault else "") + ").")
    return 1 if (report["errors"] or vault_error_count) else 0


def analyze_vault(state_path, source=None, json_output=False, max_diff_lines=200, obsidian=None):
    """Read-only entry point for `--analyze-vault`. Loads the "local"
    namespace of sync_state.json (never writes it back), analyzes every
    managed note found there (or just `source` if given) via
    ingest.reverse_analyzer.analyze_managed_notes, and prints results.

    Performs ZERO writes: no Obsidian write, no state save, no Qdrant
    access at all (VectorStore is never imported here). This is strictly
    an inspection/dry-run path -- see ingest/reverse_analyzer.py's module
    docstring for the full authority/classification model.

    obsidian: optional pre-built ObsidianClient (for tests); defaults to
    ObsidianClient() reading OBSIDIAN_MCP_URL/OBSIDIAN_MCP_AUTHORIZATION
    from the environment.

    Returns 0 always on a completed analysis run (a note being in a
    non-IN_SYNC state is information, not a process failure); returns 1
    only if the state file itself could not be read at all.
    """
    import asyncio
    import json as json_module
    from ingest.obsidian_client import ObsidianClient
    from ingest.reverse_analyzer import analyze_managed_notes
    from sync_cli import load_state

    try:
        state = load_state(state_path, namespace="local")
    except Exception as exc:  # noqa: BLE001 - report clearly, don't crash
        print(f"ERROR: could not read state file {state_path}: {exc}")
        return 1

    client = obsidian or ObsidianClient()
    results = asyncio.run(analyze_managed_notes(client, state, source_filter=source,
                                                 max_diff_lines=max_diff_lines))
    # _live_vault_body is an internal field for ingest.proposals.create_proposals
    # to consume (the exact content a future proposal would need to
    # capture) -- never surfaced in analyze_vault's own output, to avoid
    # dumping potentially large document bodies into a routine report.
    display_results = [{k: v for k, v in r.items() if k != "_live_vault_body"} for r in results]

    if json_output:
        print(json_module.dumps(display_results, indent=2))
        return 0

    if not display_results:
        print("No managed notes found in state (nothing has been written via --write-vault yet).")
        return 0

    for r in display_results:
        print(f"[{r['classification']}] {r['source_path']}")
        print(f"  managed note: {r.get('managed_note_path')}")
        print(f"  reason: {r.get('reason')}")
        print(f"  proposed action: {r.get('proposed_action')}")
        flags = r.get("flags") or {}
        active_flags = [k for k, v in flags.items() if v]
        if active_flags:
            print(f"  flags: {', '.join(active_flags)}")
        if r.get("diff"):
            print("  diff (cortex_generated -> vault_current):")
            for line in r["diff"]:
                print(f"    {line}")
            if r.get("diff_truncated"):
                print(f"    ... truncated ({r['diff_total_lines']} total diff lines)")
        print()
    return 0


def propose_vault_changes(state_path, proposals_dir=None, source=None, max_diff_lines=200,
                           json_output=False, obsidian=None):
    """Read-only entry point for `--propose-vault-changes`. Runs the
    existing analyze_vault analysis pass, then turns each actionable
    result into a durable proposal file via ingest.proposals.create_proposals.

    Performs ZERO writes to source/Obsidian/Qdrant/sync_state.json; the
    only writes are new proposal JSON files under proposals_dir. Existing
    proposals for an unchanged situation are left untouched (see
    ingest.proposals.create_proposals's idempotency contract).
    """
    import asyncio
    import json as json_module
    from ingest.obsidian_client import ObsidianClient
    from ingest.reverse_analyzer import analyze_managed_notes
    from ingest.proposals import create_proposals, DEFAULT_PROPOSALS_DIR
    from sync_cli import load_state

    proposals_dir = proposals_dir or DEFAULT_PROPOSALS_DIR
    try:
        state = load_state(state_path, namespace="local")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not read state file {state_path}: {exc}")
        return 1

    client = obsidian or ObsidianClient()
    results = asyncio.run(analyze_managed_notes(client, state, source_filter=source,
                                                 max_diff_lines=max_diff_lines))
    created = create_proposals(results, local_state=state, proposals_dir=proposals_dir, state_path=state_path)

    if json_output:
        print(json_module.dumps(created, indent=2))
        return 0

    if not created:
        print("No actionable divergence found (nothing to propose).")
        return 0

    for c in created:
        verb = "Created" if c["created"] else f"Already exists (status: {c['status']})"
        print(f"[{c['classification']}] {verb}: proposal {c['proposal_id']}")
    return 0


def _decision_cli(proposals_dir, proposal_id, action, note, obsidian=None, state_path=None):
    import asyncio
    from ingest.obsidian_client import ObsidianClient
    from ingest.proposals import approve_proposal, reject_proposal, DEFAULT_PROPOSALS_DIR

    proposals_dir = proposals_dir or DEFAULT_PROPOSALS_DIR
    if action == "approve":
        client = obsidian or ObsidianClient()
        result = asyncio.run(approve_proposal(proposals_dir, proposal_id, client, note=note, state_path=state_path))
    else:
        result = reject_proposal(proposals_dir, proposal_id, note=note)

    if result["ok"]:
        print(f"Proposal {proposal_id}: {result['status']}")
        return 0
    print(f"Proposal {proposal_id} NOT {action}d: {result['reason']}")
    return 1


def list_proposals_cli(proposals_dir=None, json_output=False):
    import json as json_module
    from ingest.proposals import list_proposals, DEFAULT_PROPOSALS_DIR

    proposals_dir = proposals_dir or DEFAULT_PROPOSALS_DIR
    entries = list_proposals(proposals_dir)

    if json_output:
        print(json_module.dumps([
            {"proposal": p, "error": e} for p, e in entries
        ], indent=2))
        return 0

    if not entries:
        print(f"No proposals found in {proposals_dir}.")
        return 0

    for proposal, error in entries:
        if error:
            print(f"[UNREADABLE] {error}")
            continue
        print(f"[{proposal['status'].upper()}] {proposal['proposal_id']}  "
              f"{proposal['classification']}  {proposal['source_path']}")
    return 0


def show_proposal_cli(proposal_id, proposals_dir=None, json_output=False):
    import json as json_module
    from ingest.proposals import load_proposal, DEFAULT_PROPOSALS_DIR

    proposals_dir = proposals_dir or DEFAULT_PROPOSALS_DIR
    proposal, error = load_proposal(proposals_dir, proposal_id)
    if proposal is None:
        print(f"ERROR: {error}")
        return 1

    if json_output:
        print(json_module.dumps(proposal, indent=2))
        return 0

    print(f"Proposal:        {proposal['proposal_id']}")
    print(f"Status:          {proposal['status']}")
    print(f"Classification:  {proposal['classification']}")
    print(f"Proposed action: {proposal['proposed_action']}")
    print(f"Source:          {proposal['source_path']}")
    print(f"Managed note:    {proposal['managed_note_path']}")
    print(f"Created:         {proposal['created_at']}")
    print(f"Reason:          {proposal['reason']}")
    fp = proposal["fingerprints"]
    print("Fingerprints:")
    print(f"  source_sha256:             {fp.get('source_sha256')}")
    print(f"  expected_generated_sha256: {fp.get('expected_generated_sha256')}")
    print(f"  live_vault_sha256:         {fp.get('live_vault_sha256')}")
    if proposal.get("ai_structured"):
        print(f"AI structured:   true (model: {proposal.get('structure_model')})")
    decision = proposal.get("decision") or {}
    if decision.get("status"):
        print(f"Decision:        {decision['status']} at {decision['decided_at']}"
              + (f" -- {decision['note']}" if decision.get("note") else ""))
    if proposal.get("diff"):
        print("Diff (cortex_generated -> vault_current):")
        for line in proposal["diff"]:
            print(f"  {line}")
        if proposal.get("diff_truncated"):
            print(f"  ... truncated ({proposal['diff_total_lines']} total diff lines)")
    return 0


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Neural ingestion → Obsidian vault (+ optional Qdrant indexing)")
    parser.add_argument("input", nargs="?", default=None, help="Input file or folder (not used with --analyze-vault)")
    parser.add_argument("--out", default=None,
                         help="Output Obsidian vault folder for generated Markdown (optional with --index)")
    parser.add_argument("--index", action="store_true",
                         help="Also embed and index content into Qdrant (qwen3-embedding:4b), "
                              "incrementally via sync_state.json. Without this flag, behavior is "
                              "unchanged: extract-to-Markdown only, no embedding, no Qdrant.")
    parser.add_argument("--ai-structure", action="store_true", dest="ai_structure",
                         help="Requires --index. Pass extracted Markdown through an AI structuring "
                              "pass (headings/summary/tags) before chunking/indexing. Opt-in; falls "
                              "back safely to raw Markdown if structuring fails or produces invalid "
                              "output. See ingest/ai_struct.py for the structuring contract.")
    parser.add_argument("--write-vault", action="store_true", dest="write_vault",
                         help="Requires --index. Also write each indexed document into a dedicated "
                              "managed subtree of the live Obsidian vault (see --managed-vault-dir), "
                              "via a conflict-safe path that refuses to touch any note it cannot "
                              "prove it owns and last wrote unmodified. Independent of --out, which "
                              "writes plain local files with no conflict protection. See "
                              "ingest/vault_writer.py for the ownership/conflict model.")
    parser.add_argument("--managed-vault-dir", default=None, dest="managed_vault_dir",
                         help="Destination subtree inside the vault for --write-vault "
                              "(default: 'Knowledge Cortex/Managed'). Only used with --write-vault.")
    parser.add_argument("--analyze-vault", action="store_true", dest="analyze_vault",
                         help="READ-ONLY: classify every managed note's sync state (IN_SYNC, "
                              "HUMAN_MODIFIED, MISSING, etc.) against the last cortex-generated "
                              "body cached in sync_state.json. Makes zero writes to Obsidian, "
                              "sync_state.json, or Qdrant. Does not require the 'input' argument. "
                              "See ingest/reverse_analyzer.py for the classification model.")
    parser.add_argument("--source", default=None,
                         help="With --analyze-vault: only analyze this one source path "
                              "(must match a key in sync_state.json's local namespace exactly).")
    parser.add_argument("--json", action="store_true", dest="json_output",
                         help="With --analyze-vault: print machine-readable JSON instead of "
                              "human-readable text.")
    parser.add_argument("--max-diff-lines", type=int, default=200, dest="max_diff_lines",
                         help="With --analyze-vault: truncate unified diffs for HUMAN_MODIFIED "
                              "notes after this many lines (default: 200).")
    parser.add_argument("--propose-vault-changes", action="store_true", dest="propose_vault_changes",
                         help="READ-ONLY apart from writing new proposal files: runs the same "
                              "analysis as --analyze-vault and creates a durable, reviewable "
                              "proposal (state/proposals/<id>.json) for every actionable "
                              "divergence (everything except IN_SYNC/ANALYSIS_INSUFFICIENT_STATE). "
                              "Never modifies source/Obsidian/Qdrant/sync_state.json.")
    parser.add_argument("--approve-proposal", default=None, dest="approve_proposal",
                         help="Approve a pending proposal by id. Re-verifies live source/vault "
                              "fingerprints first; if anything drifted since the proposal was "
                              "created, marks it 'stale' instead of approving. Only ever writes "
                              "to the proposal's own JSON file -- never source/Obsidian/Qdrant/"
                              "sync_state.json.")
    parser.add_argument("--reject-proposal", default=None, dest="reject_proposal",
                         help="Reject a pending (or stale) proposal by id. Does not require live "
                              "fingerprints to still match. Only ever writes to the proposal's "
                              "own JSON file.")
    parser.add_argument("--decision-note", default=None, dest="decision_note",
                         help="Optional human-readable note attached to --approve-proposal or "
                              "--reject-proposal's decision record.")
    parser.add_argument("--list-proposals", action="store_true", dest="list_proposals",
                         help="List every proposal under --proposals-dir with its status.")
    parser.add_argument("--show-proposal", default=None, dest="show_proposal",
                         help="Show one proposal's full details (fingerprints, diff, decision) by id.")
    parser.add_argument("--proposals-dir", type=Path, default=None, dest="proposals_dir",
                         help="Directory for proposal JSON files (default: state/proposals/ "
                              "next to this repository).")
    parser.add_argument("--project", default=None,
                         help="Requires --index. Project name stored as filterable search metadata "
                              "on every indexed chunk (overrides a 'project:' in the document's "
                              "frontmatter).")
    parser.add_argument("--state", type=Path, default=Path(__file__).with_name("sync_state.json"),
                         help="Path to the shared sync state file (default: sync_state.json)")
    args = parser.parse_args()

    if args.propose_vault_changes:
        return propose_vault_changes(args.state, proposals_dir=args.proposals_dir, source=args.source,
                                      max_diff_lines=args.max_diff_lines, json_output=args.json_output)
    if args.approve_proposal:
        return _decision_cli(args.proposals_dir, args.approve_proposal, "approve", args.decision_note, state_path=args.state)
    if args.reject_proposal:
        return _decision_cli(args.proposals_dir, args.reject_proposal, "reject", args.decision_note)
    if args.list_proposals:
        return list_proposals_cli(proposals_dir=args.proposals_dir, json_output=args.json_output)
    if args.show_proposal:
        return show_proposal_cli(args.show_proposal, proposals_dir=args.proposals_dir, json_output=args.json_output)

    if args.analyze_vault:
        return analyze_vault(args.state, source=args.source, json_output=args.json_output,
                              max_diff_lines=args.max_diff_lines)

    if args.input is None:
        parser.error("the following arguments are required: input (unless --analyze-vault is given)")

    if args.ai_structure and not args.index:
        parser.error("--ai-structure requires --index")
    if args.write_vault and not args.index:
        parser.error("--write-vault requires --index")
    if args.project and not args.index:
        parser.error("--project requires --index")

    if args.index:
        return index_to_qdrant(args.input, args.out, args.state, ai_structure=args.ai_structure,
                                write_vault=args.write_vault, managed_vault_dir=args.managed_vault_dir,
                                project=args.project)

    out_dir = args.out or "vault"
    extract_to_markdown(args.input, out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
