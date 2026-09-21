"""Orchestrates Obsidian <-> Knowledge Cortex sync.

- sync_vault_to_index: reads notes from the Obsidian vault (via MCP) and
  indexes them into Qdrant, skipping unchanged notes (by sha256) so repeat
  runs are cheap.
- index_local_folder: same, but for local files on disk (PDF/DOCX/TXT/MD via
  the existing ingest.extract pipeline) — this is the "write results into the
  vault" direction: extract -> markdown -> write into Obsidian -> index.
"""

import hashlib


async def sync_vault_to_index(obsidian, store, root="", exclude_substrings=(), state=None, max_chars=1200):
    """Index every markdown note under root into store.

    state: optional dict of {path: sha256} from a previous run, used to skip
    unchanged notes. Callers own persistence of this dict; this function only
    reads and updates it in memory and returns it.
    """
    state = {} if state is None else state
    report = {"indexed": [], "skipped_unchanged": [], "errors": []}
    async for path in obsidian.iter_markdown_paths(root=root, exclude_substrings=exclude_substrings):
        try:
            doc = await obsidian.read_note(path)
            text = doc["content"]
        except Exception as exc:  # noqa: BLE001 - report and continue, don't abort the whole sync
            report["errors"].append({"path": path, "error": str(exc)})
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if state.get(path) == digest:
            report["skipped_unchanged"].append(path)
            continue
        chunks = store.index_document(path, text, source="obsidian", max_chars=max_chars)
        state[path] = digest
        report["indexed"].append({"path": path, "chunk_count": len(chunks)})
    return report, state


def index_local_path(store, path, state=None, max_chars=1200, markdown_prefix="local_ingest",
                      ai_structure=False, structure_fn=None):
    """Extract + chunk + index one file, or every supported file under a
    directory (recursively). This is the local-file entry point wired into
    main.py: files don't only become Markdown, they also enter the same
    Qdrant semantic index used by the Obsidian vault sync.

    Reuses ingest.detect/ingest.extract/ingest.markdown for extraction and
    ingest.chunk (via store.index_document) for chunking — no duplicated
    logic. Content-hash state (like sync_vault_to_index) makes repeat runs
    skip unchanged files and safely replace chunks for changed ones with no
    orphaned points, via VectorStore.index_document's delete-then-upsert.

    ai_structure: when True, extracted Markdown is passed through
    ingest.ai_struct.structure_markdown before chunking/embedding. The
    structured text (or, on any failure, the original Markdown) becomes
    what's chunked and indexed — see the module docstring there for the
    fallback contract. Off by default; the pipeline is byte-for-byte
    unchanged when this is False.

    structure_fn: injectable in place of ingest.ai_struct.structure_markdown
    (for tests / alternate providers). Ignored when ai_structure is False.

    state: optional dict of {absolute_source_path: entry} from a previous
    run, where entry is either a legacy plain sha256 string (pre-AI-structure
    state) or a dict:
        {"source_sha256": "...",              # hash of extracted text
         "ai_structure_requested": bool,       # was --ai-structure passed for this run
         "ai_structure_succeeded": bool,       # did structuring actually succeed
                                                # (False whenever requested but it
                                                # fell back to raw Markdown)
         "structure_model": str | None,        # model attempted, if requested
         "prompt_version": int | None}         # ai_struct.PROMPT_VERSION, if requested
    A document is reindexed (not skipped) when: the source hash differs, the
    ai_structure flag (on/off) differs from last run, structuring was
    requested but the PREVIOUS attempt failed (ai_structure_succeeded is
    False) — so a failed/fallback attempt is always retried on the next run
    rather than being permanently stuck on raw Markdown — or, when
    structuring is requested and previously succeeded, the prompt_version
    differs. Callers own persistence (see sync_cli.load_state/save_state
    with namespace="local").

    Returns (report, state). report has:
      indexed:            [{source_file, markdown_path, chunk_count, markdown,
                             ai_structured, structure_model, structure_reason}]
      skipped_empty:       [source_file, ...]
      skipped_unsupported: [source_file, ...]
      skipped_unchanged:   [source_file, ...]
      errors:              [{path, error}]
    One bad file (extraction failure or embedding/indexing failure) is
    reported and skipped; it never aborts the batch or corrupts state for
    other files.
    """
    from pathlib import Path
    from ingest.detect import detect_file_type
    from ingest.extract import extract_text
    from ingest.markdown import to_markdown, split_frontmatter
    from utils.fs import iter_files

    if structure_fn is None:
        from ingest.ai_struct import structure_markdown as structure_fn, PROMPT_VERSION
    else:
        from ingest.ai_struct import PROMPT_VERSION

    state = {} if state is None else state
    root = Path(path)
    files = iter_files(root) if root.is_dir() else iter([str(root)])
    report = {"indexed": [], "skipped_empty": [], "skipped_unsupported": [],
              "skipped_unchanged": [], "errors": []}

    for file in files:
        abs_path = str(Path(file).resolve())
        try:
            ftype = detect_file_type(file)
            if ftype == "unknown":
                report["skipped_unsupported"].append(abs_path)
                continue
            text = extract_text(file, ftype)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            report["errors"].append({"path": abs_path, "error": str(exc)})
            continue
        if not text.strip():
            report["skipped_empty"].append(abs_path)
            continue

        source_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        model_for_state = None
        prior = state.get(abs_path)
        if isinstance(prior, str) or prior is None:
            # Legacy pre-AI-structure state (plain sha256 string) or no
            # prior entry at all: never requested, never succeeded.
            prior_digest = prior
            prior_requested = False
            prior_succeeded = False
            prior_prompt_version = None
            prior_has_cached_body = False
        else:
            prior_digest = prior.get("source_sha256")
            if "ai_structure_requested" in prior:
                prior_requested = bool(prior.get("ai_structure_requested"))
                prior_succeeded = bool(prior.get("ai_structure_succeeded"))
            else:
                # Transitional schema from before ai_structure_succeeded
                # existed: it only recorded whether structuring was
                # requested, not whether it actually succeeded. Treat as
                # "not confirmed successful" so it gets one safe retry
                # rather than risking a permanently-stuck raw fallback.
                prior_requested = bool(prior.get("ai_structure"))
                prior_succeeded = False
            prior_prompt_version = prior.get("prompt_version")
            # "generated_body" (the deterministic document body, minus the
            # ingestion frontmatter -- see below) is what --write-vault
            # reuses to verify/update the managed vault note WITHOUT
            # rerunning extraction or (expensive) AI structuring when
            # Qdrant itself skips an unchanged file. Older state written
            # before this field existed won't have it; treat that as "not
            # yet cacheable" so the file gets one forced reprocess to
            # populate it, exactly like the prompt_version-mismatch case
            # above -- not a special-cased skip condition, just another
            # instance of the same "prior state is insufficient" pattern.
            prior_has_cached_body = "generated_body" in prior

        # Skip only when: the source content is unchanged, the ai_structure
        # on/off setting matches the last run, the previous attempt actually
        # SUCCEEDED (if structuring is on) with the current prompt/contract
        # version, AND a cached generated_body is available for vault
        # write-back reuse. A previous attempt that requested structuring
        # but fell back to raw Markdown (prior_succeeded is False) is never
        # considered "unchanged" — it is always retried, so a document
        # can't get permanently stuck raw just because one run happened to
        # time out or fail validation.
        if (prior_digest == source_digest and prior_requested == bool(ai_structure)
                and (not ai_structure or (prior_succeeded and prior_prompt_version == PROMPT_VERSION))
                and prior_has_cached_body):
            report["skipped_unchanged"].append(abs_path)
            continue

        try:
            md = to_markdown(text, source=abs_path)
            structure_reason = None
            structured_ok = False
            if ai_structure:
                # Structure only the document body, not the frontmatter
                # (source path + live ingestion timestamp): the frontmatter
                # is pipeline-generated provenance metadata, not source
                # content, so it must not be sent to the model or be subject
                # to the "no dropped source values" validation in ai_struct
                # — a regenerated/reformatted timestamp would otherwise be
                # misread as lost information and cause a false rejection.
                frontmatter, _, body = md.partition("\n\n")
                result = structure_fn(body)
                indexed_text = f"{frontmatter}\n\n{result.text}" if result.ok else md
                structured_ok = result.ok
                structure_reason = result.reason
                model_for_state = result.model
            else:
                indexed_text = md

            markdown_path = f"{markdown_prefix}/{Path(file).name}"
            # document_body is the ACTUAL document content, with the
            # ingestion frontmatter (source:/ingested:) stripped off. This
            # is what gets reused for --write-vault: the managed note's
            # body should be the document itself, not a second stacked
            # frontmatter block. What gets chunked/embedded into Qdrant
            # (indexed_text) is unchanged and still includes the ingestion
            # frontmatter -- this only affects what a future write-vault
            # stage receives as the note body.
            _, document_body = split_frontmatter(indexed_text)
            metadata = {
                "source_file": abs_path,
                "markdown_path": markdown_path,
                "ai_structured": bool(ai_structure and structured_ok),
                "source_sha256": source_digest,
            }
            if ai_structure:
                metadata["structure_model"] = model_for_state
                # Only record structured_sha256 when structuring actually
                # succeeded — "ai_structured: true" must mean the indexed
                # text IS the AI-structured version, never merely that
                # structuring was attempted. On fallback, indexed_text is
                # the raw Markdown, so a "structured_sha256" here would be
                # actively misleading (a hash of raw content mislabeled as
                # structured output).
                if structured_ok:
                    metadata["structured_sha256"] = hashlib.sha256(indexed_text.encode("utf-8")).hexdigest()
            chunks = store.index_document(
                markdown_path, indexed_text, source="local_ingest", max_chars=max_chars,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue; do not update state on failure
            report["errors"].append({"path": abs_path, "error": str(exc)})
            continue

        state[abs_path] = {
            "source_sha256": source_digest,
            "ai_structure_requested": bool(ai_structure),
            "ai_structure_succeeded": bool(ai_structure and structured_ok),
            "structure_model": model_for_state if ai_structure else None,
            "prompt_version": PROMPT_VERSION if ai_structure else None,
            # Cached so --write-vault can verify/update the managed note on
            # a run where Qdrant itself skips this file as unchanged,
            # without re-extracting the source or re-invoking (possibly
            # expensive) AI structuring. See the skip-condition comment
            # above for why its absence forces one reprocess.
            "generated_body": document_body,
        }
        report["indexed"].append({
            "source_file": abs_path, "markdown_path": markdown_path,
            "chunk_count": len(chunks), "markdown": indexed_text,
            "document_body": document_body,
            "ai_structured": bool(ai_structure and structured_ok),
            "structure_model": model_for_state if ai_structure else None,
            "structure_reason": structure_reason,
        })
    return report, state


def index_local_folder(store, root_folder, obsidian_prefix="local_ingest", max_chars=1200):
    """Deprecated alias for index_local_path (no state tracking). Prefer
    index_local_path, which adds recursive-or-single-file handling, per-file
    error isolation, and incremental sha256-based skip/reindex."""
    report, _ = index_local_path(store, root_folder, state={}, max_chars=max_chars, markdown_prefix=obsidian_prefix)
    return {
        "indexed": [{"path": e["markdown_path"], "source_file": e["source_file"],
                     "chunk_count": e["chunk_count"], "markdown": e["markdown"]} for e in report["indexed"]],
        "skipped_empty": report["skipped_empty"],
        "skipped_unsupported": report["skipped_unsupported"],
    }
