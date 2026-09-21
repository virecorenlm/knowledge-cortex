"""Orchestrates Obsidian <-> Knowledge Cortex sync.

- sync_vault_to_index: reads notes from the Obsidian vault (via MCP) and
  indexes them into Qdrant, skipping unchanged notes (by sha256) so repeat
  runs are cheap.
- index_local_folder: same, but for local files on disk (PDF/DOCX/TXT/MD via
  the existing ingest.extract pipeline) — this is the "write results into the
  vault" direction: extract -> markdown -> write into Obsidian -> index.
"""

import hashlib

from ingest.chunk import build_chunks


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


def index_local_path(store, path, state=None, max_chars=1200, markdown_prefix="local_ingest"):
    """Extract + chunk + index one file, or every supported file under a
    directory (recursively). This is the local-file entry point wired into
    main.py: files don't only become Markdown, they also enter the same
    Qdrant semantic index used by the Obsidian vault sync.

    Reuses ingest.detect/ingest.extract/ingest.markdown for extraction and
    ingest.chunk (via store.index_document) for chunking — no duplicated
    logic. Content-hash state (like sync_vault_to_index) makes repeat runs
    skip unchanged files and safely replace chunks for changed ones with no
    orphaned points, via VectorStore.index_document's delete-then-upsert.

    state: optional dict of {absolute_source_path: sha256} from a previous
    run. Callers own persistence (see sync_cli.load_state/save_state with
    namespace="local"). Only successfully-indexed files update state, so a
    failed file is retried on the next run rather than silently marked done.

    Returns (report, state). report has:
      indexed:            [{source_file, markdown_path, chunk_count, markdown}]
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
    from ingest.markdown import to_markdown
    from utils.fs import iter_files

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

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if state.get(abs_path) == digest:
            report["skipped_unchanged"].append(abs_path)
            continue

        try:
            md = to_markdown(text, source=abs_path)
            markdown_path = f"{markdown_prefix}/{Path(file).name}"
            chunks = store.index_document(
                markdown_path, md, source="local_ingest", max_chars=max_chars,
                metadata={"source_file": abs_path, "markdown_path": markdown_path},
            )
        except Exception as exc:  # noqa: BLE001 - report and continue; do not update state on failure
            report["errors"].append({"path": abs_path, "error": str(exc)})
            continue

        state[abs_path] = digest
        report["indexed"].append({
            "source_file": abs_path, "markdown_path": markdown_path,
            "chunk_count": len(chunks), "markdown": md,
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
