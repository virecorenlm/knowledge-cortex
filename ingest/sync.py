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


def index_local_folder(store, root_folder, obsidian_prefix="local_ingest", max_chars=1200):
    """Extract + chunk + index every supported file under root_folder.

    Does not write into Obsidian; pair with obsidian.write_note per-file if
    the vault copy is also wanted (kept as two explicit steps so a caller can
    inspect/reject the generated markdown before writing).
    """
    from pathlib import Path
    from ingest.detect import detect_file_type
    from ingest.extract import extract_text
    from ingest.markdown import to_markdown
    from utils.fs import iter_files

    report = {"indexed": [], "skipped_empty": [], "skipped_unsupported": []}
    for file in iter_files(root_folder):
        ftype = detect_file_type(file)
        if ftype == "unknown":
            report["skipped_unsupported"].append(file)
            continue
        text = extract_text(file, ftype)
        if not text.strip():
            report["skipped_empty"].append(file)
            continue
        md = to_markdown(text, source=file)
        rel_path = f"{obsidian_prefix}/{Path(file).name}"
        chunks = store.index_document(rel_path, md, source="local_ingest", max_chars=max_chars)
        report["indexed"].append({"path": rel_path, "source_file": file, "chunk_count": len(chunks), "markdown": md})
    return report
