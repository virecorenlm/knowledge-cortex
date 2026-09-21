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


def index_to_qdrant(input_path, out_dir, state_path, store=None, ai_structure=False, structure_fn=None):
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
                                      structure_fn=structure_fn)
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
    for path in report["skipped_unchanged"]:
        print("Unchanged, skipped:", path)
    for path in report["skipped_unsupported"]:
        print("Unsupported, skipped:", path)
    for path in report["skipped_empty"]:
        print("Empty, skipped:", path)
    for err in report["errors"]:
        print("ERROR:", err["path"], "-", err["error"])

    print()
    print(f"Indexed {len(report['indexed'])} file(s) into Qdrant collection "
          f"'{store.collection}' using {store.embedding_model} "
          f"({len(report['skipped_unchanged'])} unchanged, "
          f"{len(report['errors'])} error(s), "
          f"{len(skipped_existing_notes)} existing note(s) not overwritten).")
    return 1 if report["errors"] else 0


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Neural ingestion → Obsidian vault (+ optional Qdrant indexing)")
    parser.add_argument("input", help="Input file or folder")
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
    parser.add_argument("--state", type=Path, default=Path(__file__).with_name("sync_state.json"),
                         help="Path to the shared sync state file (default: sync_state.json)")
    args = parser.parse_args()

    if args.ai_structure and not args.index:
        parser.error("--ai-structure requires --index")

    if args.index:
        return index_to_qdrant(args.input, args.out, args.state, ai_structure=args.ai_structure)

    out_dir = args.out or "vault"
    extract_to_markdown(args.input, out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
