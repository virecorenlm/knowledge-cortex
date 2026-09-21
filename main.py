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


def index_to_qdrant(input_path, out_dir, state_path, store=None):
    """New behavior (--index): extract + chunk + embed + upsert into the
    same Qdrant collection VectorStore/sync_cli use, via
    ingest.sync.index_local_path — no duplicated extraction, chunking, or
    embedding logic. Also writes the generated Markdown into out_dir (when
    out_dir is given) so the vault copy and the semantic index stay in sync.
    Incremental via sha256 state stored under the "local" namespace of the
    same sync_state.json vault sync already uses.

    store: optional pre-built VectorStore (for tests); defaults to
    VectorStore() reading OLLAMA_URL/EMBEDDING_MODEL/QDRANT_URL/QDRANT_COLLECTION
    from the environment, same as sync_cli.py.
    """
    from graph.store import VectorStore
    from ingest.sync import index_local_path
    from sync_cli import load_state, save_state

    store = store or VectorStore()
    state = load_state(state_path, namespace="local")
    report, state = index_local_path(store, input_path, state=state)
    save_state(state_path, state, namespace="local")

    for entry in report["indexed"]:
        print("Processing:", entry["source_file"])
        if out_dir:
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
          f"{len(report['errors'])} error(s)).")
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
    parser.add_argument("--state", type=Path, default=Path(__file__).with_name("sync_state.json"),
                         help="Path to the shared sync state file (default: sync_state.json)")
    args = parser.parse_args()

    if args.index:
        return index_to_qdrant(args.input, args.out, args.state)

    out_dir = args.out or "vault"
    extract_to_markdown(args.input, out_dir)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
