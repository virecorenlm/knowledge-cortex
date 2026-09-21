"""CLI entry point: sync an Obsidian vault into the Qdrant semantic index.

Usage:
    python sync_cli.py                      # sync entire vault, incremental
    python sync_cli.py --root Vire_Realm     # sync one subtree
    python sync_cli.py --full                # ignore saved state, reindex everything
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from graph.store import VectorStore
from ingest.obsidian_client import ObsidianClient
from ingest.sync import sync_vault_to_index

DEFAULT_STATE_PATH = Path(__file__).with_name("sync_state.json")
DEFAULT_EXCLUDES = ("cache/", "venv/", "incoming_memory_dump", "memory_logs/", ".git/")


def load_state(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path, state):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(path)


async def main_async(args):
    obsidian = ObsidianClient()
    store = VectorStore()
    state = {} if args.full else load_state(args.state)
    report, state = await sync_vault_to_index(
        obsidian, store, root=args.root, exclude_substrings=DEFAULT_EXCLUDES, state=state,
    )
    save_state(args.state, state)
    print(json.dumps({
        "indexed_count": len(report["indexed"]),
        "skipped_unchanged_count": len(report["skipped_unchanged"]),
        "error_count": len(report["errors"]),
        "errors": report["errors"],
        "collection": store.collection,
        "embedding_model": store.embedding_model,
    }, indent=2))
    return 1 if report["errors"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="", help="Vault subdirectory to sync (default: whole vault)")
    parser.add_argument("--full", action="store_true", help="Ignore saved state; reindex every note")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH, help="Path to the sync state file")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
