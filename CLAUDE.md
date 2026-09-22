# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Use the project venv (`.venv/bin/python`). No build step, no linter configured.

```bash
# All tests (stdlib unittest; no live Ollama/Qdrant/Obsidian needed)
.venv/bin/python -m unittest discover -s tests -v

# One file / class / test
.venv/bin/python -m unittest tests.test_proposals
.venv/bin/python -m unittest tests.test_main_cli.IndexToQdrantTests
.venv/bin/python -m unittest tests.test_main_cli.IndexToQdrantTests.<test_name>

# Entry points
python sync_cli.py [--root SUBDIR] [--full]          # Obsidian vault -> Qdrant
python main.py <file|dir> --out <dir>                  # extract-only to Markdown (no embedding)
python main.py <file|dir> --index [--ai-structure] [--write-vault] [--project NAME]
python main.py --analyze-vault [--json] [--source PATH]
python main.py --propose-vault-changes | --list-proposals | --show-proposal ID
python main.py --approve-proposal ID | --reject-proposal ID [--decision-note "..."]
python mcp_server.py [--http --port 8770]              # exposes semantic_search / sync_vault / index_status
```

Runtime config is env-only: `OLLAMA_URL`, `EMBEDDING_MODEL` (default `qwen3-embedding:4b`), `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY`, `OBSIDIAN_MCP_URL`, `OBSIDIAN_MCP_AUTHORIZATION`, `AI_STRUCTURE_MODEL`, `AI_STRUCTURE_TIMEOUT_SECONDS`. `config.yaml` is empty/unused.

## Architecture

Pipeline: source (Obsidian vault over MCP, or local PDF/DOCX/TXT/MD) → extract → chunk → embed via Ollama → upsert into Qdrant. Both vault sync and local ingestion share the same `graph/store.py:VectorStore` and `ingest/chunk.py`.

- **`graph/store.py` `VectorStore`** — the only Ollama+Qdrant layer. Reads env at construction, and accepts injected `ollama_client`/`qdrant_client`. `ensure_collection` refuses a vector-size mismatch; use a new collection when the embedding model changes. (`graph/autolink.py` is legacy and not imported anywhere.)
- **`ingest/chunk.py`** — chunk IDs are derived from `(path, sha256, chunk_index)`. Re-indexing a document deletes its old chunks by path and then upserts, so unchanged content is a no-op and nothing is left orphaned.
- **`sync_cli.py`** owns the `sync_state.json` format: `load_state/save_state(path, namespace=...)`. There are two namespaces: `"vault"` (vault sync, `{path: sha256}`) and `"local"` (local ingestion; richer per-file records including cached `generated_body` and `vault_write`). The legacy flat format is migrated into `"vault"`. `main.py` and `mcp_server.py` import these helpers from `sync_cli`.
- **`ingest/sync.py`** — `sync_vault_to_index` (async, over `ingest/obsidian_client.py`) and `index_local_path` (local files). Callers handle persistence. An error in one file is recorded and skipped without aborting the batch, and that file's state is not updated, so it is retried next run.
- **`main.py`** — argparse CLI whose modes are thin functions (`extract_to_markdown`, `index_to_qdrant`, `analyze_vault`, `propose_vault_changes`, `_decision_cli`, `list_proposals_cli`, `show_proposal_cli`). Tests call these functions directly with fakes; only parser-validation tests run it as a subprocess.
- **`ingest/metadata.py`** derives the filterable payload fields `project`, `tags` and `doc_date` for every chunk. `graph/store.build_search_filter` turns a plain `filters` dict into a Qdrant `Filter` for `VectorStore.search` and MCP `semantic_search`. For local files, `index_local_path` caches `filter_metadata` in state. When that metadata changes on an unchanged file, it updates the existing points with `store.set_metadata` (`set_payload`), with no re-embedding.
- **`ingest/ai_struct.py`** — optional Ollama chat restructuring. It is fail-safe: any error, timeout, or validation rejection falls back to the raw Markdown. Only set `ai_structured: true` in metadata when the indexed text actually is the structured version.

### Obsidian write-back safety model (the core invariants)

These layers are deliberately staged, and each one does only what its phase allows. Preserve these guarantees when changing them:

1. **`ingest/vault_writer.py` (`--write-vault`)** only writes under `Knowledge Cortex/Managed/` as `<basename>-<source_id>.md`. It only overwrites a note that has `cortex_managed: true` and whose current body hash equals the recorded `cortex_generated_sha256`. The hash covers the **body only, never the frontmatter**. The live vault is the authority for conflict detection; `sync_state.json` is not. Human edits always win and are reported as conflicts. Each note has exactly one flat frontmatter block, and `markdown.split_frontmatter` strips ingestion frontmatter first.
2. **`ingest/reverse_analyzer.py` (`--analyze-vault`)** is read-only by construction. It only calls `list_dir`/`read_note`, never imports `VectorStore`, and never saves state or touches sources. It classifies notes using a fixed precedence order (INVALID_MANAGED_NOTE → UNMANAGED_AT_TARGET → MISSING → ANALYSIS_INSUFFICIENT_STATE → HUMAN_MODIFIED → SOURCE_MISSING → SOURCE_CHANGED → IN_SYNC) and reuses the vault_writer hash helpers (`hash_managed_body`, `parse_frontmatter`).
3. **`ingest/proposals.py` (`--propose-vault-changes`)** writes JSON files under `state/proposals/` atomically (temp file + `os.replace`). A proposal ID is a hash of its fingerprints only, which makes regeneration idempotent and changes the ID when anything drifts. Approval re-checks staleness every time. `approved` and `rejected` are terminal. Approval writes only the proposal's own file. **No apply layer exists.** Nothing may write vault content back to sources, or act on approved proposals, unless that is explicitly requested as a new phase.

Tests enforce these zero-write guarantees with in-memory fakes: a fake Ollama client, `QdrantClient(location=":memory:")`, and a fake Obsidian client. Follow the same pattern in new tests. The README documents each phase's contract in detail; keep it in sync when behavior changes.
