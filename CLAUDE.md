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
python main.py --apply-proposal ID [--dry-run] [--backup-dir DIR]
python main.py --search "QUERY" [--filter JSON] [--prefer JSON] [--tag T] [--max-per-source N] [--json]
python main.py --create-payload-indexes                # explicit, idempotent
python cortex_cli.py [--user U] <command>              # Knowledge Cortex: sync, history, temporal, synthesis, users, conflicts
python cortex_cli.py sync --dry-run | sync | recover | verify | history DOC | temporal-search Q --as-of D | synthesize Q --type T
python mcp_server.py [--http --port 8770]              # semantic_search / hybrid_search / sync_vault / index_status + Cortex tools
```

Cortex config: `CORTEX_DB` (default `state/cortex.db`), `CORTEX_USER` (default `local`), `CORTEX_VAULT_DIR` (local vault dir, which enables rename/delete propagation; unset means the MCP client, where moves need `OBSIDIAN_ALLOW_MOVES=1` because Obsidian's `vault_move` rewrites links in other notes; `vault_delete` defaults to Obsidian's trash), `CORTEX_MANAGED_DIR`, `CORTEX_SOURCE_ROOTS`, `CORTEX_SYNTHESIS_MODEL`. Runtime config is env-only: `OLLAMA_URL`, `EMBEDDING_MODEL` (default `qwen3-embedding:4b`), `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY`, `OBSIDIAN_MCP_URL`, `OBSIDIAN_MCP_AUTHORIZATION`, `AI_STRUCTURE_MODEL`, `AI_STRUCTURE_TIMEOUT_SECONDS`. `config.yaml` is empty/unused.

## Architecture

Pipeline: source (Obsidian vault over MCP, or local PDF/DOCX/TXT/MD) → extract → chunk → embed via Ollama → upsert into Qdrant. Both vault sync and local ingestion share the same `graph/store.py:VectorStore` and `ingest/chunk.py`.

- **`graph/store.py` `VectorStore`** — the only Ollama+Qdrant layer. Reads env at construction, and accepts injected `ollama_client`/`qdrant_client`. `ensure_collection` refuses a vector-size mismatch; use a new collection when the embedding model changes. (`graph/autolink.py` is legacy and not imported anywhere.)
- **`graph/retrieval.py` `hybrid_search`** owns retrieval policy; `VectorStore` keeps all Qdrant access. Hard filters compile to native Qdrant filters over the audited fields only (`source`, `project`, `tags`, `path`, `source_file`, `ai_structured` where missing counts as false, and `doc_date`). Soft preferences give `final = semantic + min(Σ weight·match, max_total_boost)`. All tuning values live in `RankingConfig`; don't scatter constants. It is read-only and must never create indexes or collections. `store.search` and MCP `semantic_search` stay backward compatible.
- **`ingest/chunk.py`** — chunk IDs are derived from `(path, sha256, chunk_index)`. Re-indexing a document deletes its old chunks by path and then upserts, so unchanged content is a no-op and nothing is left orphaned.
- **`sync_cli.py`** owns the `sync_state.json` format: `load_state/save_state(path, namespace=...)`. There are two namespaces: `"vault"` (vault sync, `{path: sha256}`) and `"local"` (local ingestion; richer per-file records including cached `generated_body` and `vault_write`). The legacy flat format is migrated into `"vault"`. `main.py` and `mcp_server.py` import these helpers from `sync_cli`.
- **`ingest/sync.py`** — `sync_vault_to_index` (async, over `ingest/obsidian_client.py`) and `index_local_path` (local files). Callers handle persistence. An error in one file is recorded and skipped without aborting the batch, and that file's state is not updated, so it is retried next run.
- **`main.py`** — argparse CLI whose modes are thin functions (`extract_to_markdown`, `index_to_qdrant`, `analyze_vault`, `propose_vault_changes`, `_decision_cli`, `list_proposals_cli`, `show_proposal_cli`). Tests call these functions directly with fakes; only parser-validation tests run it as a subprocess.
- **`ingest/metadata.py`** derives the filterable payload fields `project`, `tags` and `doc_date` for every chunk. `graph/store.build_search_filter` turns a plain `filters` dict into a Qdrant `Filter` for `VectorStore.search` and MCP `semantic_search`. For local files, `index_local_path` caches `filter_metadata` in state. When that metadata changes on an unchanged file, it updates the existing points with `store.set_metadata` (`set_payload`), with no re-embedding.
- **`ingest/markdown.document_body`** defines the body contract. Native `.md` is verbatim minus its own frontmatter (which is source metadata that feeds filter metadata, never body). `.txt` is verbatim. PDF/DOCX go through `normalize()`. The `.md`/`.txt` round trip (body → source → body) must stay stable, because apply and post-apply reconciliation depend on it; bump `sync.BODY_CONTRACT_VERSION` if the contract changes. Re-ingestion keeps `vault_write.dest_path` and resets the stale write outcome to `unverified`.
- **`ingest/ai_struct.py`** — optional Ollama chat restructuring. It is fail-safe: any error, timeout, or validation rejection falls back to the raw Markdown. Only set `ai_structured: true` in metadata when the indexed text actually is the structured version.

### Obsidian write-back safety model (the core invariants)

These layers are deliberately staged, and each one does only what its phase allows. Preserve these guarantees when changing them:

1. **`ingest/vault_writer.py` (`--write-vault`)** only writes under `Knowledge Cortex/Managed/` as `<basename>-<source_id>.md`. It only overwrites a note that has `cortex_managed: true` and whose current body hash equals the recorded `cortex_generated_sha256`. The hash covers the **body only, never the frontmatter**. The live vault is the authority for conflict detection; `sync_state.json` is not. Human edits always win and are reported as conflicts. The one exception to "edited → conflict" is a live body that already equals the body being written: it is re-baselined (frontmatter only, body untouched, and only when the frontmatter is purely cortex keys). Each note has exactly one flat frontmatter block, and `markdown.split_frontmatter` strips ingestion frontmatter first.
2. **`ingest/reverse_analyzer.py` (`--analyze-vault`)** is read-only by construction. It only calls `list_dir`/`read_note`, never imports `VectorStore`, and never saves state or touches sources. It classifies notes using a fixed precedence order (INVALID_MANAGED_NOTE → UNMANAGED_AT_TARGET → MISSING → ANALYSIS_INSUFFICIENT_STATE → HUMAN_MODIFIED → SOURCE_MISSING → SOURCE_CHANGED → IN_SYNC) and reuses the vault_writer hash helpers (`hash_managed_body`, `parse_frontmatter`). A note whose body still matches its own `cortex_generated_sha256` is never HUMAN_MODIFIED; if it differs from the current generated body, that is SOURCE_CHANGED.
3. **`ingest/proposals.py` (`--propose-vault-changes`)** writes JSON files under `state/proposals/` atomically (temp file + `os.replace`). A proposal ID is a hash of its fingerprints only, which makes regeneration idempotent and changes the ID when anything drifts. Approval re-checks staleness every time. `approved` and `rejected` are terminal. Approval writes only the proposal's own file.
4. **`ingest/apply.py` (`--apply-proposal [--dry-run]`)** is the only code that writes a source file. It only applies `approved` + `HUMAN_MODIFIED`/`review_human_changes`, non-AI-structured (deliberately; keep it that way) proposals to `.md`/`.txt` sources whose candidate round-trips through `document_body`, and writes exactly `reviewed_live_vault_body` (keeping `.md` source frontmatter verbatim, never Cortex frontmatter). It re-runs `proposals.detect_drift` right before writing (drift → `stale`, no write), backs up the source under `state/apply_backups/` (never overwrites), replaces the source atomically, verifies the result, then sets `applied`. A partial failure is recovered through `apply_intent` without a second write. It never writes the vault, Qdrant, or `sync_state.json`. Other classifications, merging, and rename/delete propagation remain unimplemented; do not add them unless explicitly requested as a new phase.

### Knowledge Cortex (`cortex/`, `cortex_cli.py`) invariants

- **`cortex/db.py`** is one SQLite store. Current state (documents, grants, tombstones, conflicts, synthesis status) is mutable with row versions. History (`revisions`, `heads`, `events`) is append-only and trigger-enforced. Never add UPDATE/DELETE paths for history, and never open the DB on import.
- **Identity:** `document_id` (UUID) is stable. Paths are attributes. The note carries `cortex_document_id`. Sources never carry Cortex metadata, so source renames are detected by content fingerprint and only when unambiguous.
- **`cortex/sync_engine.py`** classifies before acting, and every write has an exact-content precondition. Vault-only edits go through proposals/apply, never directly. Only a clean `cortex/merge.py` diff3 merge is automatic; overlap or adjacency means a conflict record and no writes. Deletes become tombstones, then approval, then recovery areas; purge is admin-only. Moves need `vault.supports_moves` and fail closed otherwise. The vault recovery area is `_cortex-trash` (Obsidian hides dot folders).
- **Journal:** `_run_op` writes `begin` with the full plan (every after-hash precomputed) before any file I/O, and `commit` in the same transaction as the plan's DB changes. `recover()` replays or rolls back. Keep that ordering whenever you add an operation.
- **Authorization:** `cortex/users.py` enforces it everywhere. A source rewrite needs `apply` in addition to `write`. MCP tools take no user argument (the actor is `CORTEX_USER`). Retrieval visibility is enforced via `hybrid_search(exclude_document_ids=...)`.
- **Temporal:** transaction time comes from `heads`, valid time only from explicit frontmatter (never guessed). `<collection>__history` is a derived index.
- **Synthesis:** every stored statement cites evidence refs. Content and provenance are immutable, and only the status changes (current/stale/superseded). Synthesis never writes sources, notes or the evidence index. AI conflict suggestions are never applied except via an explicit `accept_suggestion`.

Tests enforce these zero-write guarantees with in-memory fakes: a fake Ollama client, `QdrantClient(location=":memory:")`, and a fake Obsidian client. Follow the same pattern in new tests. The README documents each phase's contract in detail; keep it in sync when behavior changes.
