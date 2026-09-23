"""CortexService: one place that wires the Cortex store, sync engine, vault
backend, vector store and synthesis model from configuration, for the CLI
(cortex_cli.py) and the MCP server. Nothing is constructed on import.

Configuration (environment, overridable per call):
    CORTEX_DB         Cortex SQLite store (default state/cortex.db)
    CORTEX_USER       acting user (default "local", the migrated admin)
    CORTEX_VAULT_DIR  local Obsidian vault directory -> FilesystemVault (supports
                      rename/delete propagation); unset -> ObsidianClient (MCP)
    CORTEX_MANAGED_DIR managed subtree (default "Knowledge Cortex/Managed")
    CORTEX_SOURCE_ROOTS  os.pathsep-separated roots for source rename detection
                      and restores (default: parents of tracked sources)
plus the existing OLLAMA_URL / EMBEDDING_MODEL / QDRANT_* / OBSIDIAN_MCP_*.
"""

import os
from pathlib import Path

from cortex.db import DEFAULT_USER, CortexDB


class CortexService:
    def __init__(self, db_path=None, state_path=None, vault=None, vault_dir=None, managed_dir=None,
                 proposals_dir=None, store=None, use_store=True, source_roots=None, model=None, clock=None,
                 user=None):
        from ingest.proposals import DEFAULT_PROPOSALS_DIR, DEFAULT_STATE_PATH
        self.db = CortexDB(db_path, clock=clock)
        self.user = user or os.getenv("CORTEX_USER") or DEFAULT_USER
        self.state_path = Path(state_path or DEFAULT_STATE_PATH)
        self.proposals_dir = Path(proposals_dir or DEFAULT_PROPOSALS_DIR)
        self.managed_dir = managed_dir or os.getenv("CORTEX_MANAGED_DIR") or "Knowledge Cortex/Managed"
        self._vault, self._vault_dir = vault, vault_dir or os.getenv("CORTEX_VAULT_DIR")
        self._store, self._use_store = store, use_store
        roots = source_roots or ([r for r in os.getenv("CORTEX_SOURCE_ROOTS", "").split(os.pathsep) if r] or None)
        self._roots = roots
        self._model = model
        self._engine = None

    @property
    def vault(self):
        if self._vault is None:
            if self._vault_dir:
                from cortex.vault_fs import FilesystemVault
                self._vault = FilesystemVault(self._vault_dir)
            else:
                from ingest.obsidian_client import ObsidianClient
                self._vault = ObsidianClient()
        return self._vault

    @property
    def store(self):
        if self._store is None and self._use_store:
            from graph.store import VectorStore
            self._store = VectorStore()
        return self._store

    @property
    def model(self):
        if self._model is None:
            from cortex.synthesis import OllamaChatModel
            self._model = OllamaChatModel()
        return self._model

    @property
    def engine(self):
        if self._engine is None:
            from cortex.sync_engine import SyncEngine
            self._engine = SyncEngine(self.db, self.vault, self.state_path, managed_dir=self.managed_dir,
                                      proposals_dir=self.proposals_dir, store=self.store,
                                      source_roots=self._roots)
        return self._engine

    def close(self):
        self.db.close()
