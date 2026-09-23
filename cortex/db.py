"""The Cortex store: one local SQLite file (stdlib sqlite3, no server).

Why SQLite rather than more JSON files: revisions, journal events,
conflicts, tombstones, syntheses and users need multi-record atomic
transactions (a sync operation commits a revision, a document update and a
journal event together), indexed lookups (history by document, "current at
time T"), and enforced immutability. SQLite gives all three in one file; it
is not a new service and Qdrant remains the vector layer.

Current state vs history (rule: "current and historical state are distinct"):
  current state (mutable, row-versioned):  documents, grants, tombstones,
                                            conflicts, syntheses (status only),
                                            contradictions (status only), users
  history (append-only, triggers reject UPDATE/DELETE):
                                            revisions, events (journal),
                                            heads (which revision was current when)

Opening a CortexDB creates/migrates the schema; nothing happens on import.
"""

import fcntl
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "state" / "cortex.db"
DEFAULT_USER = "local"
SYSTEM_USER = "system"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT,
    role TEXT NOT NULL CHECK (role IN ('admin', 'member', 'viewer', 'system')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES users(user_id),
    visibility TEXT NOT NULL CHECK (visibility IN ('private', 'shared', 'system')),
    source_path TEXT,
    vault_path TEXT,
    source_type TEXT,
    reverse_writable INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('active', 'tombstoned')),
    base_revision_id TEXT,
    current_revision_id TEXT,
    project TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    modified_at TEXT,
    indexed_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS documents_active_source ON documents(source_path) WHERE status = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS documents_active_vault ON documents(vault_path) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS grants (
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    user_id TEXT NOT NULL REFERENCES users(user_id),
    capability TEXT NOT NULL,
    granted_by TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    PRIMARY KEY (document_id, user_id, capability)
);

CREATE TABLE IF NOT EXISTS revisions (
    revision_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    parent_ids TEXT NOT NULL,
    created_at TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('source', 'vault', 'merge', 'edit', 'restore', 'resolution', 'system')),
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL,
    source_path TEXT,
    vault_path TEXT,
    reason TEXT,
    provenance TEXT NOT NULL,
    valid_from TEXT,
    valid_to TEXT
);
CREATE INDEX IF NOT EXISTS revisions_doc_time ON revisions(document_id, created_at);
CREATE INDEX IF NOT EXISTS revisions_time ON revisions(created_at);
CREATE TRIGGER IF NOT EXISTS revisions_immutable_u BEFORE UPDATE ON revisions
    BEGIN SELECT RAISE(ABORT, 'revisions are immutable'); END;
CREATE TRIGGER IF NOT EXISTS revisions_immutable_d BEFORE DELETE ON revisions
    BEGIN SELECT RAISE(ABORT, 'revisions are immutable'); END;

CREATE TABLE IF NOT EXISTS heads (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    revision_id TEXT,
    became_current_at TEXT NOT NULL,
    op_id TEXT
);
CREATE INDEX IF NOT EXISTS heads_doc_time ON heads(document_id, became_current_at);
CREATE TRIGGER IF NOT EXISTS heads_immutable_u BEFORE UPDATE ON heads
    BEGIN SELECT RAISE(ABORT, 'head history is immutable'); END;
CREATE TRIGGER IF NOT EXISTS heads_immutable_d BEFORE DELETE ON heads
    BEGIN SELECT RAISE(ABORT, 'head history is immutable'); END;

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    op_id TEXT NOT NULL,
    document_id TEXT,
    op_type TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('begin', 'commit', 'fail', 'recover', 'rollback', 'review')),
    actor_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_op ON events(op_id);
CREATE INDEX IF NOT EXISTS events_doc ON events(document_id, seq);
CREATE TRIGGER IF NOT EXISTS events_immutable_u BEFORE UPDATE ON events
    BEGIN SELECT RAISE(ABORT, 'journal events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS events_immutable_d BEFORE DELETE ON events
    BEGIN SELECT RAISE(ABORT, 'journal events are immutable'); END;

CREATE TABLE IF NOT EXISTS tombstones (
    tombstone_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    deleted_at TEXT NOT NULL,
    deleted_by TEXT NOT NULL,
    deleted_from TEXT NOT NULL CHECK (deleted_from IN ('source', 'vault', 'both')),
    last_source_hash TEXT,
    last_vault_hash TEXT,
    status TEXT NOT NULL CHECK (status IN ('detected', 'propagated', 'restored', 'purged')),
    trash TEXT NOT NULL,
    saved_state TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tombstones_doc ON tombstones(document_id);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    kind TEXT NOT NULL CHECK (kind IN ('sync', 'concurrent_edit')),
    base_revision_id TEXT,
    left_revision_id TEXT NOT NULL,
    right_revision_id TEXT NOT NULL,
    left_label TEXT NOT NULL,
    right_label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'resolved', 'superseded')),
    regions TEXT NOT NULL,
    observed TEXT NOT NULL,
    suggestion TEXT,
    resolution TEXT,
    resolved_at TEXT,
    resolved_by TEXT,
    resolution_revision_id TEXT,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS conflicts_doc ON conflicts(document_id, status);
CREATE TRIGGER IF NOT EXISTS conflicts_final BEFORE UPDATE ON conflicts WHEN OLD.status != 'open'
    BEGIN SELECT RAISE(ABORT, 'resolved/superseded conflicts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS conflicts_no_delete BEFORE DELETE ON conflicts
    BEGIN SELECT RAISE(ABORT, 'conflicts are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS conflicts_frozen_fields BEFORE UPDATE ON conflicts
    WHEN NEW.document_id IS NOT OLD.document_id OR NEW.base_revision_id IS NOT OLD.base_revision_id
      OR NEW.left_revision_id IS NOT OLD.left_revision_id OR NEW.right_revision_id IS NOT OLD.right_revision_id
      OR NEW.regions IS NOT OLD.regions OR NEW.observed IS NOT OLD.observed OR NEW.created_at IS NOT OLD.created_at
    BEGIN SELECT RAISE(ABORT, 'conflict evidence is immutable'); END;

CREATE TABLE IF NOT EXISTS syntheses (
    synthesis_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    query TEXT NOT NULL,
    params TEXT NOT NULL,
    created_at TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    prompt_version INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    evidence_fingerprint TEXT NOT NULL,
    items TEXT NOT NULL,
    body TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('current', 'stale', 'superseded')),
    status_reason TEXT,
    superseded_by TEXT,
    updated_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS syntheses_frozen BEFORE UPDATE ON syntheses
    WHEN NEW.type IS NOT OLD.type OR NEW.query IS NOT OLD.query OR NEW.params IS NOT OLD.params
      OR NEW.created_at IS NOT OLD.created_at OR NEW.actor_id IS NOT OLD.actor_id OR NEW.model IS NOT OLD.model
      OR NEW.provider IS NOT OLD.provider OR NEW.prompt_version IS NOT OLD.prompt_version
      OR NEW.evidence IS NOT OLD.evidence OR NEW.evidence_fingerprint IS NOT OLD.evidence_fingerprint
      OR NEW.items IS NOT OLD.items OR NEW.body IS NOT OLD.body OR NEW.content_hash IS NOT OLD.content_hash
    BEGIN SELECT RAISE(ABORT, 'synthesis content and provenance are immutable'); END;
CREATE TRIGGER IF NOT EXISTS syntheses_no_delete BEFORE DELETE ON syntheses
    BEGIN SELECT RAISE(ABORT, 'syntheses are never deleted'); END;

CREATE TABLE IF NOT EXISTS contradictions (
    contradiction_id TEXT PRIMARY KEY,
    synthesis_id TEXT NOT NULL REFERENCES syntheses(synthesis_id),
    claim_a TEXT NOT NULL,
    claim_b TEXT NOT NULL,
    evidence_a TEXT NOT NULL,
    evidence_b TEXT NOT NULL,
    time_context TEXT NOT NULL,
    source_context TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('unresolved', 'resolved', 'superseded', 'time_dependent', 'not_conflicting')),
    status_basis TEXT NOT NULL,
    resolved_by TEXT,
    resolved_at TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS contradictions_frozen BEFORE UPDATE ON contradictions
    WHEN NEW.claim_a IS NOT OLD.claim_a OR NEW.claim_b IS NOT OLD.claim_b OR NEW.evidence_a IS NOT OLD.evidence_a
      OR NEW.evidence_b IS NOT OLD.evidence_b OR NEW.synthesis_id IS NOT OLD.synthesis_id
    BEGIN SELECT RAISE(ABORT, 'contradiction claims and evidence are immutable'); END;
"""

_USER_ID_RE = re.compile(r"\A[a-z0-9][a-z0-9_.-]{0,63}\Z")


class CortexError(Exception):
    """Base error with a stable machine-readable code."""

    code = "error"

    def __init__(self, message, code=None):
        super().__init__(message)
        if code:
            self.code = code


class NotFound(CortexError):
    code = "not_found"


class ConcurrentModification(CortexError):
    code = "concurrent_modification"


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def to_bound(value, end_of_day=False):
    """ISO date/datetime -> store timestamp string. Date-only + end_of_day =
    the last microsecond of that day."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise CortexError(f"invalid time {value!r}", code="invalid_time")
    text = value.strip()
    try:
        if len(text) == 10:
            day = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
            dt = day + timedelta(days=1) - timedelta(microseconds=1) if end_of_day else day
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except ValueError as exc:
        raise CortexError(f"invalid time {value!r}", code="invalid_time") from exc
    return dt.strftime(_FMT)


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def valid_user_id(user_id):
    return isinstance(user_id, str) and bool(_USER_ID_RE.match(user_id))


class CortexDB:
    """Connection + schema + clock. `clock` is injectable so tests (and
    history replays) are deterministic."""

    def __init__(self, path=None, clock=None, default_user=DEFAULT_USER):
        self.path = Path(path or os.getenv("CORTEX_DB") or DEFAULT_DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or utcnow
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._depth = 0
        self._migrate(default_user)

    def close(self):
        self.conn.close()

    def now(self):
        return self.clock()

    def new_id(self):
        return str(uuid.uuid4())

    def _migrate(self, default_user):
        # executescript commits implicitly, so the idempotent DDL (all
        # IF NOT EXISTS) runs as its own atomic script before the data step.
        self.conn.executescript("BEGIN IMMEDIATE;" + _SCHEMA + "COMMIT;")
        with self.transaction():
            row =self.conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                self.conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row["value"]) > SCHEMA_VERSION:
                raise CortexError(f"{self.path} has schema version {row['value']}, newer than this build "
                                  f"({SCHEMA_VERSION})", code="unsupported_schema")
            now = self.now()
            # The default local user owns migrated single-user data and keeps
            # existing single-user installs working unchanged (admin).
            self.conn.execute("INSERT OR IGNORE INTO users VALUES (?, ?, 'admin', ?)",
                              (default_user, "Local user", now))
            self.conn.execute("INSERT OR IGNORE INTO users VALUES (?, ?, 'system', ?)",
                              (SYSTEM_USER, "Cortex system", now))

    @contextmanager
    def transaction(self):
        """Nestable IMMEDIATE transaction (savepoints when nested)."""
        if self._depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        else:
            self.conn.execute(f"SAVEPOINT sp{self._depth}")
        self._depth += 1
        try:
            yield self.conn
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("ROLLBACK")
            else:
                self.conn.execute(f"ROLLBACK TO sp{self._depth}")
                self.conn.execute(f"RELEASE sp{self._depth}")
            raise
        self._depth -= 1
        if self._depth == 0:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute(f"RELEASE sp{self._depth}")

    @contextmanager
    def lock(self):
        """Process-level exclusive lock for mutating sync operations, so two
        concurrent `sync` runs cannot interleave file writes."""
        lock_path = self.path.with_suffix(".lock")
        with open(lock_path, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def one(self, sql, params=()):
        return self.conn.execute(sql, params).fetchone()

    def all(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()


def row_dict(row, json_fields=()):
    if row is None:
        return None
    data = dict(row)
    for key in json_fields:
        if data.get(key) is not None:
            data[key] = json.loads(data[key])
    return data
