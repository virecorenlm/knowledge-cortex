"""Append-only transaction journal for mutating operations.

Every mutating operation writes, in order:
  1. a `begin` event (its own committed transaction) holding the full plan:
     preconditions, before/intended-after hashes of every file it will
     touch, backup paths, and the exact DB changes (revision records with
     precomputed ids, document update, heads, sync_state changes)
  2. the file effects (backups first, then atomic writes)
  3. a `commit` event, in the SAME SQLite transaction as the plan's DB
     changes -- or a `fail` event if a step failed (after compensation)

Because the plan is fully precomputed in the begin event, an operation
interrupted between 2 and 3 can be finished or rolled back later by
cortex.sync_engine.SyncEngine.recover(): if every file is at its intended
after-state the plan is replayed (`recover`), if every file is still at its
before-state nothing happened (`rollback`), and anything else is restored
from the recorded backups or left for a human (`review`). Events are never
updated or deleted (SQLite triggers).
"""

import json

from cortex.db import canonical_json

TERMINAL = ("commit", "fail", "recover", "rollback", "review")


class Journal:
    def __init__(self, db):
        self.db = db

    def _insert(self, op_id, document_id, op_type, phase, actor_id, data):
        self.db.conn.execute(
            "INSERT INTO events (event_id, op_id, document_id, op_type, phase, actor_id, created_at, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (self.db.new_id(), op_id, document_id, op_type, phase, actor_id, self.db.now(), canonical_json(data)))

    def begin(self, op_type, document_id, actor_id, data):
        op_id = self.db.new_id()
        with self.db.transaction():
            self._insert(op_id, document_id, op_type, "begin", actor_id, data)
        return op_id

    def finish(self, op_id, phase, actor_id, data, db_changes=None):
        """Terminal event; db_changes (callable) runs in the same transaction."""
        if phase not in TERMINAL:
            raise ValueError(f"not a terminal phase: {phase}")
        begin = self.begin_event(op_id)
        with self.db.transaction():
            if self.terminal_event(op_id) is not None:
                raise RuntimeError(f"operation {op_id} is already finished")
            if db_changes:
                db_changes()
            self._insert(op_id, begin["document_id"], begin["op_type"], phase, actor_id, data)

    def begin_event(self, op_id):
        row = self.db.one("SELECT * FROM events WHERE op_id = ? AND phase = 'begin'", (op_id,))
        return _event(row)

    def terminal_event(self, op_id):
        row = self.db.one("SELECT * FROM events WHERE op_id = ? AND phase != 'begin' ORDER BY seq DESC LIMIT 1",
                          (op_id,))
        return _event(row)

    def pending(self):
        rows = self.db.all(
            "SELECT * FROM events b WHERE b.phase = 'begin' AND NOT EXISTS "
            "(SELECT 1 FROM events t WHERE t.op_id = b.op_id AND t.phase != 'begin') ORDER BY b.seq")
        return [_event(r) for r in rows]

    def events(self, document_id=None, limit=None):
        sql, params = "SELECT * FROM events", []
        if document_id:
            sql += " WHERE document_id = ?"
            params.append(document_id)
        sql += " ORDER BY seq"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [_event(r) for r in self.db.all(sql, params)]

    def verify(self):
        """Every op has exactly one begin, at most one terminal event, and the
        terminal event comes after the begin."""
        problems = []
        ops = {}
        for row in self.db.all("SELECT op_id, phase, seq FROM events ORDER BY seq"):
            ops.setdefault(row["op_id"], []).append((row["phase"], row["seq"]))
        for op_id, phases in ops.items():
            begins = [s for p, s in phases if p == "begin"]
            terminals = [s for p, s in phases if p != "begin"]
            if len(begins) != 1:
                problems.append(f"operation {op_id} has {len(begins)} begin events")
            if len(terminals) > 1:
                problems.append(f"operation {op_id} has {len(terminals)} terminal events")
            if begins and terminals and terminals[0] < begins[0]:
                problems.append(f"operation {op_id} finished before it began")
        return problems


def _event(row):
    if row is None:
        return None
    data = dict(row)
    data["data"] = json.loads(data["data"])
    return data
