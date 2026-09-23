"""Stable logical document identity.

Every managed document gets a durable `document_id` (UUID4) that is
independent of its paths. source_path and vault_path are mutable attributes
of the document, so renames on either side keep the identity. The managed
vault note carries it as `cortex_document_id` frontmatter (source files
never carry Cortex metadata, so source renames are recognized by content
fingerprint instead; see cortex/sync_engine.py).

Documents rows are current state, updated with optimistic row versioning:
update_document(..., expected_version=v) changes the row only if its
version is still v, else ConcurrentModification.

Migration: register_from_sync_state() registers every sync_state.json
"local" entry that has a managed note (vault_write.dest_path) and no
document yet. Idempotent. The cached generated_body becomes the base
revision (origin "system"), owned by the default local user, so existing
single-user installs keep working. Entries without generated_body are
registered without a base (INSUFFICIENT_HISTORY until both sides agree).
"""

from pathlib import Path

from cortex import versions
from cortex.db import ConcurrentModification, CortexError, NotFound, row_dict

WRITABLE_TYPES = ("md", "txt")


def get_document(db, document_id, missing_ok=False):
    doc = row_dict(db.one("SELECT * FROM documents WHERE document_id = ?", (document_id,)))
    if doc is None and not missing_ok:
        raise NotFound(f"no document {document_id}")
    return doc


def find_by_source(db, source_path):
    return row_dict(db.one("SELECT * FROM documents WHERE source_path = ? AND status = 'active'", (source_path,)))


def find_by_vault(db, vault_path):
    return row_dict(db.one("SELECT * FROM documents WHERE vault_path = ? AND status = 'active'", (vault_path,)))


def list_documents(db, status=None):
    if status:
        rows = db.all("SELECT * FROM documents WHERE status = ? ORDER BY source_path, document_id", (status,))
    else:
        rows = db.all("SELECT * FROM documents ORDER BY source_path, document_id")
    return [row_dict(r) for r in rows]


def source_type_of(path):
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix if suffix in ("md", "txt", "pdf", "docx") else "unknown"


def is_reverse_writable(source_type, ai_structured):
    return source_type in WRITABLE_TYPES and not ai_structured


def insert_document(db, document_id, owner_id, source_path, vault_path, source_type, reverse_writable,
                    visibility="private", project=None, created_at=None):
    now = created_at or db.now()
    db.conn.execute(
        "INSERT INTO documents (document_id, owner_id, visibility, source_path, vault_path, source_type, "
        "reverse_writable, status, project, version, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, 1, ?, ?)",
        (document_id, owner_id, visibility, source_path, vault_path, source_type, int(bool(reverse_writable)),
         project, now, now))


def update_document(db, document_id, expected_version, **fields):
    allowed = {"source_path", "vault_path", "status", "base_revision_id", "current_revision_id", "project",
               "modified_at", "indexed_at", "reverse_writable", "source_type", "visibility"}
    unknown = set(fields) - allowed
    if unknown:
        raise CortexError(f"cannot update document fields {sorted(unknown)}")
    sets = ", ".join(f"{k} = ?" for k in sorted(fields))
    params = [fields[k] for k in sorted(fields)]
    cur = db.conn.execute(
        f"UPDATE documents SET {sets}{', ' if sets else ''}version = version + 1, updated_at = ? "
        "WHERE document_id = ? AND version = ?", params + [db.now(), document_id, expected_version])
    if cur.rowcount != 1:
        raise ConcurrentModification(f"document {document_id} changed concurrently (expected version "
                                     f"{expected_version}); nothing was overwritten")
    return get_document(db, document_id)


def register_from_sync_state(db, local_state, owner_id, source_prefix_reader=None):
    """Register managed documents found in the sync_state "local" namespace.
    Returns the list of newly registered document ids. Never writes files."""
    registered = []
    with db.transaction():
        for source_path, entry in sorted(local_state.items()):
            if not isinstance(entry, dict):
                continue
            vault_write = entry.get("vault_write")
            dest = vault_write.get("dest_path") if isinstance(vault_write, dict) else None
            if not isinstance(dest, str) or not dest:
                continue
            if find_by_source(db, source_path) or find_by_vault(db, dest):
                continue
            document_id = db.new_id()
            stype = source_type_of(source_path)
            ai = bool(entry.get("ai_structure_succeeded"))
            project = (entry.get("filter_metadata") or {}).get("project") if isinstance(
                entry.get("filter_metadata"), dict) else None
            insert_document(db, document_id, owner_id, source_path, dest, stype,
                            is_reverse_writable(stype, ai), project=project)
            body = entry.get("generated_body")
            if isinstance(body, str):
                metadata = {"source_sha256": entry.get("source_sha256"), "ai_structured": ai,
                            "migrated": True}
                if source_prefix_reader and stype == "md":
                    prefix = source_prefix_reader(source_path, entry.get("source_sha256"))
                    if prefix is not None:
                        metadata["source_prefix"] = prefix
                rev = versions.build_revision(db, document_id, body, "system", owner_id, metadata=metadata,
                                              reason="migrated from sync_state.json", source_path=source_path,
                                              vault_path=dest)
                versions.insert_revision(db, rev)
                versions.record_head(db, document_id, rev["revision_id"], rev["created_at"])
                doc = get_document(db, document_id)
                update_document(db, document_id, doc["version"], base_revision_id=rev["revision_id"],
                                current_revision_id=rev["revision_id"])
            registered.append(document_id)
    return registered
