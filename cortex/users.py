"""Local actor identity and a small explicit authorization layer.

Not an IAM platform: users are local records, the acting user is chosen by
the caller (CLI --user / CORTEX_USER; the MCP server uses only its own
configured CORTEX_USER, never a per-call argument), and there is no cloud
auth. Unknown or malformed actor ids are refused.

Roles:      admin   every capability on every document
            member  full control of documents it owns; others per visibility/grants
            viewer  read only (visibility/grants), can never mutate
            system  internal attribution only; cannot be used as an acting user

Visibility: private  owner (and admins, and explicit grants) only
            shared   everyone may read; mutation needs ownership or a grant
            system   everyone may read; only admins mutate

Capabilities: read, write, approve, apply, delete, admin.
Unregistered content (not a Cortex document) is treated as legacy shared:
readable by all, mutable only by admins.
"""

from cortex.db import SYSTEM_USER, CortexError, NotFound, row_dict, valid_user_id

CAPABILITIES = ("read", "write", "approve", "apply", "delete", "admin")
ROLES = ("admin", "member", "viewer")
VISIBILITIES = ("private", "shared", "system")


class AuthorizationError(CortexError):
    code = "unauthorized"


def get_user(db, user_id):
    if not valid_user_id(user_id):
        raise AuthorizationError(f"invalid actor identity {user_id!r}", code="invalid_actor")
    user = row_dict(db.one("SELECT * FROM users WHERE user_id = ?", (user_id,)))
    if user is None:
        raise AuthorizationError(f"unknown actor {user_id!r}", code="invalid_actor")
    return user


def resolve_actor(db, user_id):
    """The acting user; the internal 'system' account cannot act."""
    user = get_user(db, user_id)
    if user["role"] == "system" or user_id == SYSTEM_USER:
        raise AuthorizationError("the system account cannot be used as an acting user", code="invalid_actor")
    return user


def add_user(db, actor_id, user_id, role="member", display_name=None):
    require_admin(db, actor_id)
    if not valid_user_id(user_id):
        raise CortexError(f"invalid user id {user_id!r} (lowercase letters, digits, _ . -)", code="invalid_user")
    if role not in ROLES:
        raise CortexError(f"role must be one of {ROLES}", code="invalid_role")
    with db.transaction():
        if db.one("SELECT 1 FROM users WHERE user_id = ?", (user_id,)):
            raise CortexError(f"user {user_id!r} already exists", code="exists")
        db.conn.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (user_id, display_name, role, db.now()))
    return get_user(db, user_id)


def list_users(db):
    return [row_dict(r) for r in db.all("SELECT * FROM users ORDER BY user_id")]


def require_admin(db, actor_id):
    user = resolve_actor(db, actor_id)
    if user["role"] != "admin":
        raise AuthorizationError(f"{actor_id} is not an admin")
    return user


def grants_for(db, document_id, user_id):
    return {r["capability"] for r in db.all(
        "SELECT capability FROM grants WHERE document_id = ? AND user_id = ?", (document_id, user_id))}


def can(db, actor_id, capability, document):
    """True/False. `document` is a documents row dict, or None for
    unregistered (legacy) content."""
    if capability not in CAPABILITIES:
        raise CortexError(f"unknown capability {capability!r}", code="invalid_capability")
    user = resolve_actor(db, actor_id)
    if user["role"] == "admin":
        return True
    if capability == "admin":
        return False
    if document is None:
        return capability == "read"
    if user["role"] == "viewer" and capability != "read":
        return False
    granted = grants_for(db, document["document_id"], actor_id)
    if capability == "read":
        return (document["visibility"] in ("shared", "system") or document["owner_id"] == actor_id
                or "read" in granted)
    if document["visibility"] == "system":
        return False
    return document["owner_id"] == actor_id or capability in granted


def authorize(db, actor_id, capability, document):
    if not can(db, actor_id, capability, document):
        target = document["document_id"] if document else "unregistered content"
        raise AuthorizationError(f"{actor_id} may not {capability} {target}")


def grant(db, actor_id, document_id, user_id, capability):
    doc = row_dict(db.one("SELECT * FROM documents WHERE document_id = ?", (document_id,)))
    if doc is None:
        raise NotFound(f"no document {document_id}")
    if not (doc["owner_id"] == actor_id or can(db, actor_id, "admin", doc)):
        raise AuthorizationError(f"only the owner or an admin may grant on {document_id}")
    get_user(db, user_id)
    if capability not in CAPABILITIES or capability == "admin":
        raise CortexError(f"cannot grant {capability!r}", code="invalid_capability")
    with db.transaction():
        db.conn.execute("INSERT OR IGNORE INTO grants VALUES (?, ?, ?, ?, ?)",
                        (document_id, user_id, capability, actor_id, db.now()))


def set_visibility(db, actor_id, document_id, visibility):
    doc = row_dict(db.one("SELECT * FROM documents WHERE document_id = ?", (document_id,)))
    if doc is None:
        raise NotFound(f"no document {document_id}")
    if visibility not in VISIBILITIES:
        raise CortexError(f"visibility must be one of {VISIBILITIES}", code="invalid_visibility")
    if visibility == "system":
        require_admin(db, actor_id)
    elif not (doc["owner_id"] == actor_id or can(db, actor_id, "admin", doc)):
        raise AuthorizationError(f"only the owner or an admin may change visibility of {document_id}")
    with db.transaction():
        db.conn.execute("UPDATE documents SET visibility = ?, version = version + 1, updated_at = ? "
                        "WHERE document_id = ?", (visibility, db.now(), document_id))


def readable_document_ids(db, actor_id):
    """(readable_ids, unreadable_ids) among registered documents."""
    readable, unreadable = [], []
    for row in db.all("SELECT * FROM documents"):
        (readable if can(db, actor_id, "read", dict(row)) else unreadable).append(row["document_id"])
    return readable, unreadable
