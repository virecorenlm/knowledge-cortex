"""Managed, conflict-safe write-back from generated Markdown into the
Obsidian vault.

This is a ONE-WAY, explicitly-scoped write path: knowledge-cortex may only
create or update notes it can PROVE it owns, inside a dedicated managed
subtree. It never touches a note it cannot positively identify as its own,
and it never overwrites a cortex-owned note whose vault content has
diverged from what cortex itself last wrote there (a human edit).

Ownership marker (frontmatter, flat key: value — matches ingest/markdown.py's
existing style, no nested/list values):

    cortex_managed: true
    cortex_source_id: <12-hex-char sha256 of the absolute source path>
    cortex_source_path: <absolute path to the original local file>
    cortex_source_sha256: <sha256 of the extracted source text>
    cortex_generated_sha256: <sha256 of THIS note's managed body, i.e. the
                              content below the frontmatter, normalized>
    cortex_last_write: <ISO timestamp of this write>
    cortex_prompt_version: <ai_struct.PROMPT_VERSION, or "" if not AI-structured>
    cortex_structure_model: <model name, or "" if not AI-structured>

A managed note contains EXACTLY ONE frontmatter block -- this one. The
ingestion frontmatter that ingest/markdown.py's to_markdown() prepends
(source:/ingested:) is deliberately NOT carried into the managed note: it
would otherwise be nested as a second `---`-delimited block inside the
note's body, which is confusing and pointless. Its two fields are
subsumed by fields already present here: `source:` <-> cortex_source_path,
`ingested:` <-> cortex_last_write (the vault-write timestamp is the more
meaningful of the two for a note the user will actually read). Callers
must pass the DOCUMENT BODY ONLY (see ingest/markdown.split_frontmatter,
or ingest/sync.py's index_local_path, which already strips it and caches
the result as "document_body"/state["generated_body"]) -- this module
does not strip frontmatter itself, to keep its contract simple and
because different callers may already have the body in hand.

Conflict-detection algorithm (why this design):
    The VAULT NOTE ITSELF is the single authoritative source for conflict
    detection, not sync_state.json. Local state can be deleted, copied
    between machines, or fall out of sync; the vault is the one place a
    human editor and knowledge-cortex both actually write to, so it's the
    only reliable place to detect a human edit. Concretely: every write
    records cortex_generated_sha256 = sha256(current body). On the next
    write attempt, we read the live note, strip its frontmatter, hash the
    remaining body, and compare that hash to the RECORDED
    cortex_generated_sha256. If they still match, nothing has touched the
    note since cortex wrote it -> safe to update. If they differ, either a
    human edited it or something else changed it -> conflict, refuse to
    overwrite.

    Critically, the hash is computed ONLY over the body -- never the
    frontmatter block, and (per the note above) the body passed in here
    never contains a second, nested ingestion-frontmatter block either.
    This avoids a self-referential hashing bug: if the hash were computed
    over the whole file (frontmatter included), then cortex_last_write
    (which changes on every write) and cortex_generated_sha256 itself
    (stored inside the frontmatter it would be hashing) would make the
    "unchanged" comparison impossible to satisfy even when nothing
    meaningful changed. Concretely, the ONLY things that ever change
    cortex_generated_sha256 are changes to the actual document content
    (the AI-structured or raw body text) -- never the ingestion timestamp,
    never cortex_last_write, never which model was used.
"""

import hashlib
import re
from datetime import datetime

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n\n?", re.S)


def _parse_frontmatter(content):
    """Split a Markdown file's leading `---`-delimited frontmatter block
    (flat key: value lines only, matching this project's existing style)
    from its body. Returns (frontmatter_dict, body). No frontmatter block
    -> ({}, content unchanged)."""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    fm = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
    return fm, content[match.end():]


def _hash_body(body):
    """Stable hash of managed content, deliberately excluding frontmatter
    (see module docstring for why)."""
    return hashlib.sha256(body.strip().encode("utf-8")).hexdigest()


def _build_note(body, metadata):
    frontmatter = "\n".join(f"{k}: {v}" for k, v in metadata.items())
    return f"---\n{frontmatter}\n---\n\n{body.strip()}\n"


def source_id_for(abs_source_path):
    """Stable, content-independent identifier for a source file, usable
    even if the destination note is later renamed."""
    return hashlib.sha256(abs_source_path.encode("utf-8")).hexdigest()[:12]


async def _note_exists(obsidian, path):
    """True if `path` exists in the vault. Uses list_dir on the parent
    directory rather than a try/except around read_note: the MCP client
    wraps arbitrary server-side error text in a generic RuntimeError, so
    string-matching for "not found" would be fragile and could silently
    misclassify a real network/auth failure as "file doesn't exist" --
    which would then cause an unsafe blind create over an existing note.
    list_dir's contract (a plain array of names) is unambiguous."""
    directory, _, name = path.rpartition("/")
    entries = await obsidian.list_dir(directory)
    return name in entries


# Public aliases for the private helpers above, for reuse by other modules
# needing the EXACT same frontmatter/hash contract without duplicating it
# (e.g. ingest/reverse_analyzer.py, which must classify a note's state
# using the identical body-hash and frontmatter-parsing rules write-back
# uses, rather than a second, potentially-drifting implementation).
parse_frontmatter = _parse_frontmatter
hash_managed_body = _hash_body
note_exists = _note_exists


async def write_managed_note(obsidian, dest_path, generated_body, metadata):
    """Attempt a managed, conflict-safe write of `generated_body` to
    `dest_path` in the vault.

    metadata: dict with keys source_path, source_sha256, prompt_version
    (str or None), structure_model (str or None) -- used to build the
    frontmatter. cortex_source_id/cortex_generated_sha256/cortex_last_write
    are always computed by this function, not passed in.

    Returns a dict:
        {"status": "created" | "updated" | "skipped" | "conflict" | "error",
         "path": dest_path,
         "reason": str | None,
         "generated_sha256": str}   # hash of the body just written/checked,
                                     # useful for the caller's local state cache

    Never raises for ordinary conflict/skip cases; only lets through
    genuine unexpected exceptions from the Obsidian client (network/auth
    failures), which the caller is expected to catch and report exactly
    like any other per-file error (see ingest/sync.py's pattern).

    Missing-note semantics (a previously-created managed note that no
    longer exists at dest_path, e.g. deleted by a human or from outside
    cortex): this is treated identically to Case A (note never existed) --
    it is recreated. Rationale: dest_path is itself a cortex-owned,
    deterministic identifier (derived from source_id_for(source_path));
    if --write-vault is being explicitly re-run for that source, an
    absent note at cortex's own designated path cannot be a human's
    content that needs protecting (there is nothing there to protect),
    so recreating it is the same "safe create" as the first-ever run,
    not an unsafe overwrite. This does NOT reach into deletion tracking
    or propagate deletes in the other direction -- it only means a
    missing managed note doesn't permanently block future write-back for
    that source.
    """
    generated_hash = _hash_body(generated_body)
    frontmatter = {
        "cortex_managed": "true",
        "cortex_source_id": source_id_for(metadata["source_path"]),
        "cortex_source_path": metadata["source_path"],
        "cortex_source_sha256": metadata["source_sha256"],
        "cortex_generated_sha256": generated_hash,
        "cortex_last_write": datetime.now().isoformat(),
        "cortex_prompt_version": metadata.get("prompt_version") or "",
        "cortex_structure_model": metadata.get("structure_model") or "",
    }

    exists = await _note_exists(obsidian, dest_path)

    if not exists:
        await obsidian.write_note(dest_path, _build_note(generated_body, frontmatter))
        return {"status": "created", "path": dest_path, "reason": None,
                "generated_sha256": generated_hash}

    current = await obsidian.read_note(dest_path)
    current_fm, current_body = _parse_frontmatter(current["content"])

    if current_fm.get("cortex_managed") != "true":
        return {"status": "conflict", "path": dest_path,
                "reason": "note exists and is not cortex-managed; refusing to overwrite",
                "generated_sha256": generated_hash}

    recorded_hash = current_fm.get("cortex_generated_sha256")
    live_hash = _hash_body(current_body)
    if recorded_hash != live_hash:
        return {"status": "conflict", "path": dest_path,
                "reason": "vault note content changed since knowledge-cortex last wrote it "
                          "(likely a human edit); refusing to overwrite",
                "generated_sha256": generated_hash}

    if generated_hash == recorded_hash:
        return {"status": "skipped", "path": dest_path,
                "reason": "generated content unchanged", "generated_sha256": generated_hash}

    await obsidian.write_note(dest_path, _build_note(generated_body, frontmatter))
    return {"status": "updated", "path": dest_path, "reason": None,
            "generated_sha256": generated_hash}
