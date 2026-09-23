"""Knowledge Cortex CLI: identity, bidirectional sync, history, temporal
search, synthesis, users and conflicts. Output is JSON. Exit code 0 on
success, 1 on a refusal/error (the JSON then has "error" and "code").

    python cortex_cli.py [--user U] <command> ...

Sync         migrate | sync [--dry-run] [--document ID] | sync-status | recover | verify | journal
Deletions    tombstones | approve-tombstone ID | restore-tombstone ID | purge-tombstone ID
History      documents | history DOC | show-revision REV | diff-revisions A B | restore-revision REV
Edits        edit DOC --file F --expected-revision REV
Conflicts    conflicts | show-conflict ID | resolve-conflict ID --action A [--body-file F]
             | suggest-resolution ID
Temporal     temporal-search QUERY [--as-of D] [--changed-after D] [--changed-before D] [--valid-at D]
             | changes [--after D] [--before D] [--project P] | reindex-history
Synthesis    synthesize QUERY --type T | list-syntheses | show-synthesis ID | check-synthesis ID
             | resynthesize ID | resolve-contradiction ID --status S
Users        users | add-user ID [--role R] | grant DOC USER CAP | set-visibility DOC VIS
See README "Cortex architecture" for semantics and safety rules.
"""

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

from cortex.db import CortexError


def _parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", help="acting user (default CORTEX_USER or 'local')")
    p.add_argument("--cortex-db", dest="cortex_db", help="Cortex store (default CORTEX_DB or state/cortex.db)")
    p.add_argument("--state", help="sync_state.json path")
    p.add_argument("--vault-dir", dest="vault_dir", help="local vault directory (enables moves)")
    p.add_argument("--managed-dir", dest="managed_dir", help="managed vault subtree")
    p.add_argument("--proposals-dir", dest="proposals_dir")
    p.add_argument("--source-root", dest="source_roots", action="append", help="source root (repeatable)")
    p.add_argument("--no-index", dest="no_index", action="store_true", help="do not touch Qdrant")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, *args, **flags):
        sp = sub.add_parser(name)
        for a in args:
            sp.add_argument(a)
        for flag, kw in flags.items():
            sp.add_argument("--" + flag.replace("_", "-"), dest=flag, **kw)
        return sp

    add("migrate")
    add("sync", dry_run={"action": "store_true"}, document={"action": "append"})
    add("sync-status", document={"action": "append"})
    add("recover")
    add("verify")
    add("journal", document={}, limit={"type": int, "default": 200})
    add("tombstones")
    for name in ("approve-tombstone", "restore-tombstone", "purge-tombstone"):
        add(name, "tombstone_id")
    add("documents")
    add("history", "document_id")
    add("show-revision", "revision_id")
    add("diff-revisions", "a", "b")
    add("restore-revision", "revision_id", reason={})
    add("edit", "document_id", file={"required": True}, expected_revision={"required": True}, reason={})
    add("conflicts", status={})
    add("show-conflict", "conflict_id")
    add("resolve-conflict", "conflict_id", action={"required": True}, body_file={}, note={})
    add("suggest-resolution", "conflict_id")
    temporal = dict(as_of={}, changed_after={}, changed_before={}, valid_at={}, filter={}, prefer={},
                    limit={"type": int, "default": 10})
    add("temporal-search", "query", **temporal)
    add("changes", after={}, before={}, project={})
    add("reindex-history")
    add("synthesize", "query", type={"default": "SUMMARY"}, **{k: v for k, v in temporal.items() if k != "valid_at"})
    add("list-syntheses", status={})
    add("show-synthesis", "synthesis_id")
    add("check-synthesis", "synthesis_id")
    add("resynthesize", "synthesis_id")
    add("resolve-contradiction", "contradiction_id", status={"required": True}, note={})
    add("users")
    add("add-user", "user_id", role={"default": "member"}, display_name={})
    add("grant", "document_id", "grantee", "capability")
    add("set-visibility", "document_id", "visibility")
    return p


def _json_arg(value, name):
    if not value:
        return None
    try:
        data = json.loads(value)
    except ValueError as exc:
        raise CortexError(f"--{name} must be a JSON object: {exc}", code="invalid_argument") from exc
    if not isinstance(data, dict):
        raise CortexError(f"--{name} must be a JSON object", code="invalid_argument")
    return data


def execute(args, svc):
    from cortex import conflicts, identity, synthesis, temporal, users, versions
    db, user, cmd = svc.db, svc.user, args.command
    users.resolve_actor(db, user)
    run = asyncio.run
    if cmd == "migrate":
        return {"registered": svc.engine.register(user)}
    if cmd == "sync":
        return run(svc.engine.sync(user, dry_run=args.dry_run, document_ids=args.document))
    if cmd == "sync-status":
        return run(svc.engine.status(user, args.document))
    if cmd == "recover":
        return {"operations": run(svc.engine.recover(user))}
    if cmd == "verify":
        return {"revision_graph": versions.verify_graph(db), "journal": svc.engine.journal.verify(),
                "pending_operations": [e["op_id"] for e in svc.engine.journal.pending()]}
    if cmd == "journal":
        if args.document:
            users.authorize(db, user, "read", identity.get_document(db, args.document))
        else:
            users.require_admin(db, user)
        return {"events": svc.engine.journal.events(args.document, args.limit)}
    if cmd == "tombstones":
        return {"tombstones": svc.engine.list_tombstones(user)}
    if cmd == "approve-tombstone":
        return run(svc.engine.approve_tombstone(user, args.tombstone_id))
    if cmd == "restore-tombstone":
        return run(svc.engine.restore_tombstone(user, args.tombstone_id))
    if cmd == "purge-tombstone":
        return run(svc.engine.purge_tombstone(user, args.tombstone_id))
    if cmd == "documents":
        return {"documents": [d for d in identity.list_documents(db) if users.can(db, user, "read", d)]}
    if cmd == "history":
        users.authorize(db, user, "read", identity.get_document(db, args.document_id))
        return {"document_id": args.document_id, "revisions": versions.history(db, args.document_id),
                "heads": versions.head_history(db, args.document_id)}
    if cmd == "show-revision":
        rev = versions.get_revision(db, args.revision_id)
        users.authorize(db, user, "read", identity.get_document(db, rev["document_id"]))
        return rev
    if cmd == "diff-revisions":
        for rid in (args.a, args.b):
            users.authorize(db, user, "read", identity.get_document(db, versions.get_revision(db, rid)["document_id"]))
        return versions.diff_revisions(db, args.a, args.b)
    if cmd == "restore-revision":
        return run(svc.engine.restore_revision(user, args.revision_id, args.reason))
    if cmd == "edit":
        body = Path(args.file).read_text(encoding="utf-8")
        return run(conflicts.edit_document(svc.engine, user, args.document_id, body, args.expected_revision,
                                           args.reason))
    if cmd == "conflicts":
        return {"conflicts": conflicts.list_conflicts(db, user, args.status)}
    if cmd == "show-conflict":
        c = conflicts.get_conflict(db, args.conflict_id)
        users.authorize(db, user, "read", identity.get_document(db, c["document_id"]))
        return c
    if cmd == "resolve-conflict":
        body = Path(args.body_file).read_text(encoding="utf-8") if args.body_file else None
        return run(conflicts.resolve_conflict(svc.engine, user, args.conflict_id, args.action, body, args.note))
    if cmd == "suggest-resolution":
        return conflicts.suggest_resolution(db, user, args.conflict_id, svc.model)
    if cmd == "temporal-search":
        return temporal.temporal_search(db, svc.store, user, args.query, as_of=args.as_of,
                                        changed_after=args.changed_after, changed_before=args.changed_before,
                                        valid_at=args.valid_at, filters=_json_arg(args.filter, "filter"),
                                        prefer=_json_arg(args.prefer, "prefer"), limit=args.limit)
    if cmd == "changes":
        return {"changes": temporal.changes(db, user, args.after, args.before, args.project)}
    if cmd == "reindex-history":
        users.require_admin(db, user)
        return {"indexed_chunks": temporal.reindex_history(db, svc.store)}
    if cmd == "synthesize":
        return synthesis.synthesize(db, svc.store, user, args.query, args.type, model=svc.model,
                                    filters=_json_arg(args.filter, "filter"), prefer=_json_arg(args.prefer, "prefer"),
                                    limit=args.limit, as_of=args.as_of, changed_after=args.changed_after,
                                    changed_before=args.changed_before)
    if cmd == "list-syntheses":
        return {"syntheses": synthesis.list_syntheses(db, user, args.status)}
    if cmd == "show-synthesis":
        return synthesis.read_synthesis(db, user, args.synthesis_id)
    if cmd == "check-synthesis":
        synthesis.read_synthesis(db, user, args.synthesis_id)
        return synthesis.check_synthesis(db, svc.store, args.synthesis_id)
    if cmd == "resynthesize":
        return synthesis.resynthesize(db, svc.store, user, args.synthesis_id, model=svc.model)
    if cmd == "resolve-contradiction":
        return synthesis.resolve_contradiction(db, user, args.contradiction_id, args.status, args.note)
    if cmd == "users":
        return {"users": users.list_users(db)}
    if cmd == "add-user":
        return users.add_user(db, user, args.user_id, args.role, args.display_name)
    if cmd == "grant":
        users.grant(db, user, args.document_id, args.grantee, args.capability)
        return {"granted": args.capability, "document_id": args.document_id, "user": args.grantee}
    if cmd == "set-visibility":
        users.set_visibility(db, user, args.document_id, args.visibility)
        return identity.get_document(db, args.document_id)
    raise CortexError(f"unknown command {cmd}", code="invalid_argument")


def run(argv=None, service=None):
    args = _parser().parse_args(argv)
    from cortex.api import CortexService
    svc = service
    try:
        if svc is None:
            svc = CortexService(db_path=args.cortex_db, state_path=args.state, vault_dir=args.vault_dir,
                                managed_dir=args.managed_dir, proposals_dir=args.proposals_dir,
                                use_store=not args.no_index, source_roots=args.source_roots, user=args.user)
        elif args.user:
            svc.user = args.user
        result = execute(args, svc)
        code = 0
    except CortexError as exc:
        result, code = {"error": str(exc), "code": exc.code}, 1
    except FileNotFoundError as exc:
        result, code = {"error": str(exc), "code": "not_found"}, 1
    except sqlite3.DatabaseError as exc:
        result, code = {"error": f"Cortex store error: {exc}", "code": "store_error"}, 1
    finally:
        if service is None and svc is not None:
            svc.close()
    print(json.dumps(result, indent=2, sort_keys=True, default=str, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(run())
