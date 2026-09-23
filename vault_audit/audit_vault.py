#!/usr/bin/env python3
"""Read-only Obsidian vault cleanup audit over a local snapshot.

The source vault is never opened for writing. All outputs go to --output-dir.
Only a loopback Ollama endpoint is accepted; cloud-tagged models are rejected.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import requests
from qdrant_client import QdrantClient, models

WIKI_RE = re.compile(r"(!)?\[\[([^\]]+)\]\]")
MD_LINK_RE = re.compile(r"(!)?\[[^\]]*\]\(([^)]+)\)")
FRONTMATTER_RE = re.compile(r"\A---\s*\r?\n.*?\r?\n---\s*(?:\r?\n|\Z)", re.S)
URL_SCHEMES = ("http://", "https://", "mailto:", "obsidian:", "data:")
ATTACH_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".tif", ".tiff",
    ".pdf", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mov", ".webm",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".7z", ".rar",
    ".epub", ".canvas",
}
EXCLUDED_DIRS = {".obsidian", ".trash", ".git"}
COPY_SUFFIX_RE = re.compile(
    r"(?:\bcopy(?:\s*\d+)?\b|\(\d+\)|\bduplicate\b|\bold\b|\bbackup\b|\bfinal\s*\d*\b|\bnew\b)", re.I
)
UUID_RE = re.compile(r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}|[0-9a-f]{24,64})$", re.I)
DATE_CLUTTER_RE = re.compile(r"(?:^|[-_ (])(?:19|20)\d{2}[-_]?(?:0[1-9]|1[0-2])[-_]?(?:0[1-9]|[12]\d|3[01])(?:$|[-_ )])")
PUNCT_WS_RE = re.compile(r"[^\w]+", re.UNICODE)
CLASSIFICATIONS = {
    "EXACT_DUPLICATE_ALREADY_HASHED", "NEAR_DUPLICATE", "OLDER_VERSION",
    "SUBSET_OF_OTHER_NOTE", "RELATED_BUT_DISTINCT", "UNSURE",
}


def utc_iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text(path: Path) -> tuple[str, str | None]:
    try:
        return path.read_text(encoding="utf-8-sig"), None
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace"), "invalid UTF-8 replaced"
        except OSError as exc:
            return "", str(exc)
    except OSError as exc:
        return "", str(exc)


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text, count=1)


def normalize_body_for_embedding(body: str, max_chars: int = 4000) -> str:
    # Remove comments and collapse whitespace only; retain wording, headings, links, and code.
    # Long notes are represented by deterministic beginning/middle/end samples so every
    # note is embedded without routinely consuming the model's full 4096-token context.
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) <= max_chars:
        return body
    first = max_chars * 7 // 16
    middle = max_chars * 2 // 16
    last = max_chars - first - middle
    mid = max(0, len(body) // 2 - middle // 2)
    return body[:first] + "\n[...middle sample...]\n" + body[mid:mid+middle] + "\n[...tail...]\n" + body[-last:]


def stable_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_excluded(rel: str) -> bool:
    return any(part in EXCLUDED_DIRS or part.startswith(".") for part in PurePosixPath(rel).parts[:-1])


def filename_flags(path: str) -> list[str]:
    p = PurePosixPath(path)
    stem = p.stem
    flags: list[str] = []
    if stem.strip().lower().startswith("untitled"):
        flags.append("untitled")
    if COPY_SUFFIX_RE.search(stem):
        flags.append("copy/version suffix")
    if DATE_CLUTTER_RE.search(stem):
        flags.append("date/export suffix clutter")
    if UUID_RE.fullmatch(stem.strip()):
        flags.append("UUID/random identifier")
    if len(p.name) > 120:
        flags.append("extremely long filename")
    if stem != stem.strip() or re.search(r"\s{2,}", stem):
        flags.append("suspicious whitespace")
    return flags


def canonical_candidate(paths: list[str], file_meta: dict[str, dict[str, Any]]) -> tuple[str, str]:
    def score(path: str) -> tuple[int, int, int, float, str]:
        flags = filename_flags(path)
        depth = len(PurePosixPath(path).parts)
        # Prefer non-suspicious, shallower, shorter, then newer, then lexical.
        return (len(flags), depth, len(path), -float(file_meta[path]["mtime_epoch"]), path.casefold())
    chosen = min(paths, key=score)
    return chosen, "deterministic preference: fewer suspicious-name flags, shallower/shorter path, then newer mtime"


def inventory_vault(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    folders: list[str] = []
    skipped: list[dict[str, str]] = []
    for p in sorted(root.rglob("*"), key=lambda x: x.as_posix().casefold()):
        rel = stable_rel(p, root)
        try:
            if p.is_symlink():
                skipped.append({"path": rel, "reason": "symlink skipped"})
                continue
            if p.is_dir():
                folders.append(rel)
                continue
            if not p.is_file():
                skipped.append({"path": rel, "reason": "not a regular file"})
                continue
            st = p.stat()
            data = p.read_bytes()
            files.append({
                "path": rel, "size": st.st_size, "mtime_epoch": st.st_mtime,
                "mtime_utc": utc_iso(st.st_mtime), "sha256": sha256_bytes(data),
                "extension": p.suffix.lower(), "excluded_internal": is_excluded(rel),
            })
        except OSError as exc:
            skipped.append({"path": rel, "reason": str(exc)})
    notes = [f for f in files if f["extension"] == ".md" and not f["excluded_internal"]]
    attachments = [f for f in files if f["extension"] in ATTACH_EXTS and not f["excluded_internal"]]
    auxiliary = [f for f in files if f["extension"] != ".md" and f["extension"] not in ATTACH_EXTS and not f["excluded_internal"]]
    internal = [f for f in files if f["excluded_internal"]]
    return {
        "root": str(root), "total_files": len(files), "total_bytes": sum(f["size"] for f in files),
        "folders": folders, "folder_count": len(folders), "files": files, "notes": notes,
        "note_count": len(notes), "attachments": attachments, "attachment_count": len(attachments),
        "auxiliary_files": auxiliary, "internal_files": internal, "skipped": skipped,
    }


def exact_duplicate_groups(inv: dict[str, Any]) -> list[dict[str, Any]]:
    meta = {f["path"]: f for f in inv["files"]}
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for f in inv["files"]:
        if not f["excluded_internal"]:
            groups[f["sha256"]].append(f["path"])
    out = []
    for digest, paths in sorted(groups.items()):
        if len(paths) < 2:
            continue
        canonical, reason = canonical_candidate(paths, meta)
        out.append({
            "sha256": digest, "paths": paths,
            "files": [{k: meta[p][k] for k in ("path", "size", "mtime_utc")} for p in paths],
            "recommended_canonical_candidate": canonical, "confidence": "medium", "reason": reason,
        })
    return out


def analyze_names(inv: dict[str, Any]) -> dict[str, Any]:
    notes = inv["notes"]
    by_exact: dict[str, list[str]] = collections.defaultdict(list)
    by_case: dict[str, list[str]] = collections.defaultdict(list)
    by_norm: dict[str, list[str]] = collections.defaultdict(list)
    flagged = []
    for f in notes:
        name = PurePosixPath(f["path"]).name
        by_exact[name].append(f["path"])
        by_case[name.casefold()].append(f["path"])
        norm = PUNCT_WS_RE.sub("", PurePosixPath(name).stem.casefold())
        by_norm[norm].append(f["path"])
        flags = filename_flags(f["path"])
        if flags:
            flagged.append({"path": f["path"], "flags": flags})
    exact = [{"basename": k, "paths": v} for k, v in sorted(by_exact.items()) if len(v) > 1]
    case_only = []
    for _, paths in sorted(by_case.items()):
        names = {PurePosixPath(p).name for p in paths}
        if len(paths) > 1 and len(names) > 1:
            case_only.append({"paths": paths})
    normalized = []
    for key, paths in sorted(by_norm.items()):
        names = {PurePosixPath(p).stem.casefold() for p in paths}
        if key and len(paths) > 1 and len(names) > 1:
            normalized.append({"normalized": key, "paths": paths})
    return {"exact_duplicate_basenames": exact, "case_only_duplicates": case_only,
            "punctuation_whitespace_duplicates": normalized, "flagged_names": flagged}


def analyze_notes(root: Path, inv: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    notes: dict[str, dict[str, Any]] = {}
    tiny = []
    for f in inv["notes"]:
        text, err = read_text(root / f["path"])
        body = strip_frontmatter(text)
        body_trim = body.strip()
        body_chars = len(body_trim)
        words = len(re.findall(r"\b\w+\b", body_trim, re.UNICODE))
        item = dict(f)
        item.update({"text": text, "body": body, "body_chars": body_chars, "word_count": words, "read_warning": err})
        notes[f["path"]] = item
        category = None
        if f["size"] == 0:
            category = "zero-byte"
        elif not text.strip():
            category = "whitespace-only"
        elif not body_trim and FRONTMATTER_RE.match(text):
            category = "frontmatter-only"
        elif body_chars <= 80:
            category = "extremely short"
        elif body_chars <= 200:
            category = "short"
        if category:
            tiny.append({"path": f["path"], "category": category, "body_character_count": body_chars,
                         "word_count": words, "size": f["size"]})
    return notes, tiny


def build_target_indexes(notes: dict[str, dict[str, Any]], attachments: list[dict[str, Any]]) -> dict[str, Any]:
    note_by_path: dict[str, list[str]] = collections.defaultdict(list)
    note_by_stem: dict[str, list[str]] = collections.defaultdict(list)
    attachment_by_path: dict[str, list[str]] = collections.defaultdict(list)
    attachment_by_name: dict[str, list[str]] = collections.defaultdict(list)
    for path in notes:
        pp = PurePosixPath(path)
        key_path = str(pp.with_suffix("")).casefold()
        note_by_path[key_path].append(path)
        note_by_path[path.casefold()].append(path)
        note_by_stem[pp.stem.casefold()].append(path)
    for f in attachments:
        path = f["path"]
        pp = PurePosixPath(path)
        attachment_by_path[path.casefold()].append(path)
        attachment_by_name[pp.name.casefold()].append(path)
        attachment_by_name[pp.stem.casefold()].append(path)
    return {"note_by_path": note_by_path, "note_by_stem": note_by_stem,
            "attachment_by_path": attachment_by_path, "attachment_by_name": attachment_by_name}


def resolve_wikilink(raw: str, embedded: bool, indexes: dict[str, Any]) -> tuple[str, list[str]]:
    target = raw.split("|", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    if not target:
        return "resolved_heading_only", []
    suffix = PurePosixPath(target).suffix.lower()
    attachment_intent = embedded or (suffix and suffix != ".md")
    if attachment_intent:
        path_matches = indexes["attachment_by_path"].get(target.casefold(), [])
        if path_matches:
            return ("resolved" if len(path_matches) == 1 else "ambiguous"), path_matches
        matches = indexes["attachment_by_name"].get(PurePosixPath(target).name.casefold(), [])
        return ("resolved" if len(matches) == 1 else "ambiguous" if matches else "unresolved"), matches
    target_no_md = str(PurePosixPath(target).with_suffix("")) if suffix == ".md" else target
    if "/" in target_no_md:
        matches = indexes["note_by_path"].get(target_no_md.casefold(), [])
    else:
        matches = indexes["note_by_stem"].get(PurePosixPath(target_no_md).name.casefold(), [])
    return ("resolved" if len(matches) == 1 else "ambiguous" if matches else "unresolved"), matches


def resolve_markdown_link(source: str, raw: str, embedded: bool, indexes: dict[str, Any]) -> tuple[str, list[str]]:
    target = raw.strip().strip("<>").split("#", 1)[0]
    target = urllib.parse.unquote(target).replace("\\", "/")
    if not target or target.lower().startswith(URL_SCHEMES) or target.startswith("#"):
        return "external_or_heading", []
    base = PurePosixPath(source).parent
    try:
        normalized = str(PurePosixPath(os.path.normpath(str(base / target)).replace("\\", "/")))
    except Exception:
        normalized = target
    suffix = PurePosixPath(target).suffix.lower()
    if embedded or (suffix and suffix != ".md"):
        matches = indexes["attachment_by_path"].get(normalized.casefold(), [])
        if not matches:
            matches = indexes["attachment_by_name"].get(PurePosixPath(target).name.casefold(), [])
    else:
        no_md = str(PurePosixPath(normalized).with_suffix("")) if suffix == ".md" else normalized
        matches = indexes["note_by_path"].get(no_md.casefold(), [])
    return ("resolved" if len(matches) == 1 else "ambiguous" if matches else "unresolved"), matches


def mask_code_regions(text: str) -> str:
    """Exclude fenced/inline code from link parsing to avoid shell/JSON false positives."""
    text = re.sub(
        r"(?ms)^[ \t]*(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?P=fence)[ \t]*$",
        "",
        text,
    )
    return re.sub(r"`[^`\n]*`", "", text)


def analyze_links(notes: dict[str, dict[str, Any]], attachments: list[dict[str, Any]]) -> dict[str, Any]:
    indexes = build_target_indexes(notes, attachments)
    broken = []
    ambiguous = []
    resolved_attachment_paths: set[str] = set()
    total_internal = 0
    for source, note in notes.items():
        text = mask_code_regions(note["text"])
        wiki_spans = []
        for m in WIKI_RE.finditer(text):
            wiki_spans.append(m.span())
            embedded = bool(m.group(1))
            raw = m.group(2)
            # Shell test syntax also uses [[ ... ]]; do not misreport it as an
            # Obsidian wikilink, especially in imported script-heavy notes.
            if (raw != raw.strip() or raw.lstrip().startswith(("{", "[")) or
                    "${" in raw or "$#" in raw or
                    re.search(r"(?:^|\s)(?:-[a-z]|-e[qn]|-gt|-lt|=~)(?:\s|$)", raw)):
                continue
            status, matches = resolve_wikilink(raw, embedded, indexes)
            if status in {"resolved", "ambiguous", "unresolved"}:
                total_internal += 1
            if embedded:
                resolved_attachment_paths.update(matches)
            row = {"source_note": source, "link_style": "wikilink", "embedded": embedded,
                   "link_text": m.group(0), "target": raw, "matches": matches}
            if status == "unresolved": broken.append(row)
            elif status == "ambiguous": ambiguous.append(row)
        for m in MD_LINK_RE.finditer(text):
            if any(a <= m.start() < b for a, b in wiki_spans):
                continue
            embedded = bool(m.group(1)); raw = m.group(2)
            status, matches = resolve_markdown_link(source, raw, embedded, indexes)
            if status in {"resolved", "ambiguous", "unresolved"}:
                total_internal += 1
            if embedded:
                resolved_attachment_paths.update(matches)
            row = {"source_note": source, "link_style": "markdown", "embedded": embedded,
                   "link_text": m.group(0), "target": raw, "matches": matches}
            if status == "unresolved": broken.append(row)
            elif status == "ambiguous": ambiguous.append(row)
    attachment_paths = {f["path"] for f in attachments}
    possible_orphans = [f for f in attachments if f["path"] not in resolved_attachment_paths]
    by_hash: dict[str, list[str]] = collections.defaultdict(list)
    by_name: dict[str, list[str]] = collections.defaultdict(list)
    for f in attachments:
        by_hash[f["sha256"]].append(f["path"])
        by_name[PurePosixPath(f["path"]).name.casefold()].append(f["path"])
    return {
        "total_internal_links_checked": total_internal, "broken_links": broken, "ambiguous_links": ambiguous,
        "referenced_attachments": sorted(resolved_attachment_paths), "possible_orphan_attachments": possible_orphans,
        "duplicate_attachments_by_sha256": [{"sha256": h, "paths": p} for h,p in by_hash.items() if len(p)>1],
        "duplicate_attachment_filenames": [{"filename": k, "paths": p} for k,p in by_name.items() if len(p)>1],
        "attachment_count": len(attachment_paths),
    }


class OllamaLocal:
    def __init__(self, base_url: str, embedding_model: str, reasoning_model: str):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ollama URL must be loopback HTTP; remote endpoints are forbidden")
        if ":cloud" in embedding_model or ":cloud" in reasoning_model:
            raise ValueError("cloud-tagged models are forbidden")
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.reasoning_model = reasoning_model
        self.session = requests.Session()
        self.session.trust_env = False
        self.embedding_calls = 0
        self.embedding_http_attempts = 0
        self.reasoning_calls = 0
        self.reasoning_http_attempts = 0
        self.reasoning_failures: list[str] = []

    def verify_models(self) -> list[str]:
        r = self.session.get(self.base_url + "/api/tags", timeout=30)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        for wanted in (self.embedding_model, self.reasoning_model):
            if wanted not in names:
                raise RuntimeError(f"required local Ollama model unavailable: {wanted}; available={names}")
        return names

    def embed(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            self.embedding_http_attempts += 1
            try:
                r = self.session.post(self.base_url + "/api/embed", json={
                    "model": self.embedding_model, "input": texts, "truncate": True, "keep_alive": "30m",
                    "options": {"num_ctx": 1024},
                }, timeout=900)
                r.raise_for_status()
                vectors = r.json().get("embeddings")
                if not isinstance(vectors, list) or len(vectors) != len(texts):
                    raise RuntimeError("Ollama embedding response count mismatch")
                self.embedding_calls += 1
                return vectors
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                last_error = exc
                # A Vulkan device-lost response leaves the local runner unusable.
                # Ask local Ollama to unload it, then let the next attempt reload it.
                try:
                    self.session.post(self.base_url + "/api/generate", json={
                        "model": self.embedding_model, "keep_alive": 0,
                    }, timeout=60)
                except requests.RequestException:
                    pass
                if attempt < 5:
                    time.sleep(min(5 * attempt, 20))
        raise RuntimeError(f"local embedding failed after retries: {last_error}")

    def chat_json(self, prompt: str, schema: dict[str, Any] | None = None) -> Any:
        self.reasoning_calls += 1
        payload: dict[str, Any] = {
            "model": self.reasoning_model, "stream": False, "keep_alive": "30m",
            "messages": [
                {"role": "system", "content": "You are performing a conservative, read-only Obsidian vault audit. Never propose automatic edits or deletions. Output only valid JSON matching the requested shape."},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1, "num_ctx": 32768},
        }
        if schema:
            payload["format"] = schema
        last_error: Exception | None = None
        for attempt in range(1, 4):
            self.reasoning_http_attempts += 1
            try:
                r = self.session.post(self.base_url + "/api/chat", json=payload, timeout=1800)
                r.raise_for_status()
                content = r.json().get("message", {}).get("content", "")
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", content, re.S)
                    if match:
                        return json.loads(match.group(0))
                    raise RuntimeError("local reasoning model returned non-JSON output")
            except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                try:
                    self.session.post(self.base_url + "/api/generate", json={
                        "model": self.reasoning_model, "keep_alive": 0,
                    }, timeout=60)
                except requests.RequestException:
                    pass
                if attempt < 3:
                    time.sleep(min(10 * attempt, 20))
        self.reasoning_failures.append(str(last_error))
        raise RuntimeError(f"local reasoning failed after retries: {last_error}")


def embed_notes(notes: dict[str, dict[str, Any]], ollama: OllamaLocal, work_dir: Path,
                batch_size: int = 12) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    eligible = [(p, normalize_body_for_embedding(n["body"])) for p,n in notes.items() if n["body_chars"] >= 40]
    # Reuse vectors for identical embedding text during this run, and checkpoint
    # every batch in disposable local SQLite so an interruption does not discard hours of work.
    text_to_paths: dict[str, list[str]] = collections.defaultdict(list)
    for path, text in eligible:
        text_to_paths[text].append(path)
    unique_texts = list(text_to_paths)
    cache = work_dir / "embeddings_cache.sqlite3"
    con = sqlite3.connect(cache)
    con.execute("CREATE TABLE IF NOT EXISTS embeddings (cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL, dimension INTEGER NOT NULL)")
    def cache_key(text: str) -> str:
        identity = f"{ollama.embedding_model}\0ctx=1024\0cap=4000\0{text}".encode("utf-8")
        return hashlib.sha256(identity).hexdigest()
    keys = [cache_key(text) for text in unique_texts]
    existing = {row[0] for row in con.execute("SELECT cache_key FROM embeddings")}
    pending = [(text,key) for text,key in zip(unique_texts,keys) if key not in existing]
    started = time.monotonic()
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i+batch_size]
        embedded = ollama.embed([text for text,_ in batch])
        with con:
            for (_, key), vec in zip(batch, embedded):
                arr = np.asarray(vec, dtype=np.float32)
                con.execute("INSERT OR REPLACE INTO embeddings(cache_key,vector,dimension) VALUES (?,?,?)",
                            (key, arr.tobytes(), int(arr.size)))
        print(f"embedded {min(i+batch_size, len(pending))}/{len(pending)} pending unique note bodies ({len(existing)} resumed)", flush=True)
    vec_by_key: dict[str,np.ndarray] = {}
    wanted_keys = set(keys)
    for key, blob, dim in con.execute("SELECT cache_key,vector,dimension FROM embeddings"):
        if key not in wanted_keys:
            continue
        arr=np.frombuffer(blob,dtype=np.float32)
        if arr.size == dim:
            vec_by_key[key]=arr.copy()
    con.close()
    missing=[key for key in keys if key not in vec_by_key]
    if missing:
        raise RuntimeError(f"embedding cache incomplete: {len(missing)} vector(s) missing")
    vec_by_text = {text: vec_by_key[key] for text,key in zip(unique_texts,keys)}
    paths = [p for p,_ in eligible]
    matrix = np.stack([vec_by_text[t] for p,t in eligible]) if eligible else np.empty((0,0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    meta = {"eligible_notes": len(paths), "unique_embedding_inputs": len(unique_texts),
            "resumed_embedding_inputs": len(existing & set(keys)),
            "dimension": int(matrix.shape[1]) if matrix.size else 0,
            "embedding_input_char_cap": 4000,
            "embedding_context_tokens": 1024,
            "long_note_sampling": "deterministic beginning/middle/end",
            "seconds": round(time.monotonic()-started, 2), "cache_path": str(cache)}
    return paths, matrix, meta


def containment_metrics(a: str, b: str) -> dict[str, float]:
    def shingles(text: str) -> set[str]:
        words = re.findall(r"\w+", text.casefold())
        if len(words) < 5:
            return set(words)
        return {" ".join(words[i:i+5]) for i in range(len(words)-4)}
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return {"jaccard_5gram": 0.0, "smaller_containment": 0.0}
    inter = len(sa & sb)
    return {"jaccard_5gram": inter / len(sa | sb), "smaller_containment": inter / min(len(sa), len(sb))}


def near_duplicate_candidates(paths: list[str], matrix: np.ndarray, notes: dict[str, dict[str, Any]],
                              exact_groups: list[dict[str, Any]], threshold: float = 0.86,
                              neighbors: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths:
        return [], {"search": "none"}
    exact_pairs = set()
    for g in exact_groups:
        ps = g["paths"]
        for i in range(len(ps)):
            for j in range(i+1, len(ps)):
                exact_pairs.add(tuple(sorted((ps[i], ps[j]))))
    client = QdrantClient(":memory:")
    collection = "disposable_vault_audit_near_duplicates"
    client.create_collection(collection_name=collection,
                             vectors_config=models.VectorParams(size=matrix.shape[1], distance=models.Distance.COSINE))
    batch = []
    for idx, path in enumerate(paths):
        batch.append(models.PointStruct(id=idx, vector=matrix[idx].tolist(), payload={"path": path}))
        if len(batch) >= 64:
            client.upsert(collection, batch, wait=True); batch=[]
    if batch:
        client.upsert(collection, batch, wait=True)
    pairs: dict[tuple[str,str], float] = {}
    for idx, path in enumerate(paths):
        hits = client.query_points(collection_name=collection, query=matrix[idx].tolist(),
                                   limit=neighbors+1, with_payload=True).points
        for hit in hits:
            other = hit.payload["path"]
            if other == path or hit.score < threshold:
                continue
            key = tuple(sorted((path, other)))
            if key in exact_pairs:
                continue
            pairs[key] = max(pairs.get(key, -1.0), float(hit.score))
    client.delete_collection(collection)
    rows = []
    for (a,b), sim in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
        metrics = containment_metrics(notes[a]["body"], notes[b]["body"])
        rows.append({"paths": [a,b], "similarity": round(sim, 6), **{k: round(v,6) for k,v in metrics.items()},
                     "sizes": [notes[a]["size"], notes[b]["size"]],
                     "body_chars": [notes[a]["body_chars"], notes[b]["body_chars"]]})
    return rows, {"search": "QdrantClient(:memory:)", "temporary_collection": collection,
                  "temporary_collection_deleted": True, "threshold": threshold, "neighbors_per_note": neighbors}


def group_candidate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent: dict[str,str] = {}
    def find(x: str) -> str:
        parent.setdefault(x,x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x=parent[x]
        return x
    def union(a: str,b: str):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for row in pairs:
        union(*row["paths"])
    groups: dict[str,set[str]] = collections.defaultdict(set)
    for p in parent: groups[find(p)].add(p)
    result=[]
    for members in groups.values():
        member_pairs=[r for r in pairs if set(r["paths"]).issubset(members)]
        result.append({"group_id": hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:12],
                       "paths": sorted(members), "pairs": member_pairs,
                       "max_similarity": max(r["similarity"] for r in member_pairs),
                       "min_similarity": min(r["similarity"] for r in member_pairs)})
    return sorted(result, key=lambda g:(-g["max_similarity"], g["paths"]))


def excerpt(body: str, limit: int = 1800) -> str:
    clean = body.strip()
    if len(clean) <= limit:
        return clean
    half=limit//2
    return clean[:half] + "\n[...excerpt omitted...]\n" + clean[-half:]


def review_near_duplicates(groups: list[dict[str, Any]], notes: dict[str, dict[str, Any]],
                           ollama: OllamaLocal, batch_groups: int = 8) -> list[dict[str, Any]]:
    schema = {"type":"object","properties":{"reviews":{"type":"array","items":{
        "type":"object","properties":{
            "group_id":{"type":"string"},
            "classification":{"type":"string","enum":sorted(CLASSIFICATIONS)},
            "likely_canonical_note":{"type":["string","null"]},
            "reason":{"type":"string"},
            "unique_information":{"type":"object","additionalProperties":{"type":"string"}},
            "suggested_action":{"type":"string"},
            "confidence":{"type":"string","enum":["low","medium","high"]}},
        "required":["group_id","classification","likely_canonical_note","reason","unique_information","suggested_action","confidence"]}}},
        "required":["reviews"]}
    reviews=[]
    for i in range(0,len(groups),batch_groups):
        payload=[]
        for g in groups[i:i+batch_groups]:
            items=[]
            for p in g["paths"]:
                n=notes[p]
                items.append({"path":p,"size":n["size"],"mtime_utc":n["mtime_utc"],"body_chars":n["body_chars"],"excerpt":excerpt(n["body"])})
            payload.append({"group_id":g["group_id"],"pairs":g["pairs"],"notes":items})
        prompt=("Conservatively classify each candidate group. Similarity is candidate-generation evidence only. "
                "Use exactly one allowed classification. Identify unique information per file. A suggested action must always require manual review; never direct deletion. "
                "Return one review for every group_id.\n\nCANDIDATES:\n"+json.dumps(payload,ensure_ascii=False))
        try:
            response=ollama.chat_json(prompt,schema)
            got={r.get("group_id"):r for r in response.get("reviews",[]) if isinstance(r,dict)}
        except Exception as exc:
            got={}
            ollama.reasoning_failures.append(str(exc))
        for g in groups[i:i+batch_groups]:
            r=got.get(g["group_id"])
            if not r or r.get("classification") not in CLASSIFICATIONS:
                r={"group_id":g["group_id"],"classification":"UNSURE","likely_canonical_note":None,
                   "reason":"Local model review failed or returned incomplete structured output.",
                   "unique_information":{},"suggested_action":"Manual side-by-side review.","confidence":"low"}
            r["paths"]=g["paths"]; r["pairs"]=g["pairs"]
            reviews.append(r)
        print(f"reviewed near-duplicate groups {min(i+batch_groups,len(groups))}/{len(groups)}",flush=True)
    return reviews


def folder_diagnostics(inv: dict[str, Any], notes: dict[str, dict[str, Any]], paths: list[str], matrix: np.ndarray) -> dict[str, Any]:
    direct: dict[str,list[str]]=collections.defaultdict(list)
    recursive: dict[str,int]=collections.Counter()
    for p in notes:
        folder=str(PurePosixPath(p).parent)
        if folder==".": folder="/"
        direct[folder].append(p)
        parts=PurePosixPath(p).parent.parts
        recursive["/"]+=1
        for i in range(1,len(parts)+1): recursive[str(PurePosixPath(*parts[:i]))]+=1
    summaries=[]
    all_folders=sorted(set(inv["folders"])|set(direct))
    for folder in all_folders:
        if is_excluded(folder+"/x"): continue
        names=[PurePosixPath(p).name for p in sorted(direct.get(folder,[]))]
        flags=[]
        name=PurePosixPath(folder).name
        if name.startswith("_"): flags.append("leading underscore hierarchy")
        if name and name!=name.strip(): flags.append("surrounding whitespace")
        if name and re.search(r"\s{2,}",name): flags.append("repeated whitespace")
        if name and "_" in name and " " in name: flags.append("mixed spaces and underscores")
        if name and name.lower()!=name and name.upper()!=name and "_" in name: flags.append("mixed capitalization style")
        summaries.append({"folder":folder,"direct_note_count":len(names),"recursive_note_count":recursive.get(folder,0),
                          "sample_filenames":names[:15],"naming_flags":flags})
    # Embedding-based misplaced-note candidates: compare note to direct-folder centroids.
    index={p:i for i,p in enumerate(paths)}
    eligible_folders={f:[p for p in ps if p in index] for f,ps in direct.items()}
    eligible_folders={f:ps for f,ps in eligible_folders.items() if len(ps)>=3}
    centroids={}
    for f,ps in eligible_folders.items():
        c=np.mean(matrix[[index[p] for p in ps]],axis=0); c=c/max(float(np.linalg.norm(c)),1e-12); centroids[f]=c
    misplaced=[]
    for folder, ps in eligible_folders.items():
        own=centroids[folder]
        for p in ps:
            v=matrix[index[p]]; own_score=float(v@own)
            alternatives=sorted(((float(v@c),f) for f,c in centroids.items() if f!=folder),reverse=True)
            if alternatives:
                alt_score,alt=alternatives[0]
                if alt_score>=0.78 and alt_score>=own_score+0.08:
                    misplaced.append({"path":p,"current_folder":folder,"suggested_cluster_folder":alt,
                                      "current_similarity":round(own_score,5),"alternative_similarity":round(alt_score,5),
                                      "excerpt":excerpt(notes[p]["body"],600)})
    return {"folder_summaries":summaries,"misplaced_note_candidates":sorted(misplaced,key=lambda x:-x["alternative_similarity"])}


def review_organization(diag: dict[str, Any], ollama: OllamaLocal) -> list[dict[str, Any]]:
    folders=diag["folder_summaries"]
    misplaced=diag["misplaced_note_candidates"]
    recs=[]
    schema={"type":"object","properties":{"recommendations":{"type":"array","items":{
        "type":"object","properties":{
            "category":{"type":"string"},"current_path":{"type":"string"},"proposed_name_or_location":{"type":["string","null"]},
            "reason":{"type":"string"},"confidence":{"type":"string","enum":["low","medium","high"]},
            "potential_link_impact":{"type":"string"}},
        "required":["category","current_path","proposed_name_or_location","reason","confidence","potential_link_impact"]}}},
        "required":["recommendations"]}
    # Folder batches: enough context for local structure without entire note bodies.
    for i in range(0,len(folders),35):
        payload=folders[i:i+35]
        prompt=("Review this batch of Obsidian folder summaries conservatively. Recommend only clear naming/structure improvements: inconsistent names, redundant folders, broad/fragmented folders, rename opportunities, or archive candidates. "
                "Leading underscores may be an intentional ordering system: identify patterns but do not recommend removing them unless clearly inconsistent. Avoid needless reorganization. All suggestions are manual-only.\n\nFOLDERS:\n"+json.dumps(payload,ensure_ascii=False))
        try:
            response=ollama.chat_json(prompt,schema)
            for r in response.get("recommendations",[]):
                if isinstance(r,dict): recs.append(r)
        except Exception as exc:
            ollama.reasoning_failures.append(str(exc))
        print(f"reviewed folders {min(i+35,len(folders))}/{len(folders)}",flush=True)
    # Misplaced notes are reviewed separately in modest batches.
    for i in range(0,len(misplaced),30):
        payload=misplaced[i:i+30]
        prompt=("These notes are embedding-based folder mismatch candidates, not facts. Conservatively decide whether each likely belongs elsewhere. Return recommendations only for convincing cases; use category 'possible misplaced note'. Never propose automatic moves.\n\nCANDIDATES:\n"+json.dumps(payload,ensure_ascii=False))
        try:
            response=ollama.chat_json(prompt,schema)
            for r in response.get("recommendations",[]):
                if isinstance(r,dict): recs.append(r)
        except Exception as exc:
            ollama.reasoning_failures.append(str(exc))
    # Deduplicate exact repeated suggestions.
    seen=set(); out=[]
    for r in recs:
        key=(str(r.get("category","")),str(r.get("current_path","")),str(r.get("proposed_name_or_location","")))
        if key not in seen:
            seen.add(key); out.append(r)
    return out


def write_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True),encoding="utf-8")


def md_escape(s: Any) -> str:
    return str(s).replace("`","\\`")


def build_markdown(report: dict[str, Any]) -> str:
    inv=report["inventory"]; exact=report["exact_duplicates"]; near=report["near_duplicate_reviews"]
    naming=report["naming_problems"]; tiny=report["empty_tiny_notes"]; links=report["links"]
    recs=report["folder_recommendations"]
    lines=["# Vault Cleanup Audit","","**Read-only audit. No cleanup action was applied.**","",
           "## Executive Summary","",
           f"- Vault audited: `{md_escape(report['vault_identity']['remote_path'])}` (local read-only snapshot)",
           f"- Total notes analyzed: **{inv['note_count']}**",
           f"- Total files: **{inv['total_files']}**",
           f"- Total folders: **{inv['folder_count']}**",
           f"- Attachments: **{inv['attachment_count']}**",
           f"- Auxiliary non-note files: **{len(inv['auxiliary_files'])}**",
           f"- Exact duplicate groups: **{len(exact)}**",
           f"- Near-duplicate candidate groups: **{len(near)}**",
           f"- Broken links: **{len(links['broken_links'])}**",
           f"- Ambiguous links: **{len(links['ambiguous_links'])}**",
           f"- Empty/tiny notes: **{len(tiny)}**",
           f"- Possible orphan attachments: **{len(links['possible_orphan_attachments'])}**",
           f"- Folder rename/location recommendations: **{len(recs)}**","",
           "## Exact Duplicates",""]
    if not exact: lines.append("No exact duplicate groups found.")
    for i,g in enumerate(exact,1):
        lines += [f"### Group {i}","",f"- SHA256: `{g['sha256']}`",f"- Recommended canonical candidate: `{md_escape(g['recommended_canonical_candidate'])}`",
                  f"- Confidence/reason: {g['confidence']} — {g['reason']}","- Files:"]
        for f in g["files"]: lines.append(f"  - `{md_escape(f['path'])}` — {f['size']} bytes — {f['mtime_utc']}")
        lines.append("")
    lines += ["## Near Duplicates",""]
    if not near: lines.append("No near-duplicate candidate groups met the conservative candidate threshold.")
    for i,g in enumerate(near,1):
        sims=", ".join(f"{p['similarity']:.4f}" for p in g.get("pairs",[]))
        lines += [f"### Candidate Group {i}: {g.get('classification','UNSURE')}","",f"- Paths: {', '.join('`'+md_escape(p)+'`' for p in g.get('paths',[]))}",
                  f"- Similarity score(s): {sims or 'n/a'}",f"- Likely canonical note: `{md_escape(g.get('likely_canonical_note'))}`" if g.get('likely_canonical_note') else "- Likely canonical note: none selected",
                  f"- Reason: {g.get('reason','')}",f"- Suggested action: {g.get('suggested_action','Manual review.')}",f"- Confidence: {g.get('confidence','low')}","- Unique information:"]
        for p,u in g.get("unique_information",{}).items(): lines.append(f"  - `{md_escape(p)}`: {u}")
        lines.append("")
    lines += ["## Naming Problems","",f"- Exact duplicate basenames: {len(naming['exact_duplicate_basenames'])}",f"- Case-only duplicate groups: {len(naming['case_only_duplicates'])}",f"- Punctuation/whitespace-normalized groups: {len(naming['punctuation_whitespace_duplicates'])}",f"- Individually flagged names: {len(naming['flagged_names'])}",""]
    for cat in ("exact_duplicate_basenames","case_only_duplicates","punctuation_whitespace_duplicates"):
        if naming[cat]:
            lines.append(f"### {cat.replace('_',' ').title()}")
            for item in naming[cat]: lines.append(f"- {', '.join('`'+md_escape(p)+'`' for p in item['paths'])}")
            lines.append("")
    if naming["flagged_names"]:
        lines.append("### Flagged filenames")
        for item in naming["flagged_names"]: lines.append(f"- `{md_escape(item['path'])}` — {', '.join(item['flags'])}")
        lines.append("")
    lines += ["## Empty / Tiny Notes",""]
    if not tiny: lines.append("No empty or tiny notes found.")
    for n in tiny: lines.append(f"- `{md_escape(n['path'])}` — {n['category']}; {n['body_character_count']} body chars; {n['word_count']} words")
    lines += ["","## Broken Links","",f"Checked {links['total_internal_links_checked']} internal-style links.",""]
    if not links["broken_links"]: lines.append("No unresolved internal links found.")
    for x in links["broken_links"]: lines.append(f"- `{md_escape(x['source_note'])}` → `{md_escape(x['link_text'])}` (unresolved target: `{md_escape(x['target'])}`)")
    lines += ["","### Ambiguous Links",""]
    if not links["ambiguous_links"]: lines.append("No ambiguous internal links found.")
    for x in links["ambiguous_links"]: lines.append(f"- `{md_escape(x['source_note'])}` → `{md_escape(x['link_text'])}`; matches: {', '.join('`'+md_escape(p)+'`' for p in x['matches'])}")
    lines += ["","## Possible Orphan Attachments",""]
    if not links["possible_orphan_attachments"]: lines.append("No possible orphan attachments found under the conservative attachment-extension definition.")
    for a in links["possible_orphan_attachments"]: lines.append(f"- `{md_escape(a['path'])}` — {a['size']} bytes — {a['mtime_utc']}")
    lines += ["","## Folder Structure Review",""]
    if not recs: lines.append("The local model produced no folder rename/location recommendations strong enough to report.")
    for r in recs:
        lines += [f"### {r.get('category','Organization suggestion')}",f"- Current folder/path: `{md_escape(r.get('current_path',''))}`",
                  f"- Proposed name/location: `{md_escape(r.get('proposed_name_or_location'))}`" if r.get('proposed_name_or_location') else "- Proposed name/location: manual review only",
                  f"- Reason: {r.get('reason','')}",f"- Confidence: {r.get('confidence','low')}",f"- Potential link impact: {r.get('potential_link_impact','Review inbound links before any manual change.')}",""]
    lines += ["## General Vault Organization Problems","",
              f"- The vault has {inv['folder_count']} folders for {inv['note_count']} notes; review the machine-readable folder summaries before reorganizing.",
              f"- {sum(1 for x in report['folder_diagnostics']['folder_summaries'] if x['naming_flags'])} folders have deterministic naming-style flags.",
              f"- {len(report['folder_diagnostics']['misplaced_note_candidates'])} embedding-based misplaced-note candidates were generated; these are hypotheses, not move instructions.",
              "- Leading-underscore folder names appear to be a deliberate ordering convention in much of this vault. Do not flatten or rename them in bulk without deciding on a stable replacement convention.","",
              "## Suggested Cleanup Order","","1. Review exact duplicate groups; choose canonicals manually and inspect inbound links before any deletion.",
              "2. Review model-classified near-duplicate groups and preserve unique information.","3. Fix clearly broken links, starting with high-reference notes.",
              "4. Decide on a stable folder naming convention before any folder renames.","5. Move only high-confidence misplaced notes, one batch at a time, using Obsidian-aware moves.",
              "6. Review possible orphan attachments manually.","7. Run this audit again after cleanup.","8. Only then perform the first production Knowledge Cortex/Qdrant ingest.","",
              "## Manual Review Queue","","- Every exact duplicate group before deletion.","- Every `UNSURE`, `OLDER_VERSION`, `SUBSET_OF_OTHER_NOTE`, or `NEAR_DUPLICATE` group before merge/deletion.",
              "- All ambiguous links.","- All folder/location recommendations, especially those with low or medium confidence.","- Any file listed under skipped/read-warning data in the JSON report.","",
              "## Audit Method / Safety","",f"- Embedding model: `{report['models']['embedding']}` via loopback Ollama only.",f"- Reasoning model: `{report['models']['reasoning']}` via loopback Ollama only.",
              f"- Embedding HTTP calls: {report['model_usage']['embedding_calls']}",f"- Reasoning-model HTTP calls: {report['model_usage']['reasoning_calls']}",
              f"- Runtime: {report['runtime_seconds']:.1f} seconds",f"- Files skipped: {len(inv['skipped'])}","- No production Qdrant connection was created; near-neighbor search used an in-memory Qdrant client and its disposable collection was deleted.",
              "- No vault file was opened for writing by this script.",""]
    return "\n".join(lines)


def slim_inventory(inv: dict[str, Any]) -> dict[str, Any]:
    return {k:v for k,v in inv.items() if k not in {"files","notes","attachments"}}


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--work-dir",type=Path,required=True)
    ap.add_argument("--remote-vault-path",required=True)
    ap.add_argument("--ollama-url",default="http://127.0.0.1:11434")
    ap.add_argument("--embedding-model",default="qwen3-embedding:8b")
    ap.add_argument("--reasoning-model",default="qwen3.8:27b")
    ap.add_argument("--similarity-threshold",type=float,default=0.86)
    args=ap.parse_args()
    root=args.snapshot.resolve(); out=args.output_dir.resolve(); work=args.work_dir.resolve()
    if not root.is_dir(): raise SystemExit(f"snapshot does not exist: {root}")
    if out==root or root in out.parents or out in root.parents: raise SystemExit("output directory must be outside snapshot")
    out.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    ollama=OllamaLocal(args.ollama_url,args.embedding_model,args.reasoning_model)
    ollama.verify_models()
    print("Phase 1: deterministic inventory and audit",flush=True)
    inv=inventory_vault(root)
    notes,tiny=analyze_notes(root,inv)
    exact=exact_duplicate_groups(inv)
    naming=analyze_names(inv)
    links=analyze_links(notes,inv["attachments"])
    write_json(out/"exact_duplicates.json",exact)
    write_json(out/"broken_links.json",{"broken_links":links["broken_links"],"ambiguous_links":links["ambiguous_links"]})
    write_json(out/"attachment_audit.json",{k:v for k,v in links.items() if "link" not in k})
    print(f"deterministic: {inv['note_count']} notes, {len(exact)} exact groups, {len(links['broken_links'])} broken links",flush=True)
    print("Phase 2: local embeddings and in-memory nearest-neighbor search",flush=True)
    paths,matrix,embed_meta=embed_notes(notes,ollama,work)
    pairs,search_meta=near_duplicate_candidates(paths,matrix,notes,exact,threshold=args.similarity_threshold)
    groups=group_candidate_pairs(pairs)
    write_json(out/"near_duplicate_candidates.json",{"groups":groups,"pairs":pairs,"embedding":embed_meta,"search":search_meta})
    print(f"near duplicate candidates: {len(pairs)} pairs in {len(groups)} groups",flush=True)
    print("Phase 3: local reasoning review of candidates",flush=True)
    near_reviews=review_near_duplicates(groups,notes,ollama)
    write_json(out/"near_duplicates.json",near_reviews)
    print("Phase 4: local folder/organization review",flush=True)
    folder_diag=folder_diagnostics(inv,notes,paths,matrix)
    folder_recs=review_organization(folder_diag,ollama)
    write_json(out/"folder_recommendations.json",{"recommendations":folder_recs,"diagnostics":folder_diag})
    runtime=time.monotonic()-started
    report={
        "generated_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "vault_identity":{"remote_path":args.remote_vault_path,"snapshot_path":str(root)},
        "inventory":slim_inventory(inv),"exact_duplicates":exact,"near_duplicate_candidates":groups,
        "near_duplicate_reviews":near_reviews,"naming_problems":naming,"empty_tiny_notes":tiny,"links":links,
        "folder_diagnostics":folder_diag,"folder_recommendations":folder_recs,
        "models":{"embedding":args.embedding_model,"reasoning":args.reasoning_model,"ollama_url":args.ollama_url},
        "model_usage":{"embedding_calls":ollama.embedding_calls,
                       "embedding_http_attempts":ollama.embedding_http_attempts,
                       "reasoning_calls":ollama.reasoning_calls,
                       "reasoning_http_attempts":ollama.reasoning_http_attempts,
                       "reasoning_failures":ollama.reasoning_failures},
        "embedding_metadata":embed_meta,"vector_search":search_meta,"runtime_seconds":round(runtime,2),
        "safety":{"cloud_contacted":False,"production_qdrant_contacted":False,"vault_writes":False},
    }
    write_json(out/"vault_audit_report.json",report)
    (out/"vault_audit_report.md").write_text(build_markdown(report),encoding="utf-8")
    # Embeddings were cached only for the current run; remove them after reports exist.
    cache=Path(embed_meta["cache_path"])
    if cache.exists(): cache.unlink()
    write_json(out/"run_summary.json",{
        "runtime_seconds":round(runtime,2),"embedding_calls":ollama.embedding_calls,
        "embedding_http_attempts":ollama.embedding_http_attempts,
        "reasoning_calls":ollama.reasoning_calls,
        "reasoning_http_attempts":ollama.reasoning_http_attempts,
        "reasoning_failures":len(ollama.reasoning_failures),
        "embedding_cache_removed":not cache.exists(),"temporary_qdrant_collection_deleted":search_meta.get("temporary_collection_deleted",False),
    })
    print(json.dumps({"notes":inv["note_count"],"files":inv["total_files"],"folders":inv["folder_count"],
                      "attachments":inv["attachment_count"],"exact_duplicate_groups":len(exact),
                      "near_duplicate_groups":len(groups),"broken_links":len(links["broken_links"]),
                      "ambiguous_links":len(links["ambiguous_links"]),"tiny_notes":len(tiny),
                      "possible_orphans":len(links["possible_orphan_attachments"]),
                      "folder_recommendations":len(folder_recs),"runtime_seconds":round(runtime,2)},indent=2),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
