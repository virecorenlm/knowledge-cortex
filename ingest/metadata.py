"""Filterable document metadata: project, tags, and document date.

extract_filter_metadata() derives the payload fields that semantic search
can filter on (see graph/store.build_search_filter). It is deterministic and
side-effect free, so it can be recomputed cheaply on an unchanged document
to backfill Qdrant payloads without re-embedding anything.

Returned fields (always all present, so a Qdrant set_payload over an
existing point overwrites stale values instead of leaving them behind):

    project:  str | None   explicit project argument > frontmatter "project:"
                           > top-level folder of `path` (when
                           folder_as_project=True, i.e. vault notes)
    tags:     [str]        frontmatter "tags:"/"tag:" plus inline #tags
                           (outside code), lowercased, "#" stripped, and
                           nested tags expanded so "a/b" also yields "a"
                           (matching Obsidian's tag-hierarchy search)
    doc_date: str | None   RFC 3339 UTC ("...Z"): frontmatter "date:" or
                           "created:" > a YYYY-MM-DD filename prefix >
                           fallback_date (e.g. a local file's mtime)

Frontmatter is parsed with a deliberately tiny YAML subset (scalar
"key: value", inline "[a, b]" lists, and "- item" block lists) so no YAML
dependency is needed; anything it can't read is simply ignored.
"""

import re
from datetime import date, datetime, timedelta, timezone
from pathlib import PurePosixPath

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---(?:\n|\Z)", re.S)
_FENCED_CODE_RE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.S | re.M)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
# Obsidian tag rules: preceded by start-of-line or whitespace; letters,
# digits, "_", "-", "/"; must contain at least one non-digit.
_INLINE_TAG_RE = re.compile(r"(?:(?<=\s)|^)#([\w\-/]+)", re.M)
_FILENAME_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

DATE_KEYS = ("date", "created")
TAG_KEYS = ("tags", "tag")


def parse_frontmatter(text):
    """Return {key: str | [str]} for a leading '---' frontmatter block, or {}."""
    match = _FRONTMATTER_RE.match(text.replace("\r\n", "\n"))
    if not match:
        return {}
    fields = {}
    current_list_key = None
    for line in match.group(1).split("\n"):
        stripped = line.strip()
        if current_list_key and stripped.startswith("- "):
            fields[current_list_key].append(_unquote(stripped[2:]))
            continue
        current_list_key = None
        if ":" not in line or line[:1].isspace():
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not value:
            fields[key] = []
            current_list_key = key
        elif value.startswith("[") and value.endswith("]"):
            fields[key] = [_unquote(v) for v in value[1:-1].split(",") if v.strip()]
        else:
            fields[key] = _unquote(value)
    return fields


def _unquote(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def normalize_tag(tag):
    """Lowercase, strip '#' and surrounding '/'; None if not a valid tag."""
    tag = tag.strip().lstrip("#").strip("/").lower()
    if not tag or tag.isdigit() or not re.fullmatch(r"[\w\-/]+", tag):
        return None
    return tag


def _expand_nested(tag):
    parts = tag.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def extract_tags(text, frontmatter=None):
    frontmatter = parse_frontmatter(text) if frontmatter is None else frontmatter
    raw = []
    for key in TAG_KEYS:
        value = frontmatter.get(key)
        if isinstance(value, str):
            raw.extend(re.split(r"[,\s]+", value))
        elif isinstance(value, list):
            raw.extend(value)
    body = _FRONTMATTER_RE.sub("", text.replace("\r\n", "\n"), count=1)
    body = _INLINE_CODE_RE.sub("", _FENCED_CODE_RE.sub("", body))
    raw.extend(_INLINE_TAG_RE.findall(body))
    tags = set()
    for candidate in raw:
        tag = normalize_tag(candidate)
        if tag:
            tags.update(_expand_nested(tag))
    return sorted(tags)


def to_rfc3339(value):
    """Normalize a date/datetime (or ISO-ish string) to 'YYYY-MM-DDTHH:MM:SSZ'
    in UTC. Naive values are treated as UTC. Returns None if unparseable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    else:
        text = str(value).strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_date_only(value):
    try:
        date.fromisoformat(str(value).strip())
        return True
    except ValueError:
        return False


def end_of_day_exclusive(value):
    """For a date-only upper bound 'YYYY-MM-DD', the RFC 3339 start of the
    NEXT day, so 'up to 2024-05-31' includes all of May 31st."""
    return to_rfc3339(date.fromisoformat(str(value).strip()) + timedelta(days=1))


def extract_filter_metadata(text, path=None, project=None, folder_as_project=False, fallback_date=None):
    frontmatter = parse_frontmatter(text)

    resolved_project = project
    if not resolved_project:
        fm_project = frontmatter.get("project")
        if isinstance(fm_project, list):
            fm_project = fm_project[0] if fm_project else None
        resolved_project = fm_project or None
    if not resolved_project and folder_as_project and path:
        parts = PurePosixPath(path).parts
        resolved_project = parts[0] if len(parts) > 1 else None

    doc_date = None
    for key in DATE_KEYS:
        value = frontmatter.get(key)
        if isinstance(value, str) and (doc_date := to_rfc3339(value)):
            break
    if doc_date is None and path:
        match = _FILENAME_DATE_RE.match(PurePosixPath(path).name)
        if match:
            doc_date = to_rfc3339(match.group(1))
    if doc_date is None:
        doc_date = to_rfc3339(fallback_date)

    return {
        "project": resolved_project.strip() if isinstance(resolved_project, str) and resolved_project.strip() else None,
        "tags": extract_tags(text, frontmatter),
        "doc_date": doc_date,
    }
