"""Hybrid retrieval: vector similarity + structured metadata over the single
existing Qdrant collection. VectorStore keeps all Qdrant/Ollama access; this
module owns filter validation, candidate selection, ranking policy, and
result provenance. It never writes to Qdrant.

Metadata field contract (the real payload written by ingest/sync.py via
VectorStore.upsert_chunks; see README "Hybrid retrieval"):

    field          type          present on                       filter ops
    source         keyword       every chunk ("obsidian"|"local_ingest")  eq, in
    project        keyword|null  every chunk with filter metadata          eq, in
    tags           [keyword]     every chunk with filter metadata          contains, contains_any, contains_all
    path           keyword       every chunk (vault path or                eq, in
                                 "local_ingest/<file name>")
    source_file    keyword       local_ingest chunks only (absolute path)  eq, in
    ai_structured  bool          local_ingest chunks only; a missing value eq (missing == false)
                                 is treated as false (vault notes are never
                                 AI-structured)
    doc_date       RFC3339|null  every chunk with filter metadata          gte, gt, lte, lt
    document_id    keyword       chunks of Cortex-registered documents      eq, in, not_in
                                 (payload set by cortex/sync_engine.py)
    revision_id    keyword       same; the revision the chunk was indexed   eq, in, not_in
                                 from (history collection: every revision)
    indexed_at     RFC3339       same; when Cortex last indexed the chunk   gte, gt, lte, lt

There is no stored file-type field and no native keyword-prefix match, so
source_type and path-prefix filters are deliberately unsupported (vault
top-level folders are already available as `project`).

Filter grammar (hard filters: every condition must match; compiled to a
native Qdrant filter, so a non-matching chunk can never be returned):
    {"project": "x"}                          equality
    {"project": ["x", "y"]}                   any-of (list shorthand)
    {"project": {"eq": "x"}} / {"in": [...]}  explicit forms
    {"tags": "a"} / {"tags": ["a", "b"]}      contains / contains_any
    {"tags": {"contains_all": ["a", "b"]}}    contains_all
    {"ai_structured": False}                  boolean (missing counts as False)
    {"doc_date": {"gte": "2024-01-01", "lte": "2024-12-31"}}
                                              date range; a date-only upper
                                              bound includes that whole day
Tags are normalized like ingestion (lowercase, no "#"); nested tags match
their parents because ingestion stores "a/b" as both "a" and "a/b".

Soft preferences use the same fields and value forms (a range dict for
doc_date), optionally wrapped as {"value": ..., "weight": w}. They never
exclude anything; they add a bounded boost.

Ranking contract (all tuning values in RankingConfig):
    semantic_score  = the Qdrant score (cosine similarity, [-1, 1]); None in
                      metadata-only mode
    match_p         = 1/0 per preference; for a tags preference, the fraction
                      of the preferred tags the chunk carries
    contribution_p  = weight_p * match_p, weight_p clamped to
                      [0, max_preference_weight]
    metadata_boost  = min(sum(contribution_p), max_total_boost)
    final_score     = semantic_score + metadata_boost
                      (metadata-only: final_score = metadata_boost)
Order: final desc, semantic desc, source identity asc, chunk_index asc,
point id asc -- fully deterministic. A preferred chunk can overtake another
only if that one's semantic score is at most max_total_boost higher.
Preferences require a Cosine collection so that bound is meaningful.

Candidate pool: without preferences or a diversity cap, exactly `limit`
points are requested (identical to VectorStore.search). Otherwise
    candidate_limit = min(max(limit * candidate_factor, candidate_min), candidate_max)
(and never below limit), so a preferred chunk just outside the top `limit`
can still be promoted. The pool is bounded; the collection is never scanned.

Metadata-only mode (empty query): requires at least one hard filter, scans at
most metadata_only_max matching points (in point-id order) with Qdrant's
filtered scroll, and ranks them by boost, then identity/chunk_index. If more
points match, `truncated` is true.

Diversity (optional): max_per_source caps chunks per logical source
(source_file, else path) after ranking; off by default.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from qdrant_client import models

from ingest.metadata import end_of_day_exclusive, is_date_only, normalize_tag, to_rfc3339


@dataclass(frozen=True)
class RankingConfig:
    default_preference_weight: float = 0.05
    max_preference_weight: float = 0.10
    max_total_boost: float = 0.10
    candidate_factor: int = 4
    candidate_min: int = 20
    candidate_max: int = 200
    max_limit: int = 200
    metadata_only_max: int = 1000


DEFAULT_RANKING = RankingConfig()

FIELDS = {
    "source": "keyword",
    "project": "keyword",
    "tags": "tags",
    "path": "keyword",
    "source_file": "keyword",
    "ai_structured": "bool",
    "doc_date": "datetime",
    "document_id": "keyword",
    "revision_id": "keyword",
    "indexed_at": "datetime",
}
_OPS = {
    "keyword": {"eq", "in", "not_in"},
    "tags": {"contains", "contains_any", "contains_all"},
    "bool": {"eq"},
    "datetime": {"gte", "gt", "lte", "lt"},
}


# ---------------------------------------------------------------- validation

def _keyword(field, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: values must be non-empty strings, got {value!r}")
    return value


def _tag(value):
    tag = normalize_tag(value) if isinstance(value, str) else None
    if tag is None:
        raise ValueError(f"tags: invalid tag {value!r}")
    return tag


def _values(field, value, convert):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field}: expected a non-empty list, got {value!r}")
    return sorted({convert(v) for v in value})


def _date_bound(op, value):
    """ISO date/datetime -> (qdrant_op, rfc3339). Date-only bounds mean whole
    days: lte D -> lt D+1, gt D -> gte D+1."""
    if not isinstance(value, str) or to_rfc3339(value) is None:
        raise ValueError(f"doc_date: invalid {op} bound {value!r}")
    if is_date_only(value) and op == "lte":
        return "lt", end_of_day_exclusive(value)
    if is_date_only(value) and op == "gt":
        return "gte", end_of_day_exclusive(value)
    return op, to_rfc3339(value)


def normalize_condition(field, value):
    """Validate one field condition; return (op, normalized_value). Raises
    ValueError on unknown fields/operators or bad values."""
    kind = FIELDS.get(field)
    if kind is None:
        raise ValueError(f"unknown metadata field {field!r}; supported: {sorted(FIELDS)}")
    if kind == "datetime":
        if not isinstance(value, dict) or not value:
            raise ValueError("doc_date: expected a range object like {\"gte\": \"2024-01-01\"}")
        unknown = set(value) - _OPS[kind]
        if unknown:
            raise ValueError(f"doc_date: unsupported operator(s) {sorted(unknown)}; supported: {sorted(_OPS[kind])}")
        bounds = {}
        for op in sorted(value):
            qop, bound = _date_bound(op, value[op])
            bounds[qop] = bound
        return "range", bounds
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"{field}: an operator object must have exactly one operator, got {sorted(value)}")
        (op, operand), = value.items()
        if op not in _OPS[kind]:
            raise ValueError(f"{field}: unsupported operator {op!r}; supported: {sorted(_OPS[kind])}")
    elif kind == "tags":
        op, operand = ("contains_any", value) if isinstance(value, (list, tuple)) else ("contains", value)
    elif kind == "keyword" and isinstance(value, (list, tuple)):
        op, operand = "in", value
    else:
        op, operand = "eq", value

    if kind == "bool":
        if not isinstance(operand, bool):
            raise ValueError(f"{field}: expected true or false, got {operand!r}")
        return "eq", operand
    if kind == "keyword":
        return (op, _keyword(field, operand)) if op == "eq" else (op, _values(field, operand, lambda v: _keyword(field, v)))
    if op == "contains":
        return op, _tag(operand)
    return op, _values(field, operand, _tag)


def normalize_filters(filters):
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        raise ValueError("filters must be an object mapping field -> condition")
    return {field: normalize_condition(field, filters[field]) for field in sorted(filters)}


def normalize_preferences(prefer, config=DEFAULT_RANKING):
    if prefer is None:
        return {}
    if not isinstance(prefer, dict):
        raise ValueError("prefer must be an object mapping field -> preferred value(s)")
    normalized = {}
    for field in sorted(prefer):
        value, weight = prefer[field], config.default_preference_weight
        if isinstance(value, dict) and "value" in value:
            extra = set(value) - {"value", "weight"}
            if extra:
                raise ValueError(f"prefer.{field}: unexpected key(s) {sorted(extra)}")
            weight = value.get("weight", weight)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
                raise ValueError(f"prefer.{field}: weight must be a number")
            weight = min(max(float(weight), 0.0), config.max_preference_weight)
            value = value["value"]
        op, operand = normalize_condition(field, value)
        normalized[field] = {"op": op, "value": operand, "weight": weight}
    return normalized


# --------------------------------------------------------- qdrant filter

def _match(field, values):
    if len(values) == 1:
        return models.FieldCondition(key=field, match=models.MatchValue(value=values[0]))
    return models.FieldCondition(key=field, match=models.MatchAny(any=values))


def build_qdrant_filter(normalized, exclude_document_ids=None):
    must, must_not = [], []
    if exclude_document_ids:
        must_not.append(_match("document_id", sorted(set(exclude_document_ids))))
    for field, (op, value) in normalized.items():
        if op == "range":
            must.append(models.FieldCondition(key=field, range=models.DatetimeRange(**value)))
        elif FIELDS[field] == "bool":
            condition = models.FieldCondition(key=field, match=models.MatchValue(value=value))
            if value is False:  # missing ai_structured means "not AI-structured"
                condition = models.Filter(should=[
                    condition, models.IsEmptyCondition(is_empty=models.PayloadField(key=field))])
            must.append(condition)
        elif op in ("eq", "contains"):
            must.append(_match(field, [value]))
        elif op in ("in", "contains_any"):
            must.append(_match(field, value))
        elif op == "not_in":
            must_not.append(_match(field, value))
        else:  # contains_all
            must.extend(_match(field, [v]) for v in value)
    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


def describe_filter(field, op, value):
    return {"field": field, "op": op, "value": value}


# ---------------------------------------------------------- preferences

def _payload_date(value):
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _in_range(dt, bounds):
    for op, bound in bounds.items():
        b = _payload_date(bound)
        if (op == "gte" and not dt >= b) or (op == "gt" and not dt > b) \
                or (op == "lte" and not dt <= b) or (op == "lt" and not dt < b):
            return False
    return True


def preference_match(field, op, value, payload):
    """Fraction in [0, 1] of a preference satisfied by a payload. Missing or
    malformed payload values simply don't match; this never raises."""
    actual = payload.get(field)
    kind = FIELDS[field]
    if kind == "datetime":
        dt = _payload_date(actual)
        return 1.0 if dt is not None and _in_range(dt, value) else 0.0
    if kind == "bool":
        actual = False if actual is None else actual
        return 1.0 if actual is value else 0.0
    if kind == "tags":
        have = {t for t in actual if isinstance(t, str)} if isinstance(actual, list) else set()
        wanted = [value] if op == "contains" else value
        if op == "contains_all":
            return 1.0 if set(wanted) <= have else 0.0
        return sum(1 for t in wanted if t in have) / len(wanted)
    if op == "not_in":
        return 1.0 if not (isinstance(actual, str) and actual in value) else 0.0
    wanted = [value] if op == "eq" else value
    return 1.0 if isinstance(actual, str) and actual in wanted else 0.0


def score_preferences(preferences, payload, config=DEFAULT_RANKING):
    matched, total = [], 0.0
    for field, pref in preferences.items():
        m = preference_match(field, pref["op"], pref["value"], payload)
        if m > 0:
            contribution = pref["weight"] * m
            total += contribution
            matched.append({"field": field, "op": pref["op"], "value": pref["value"],
                            "weight": pref["weight"], "match": m, "contribution": contribution})
    return min(total, config.max_total_boost), matched


# ---------------------------------------------------------------- search

def source_identity(payload, point_id):
    for key in ("source_file", "path", "source_path"):  # source_path: legacy benchmark payloads
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return str(point_id)


def candidate_limit(limit, soft, diversity, config=DEFAULT_RANKING):
    if not soft and not diversity:
        return limit
    return max(limit, min(max(limit * config.candidate_factor, config.candidate_min), config.candidate_max))


def _chunk_index(payload):
    value = payload.get("chunk_index")
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _sort_key(item):
    semantic = item["scores"]["semantic"]
    return (-item["scores"]["final"], -(semantic if semantic is not None else 0.0),
            item["identity"], _chunk_index(item["payload"]), str(item["id"]))


def hybrid_search(store, query, limit=10, filters=None, prefer=None, max_per_source=None,
                  instruct=None, config=DEFAULT_RANKING, with_timings=False, exclude_document_ids=None):
    """Run one hybrid retrieval. Returns a response dict (see module
    docstring and README for the schema). Raises ValueError for invalid
    arguments before anything is embedded or queried. Read-only."""
    started = time.perf_counter()
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= config.max_limit:
        raise ValueError(f"limit must be an integer between 1 and {config.max_limit}")
    if max_per_source is not None and (isinstance(max_per_source, bool) or not isinstance(max_per_source, int)
                                       or max_per_source < 1):
        raise ValueError("max_per_source must be a positive integer (or omitted)")
    if query is not None and not isinstance(query, str):
        raise ValueError("query must be a string")
    query = (query or "").strip()
    hard = normalize_filters(filters)
    soft = normalize_preferences(prefer, config)
    metadata_only = not query
    if metadata_only and not hard and not exclude_document_ids:
        raise ValueError("metadata-only retrieval (empty query) requires at least one hard filter")
    qfilter = build_qdrant_filter(hard, exclude_document_ids)
    mode = "metadata_only" if metadata_only else ("hybrid" if hard or soft or max_per_source else "semantic")
    pool = config.metadata_only_max if metadata_only else candidate_limit(limit, soft, max_per_source, config)
    response = {
        "query": query, "mode": mode, "limit": limit, "candidate_limit": pool,
        "filters": [describe_filter(f, op, v) for f, (op, v) in hard.items()],
        "prefer": [{"field": f, **p} for f, p in soft.items()],
        "max_per_source": max_per_source, "ranking": asdict(config),
        "collection": store.collection, "truncated": False, "results": [],
    }
    timings = {"embed_ms": 0.0, "qdrant_ms": 0.0, "rerank_ms": 0.0}

    if not store.collection_exists():
        return _finish(response, timings, started, with_timings)
    if soft and not metadata_only and str(store.distance()).lower() != "cosine":
        raise ValueError("soft preferences require a Cosine collection (the boost bound assumes cosine scores)")

    if metadata_only:
        t = time.perf_counter()
        points, next_offset = store.scroll_points(qfilter, limit=pool)
        timings["qdrant_ms"] = (time.perf_counter() - t) * 1000
        response["truncated"] = next_offset is not None
        raw = [(p, None) for p in points]
    else:
        t = time.perf_counter()
        vector = store.embed_query(query, instruct=instruct)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        points = store.query_points(vector, query_filter=qfilter, limit=pool)
        timings["qdrant_ms"] = (time.perf_counter() - t) * 1000
        raw = [(p, float(p.score)) for p in points]

    t = time.perf_counter()
    items = []
    for point, semantic in raw:
        payload = point.payload if isinstance(point.payload, dict) else {}
        boost, matched = score_preferences(soft, payload, config)
        items.append({
            "id": point.id, "payload": payload, "identity": source_identity(payload, point.id),
            "scores": {"semantic": semantic, "metadata_boost": boost,
                       "final": (semantic if semantic is not None else 0.0) + boost},
            "matched_preferences": matched,
        })
    items.sort(key=_sort_key)
    selected, per_source = [], {}
    for item in items:
        if max_per_source and per_source.get(item["identity"], 0) >= max_per_source:
            continue
        per_source[item["identity"]] = per_source.get(item["identity"], 0) + 1
        selected.append(item)
        if len(selected) == limit:
            break
    response["results"] = [_format(rank, item, response["filters"]) for rank, item in enumerate(selected, 1)]
    timings["rerank_ms"] = (time.perf_counter() - t) * 1000
    return _finish(response, timings, started, with_timings)


def _format(rank, item, filters):
    payload = item["payload"]
    path = payload.get("path") if isinstance(payload.get("path"), str) else payload.get("source_path")
    return {
        "rank": rank,
        "id": str(item["id"]),
        "text": payload.get("text") if isinstance(payload.get("text"), str) else "",
        "source": payload.get("source"),
        "path": path,
        "source_file": payload.get("source_file"),
        "chunk_index": payload.get("chunk_index"),
        "source_identity": item["identity"],
        "score": item["scores"]["final"],
        "scores": item["scores"],
        "matched_filters": filters,
        "matched_preferences": item["matched_preferences"],
        "metadata": {k: v for k, v in payload.items() if k != "text"},
    }


def _finish(response, timings, started, with_timings):
    if with_timings:
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        response["timings_ms"] = {k: round(v, 3) for k, v in timings.items()}
    return response


def to_json(response):
    """Stable JSON rendering (sorted keys)."""
    return json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True, default=str)
