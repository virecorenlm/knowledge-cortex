"""Optional AI-structuring stage for local ingestion.

structure_markdown() sends already-extracted Markdown to an Ollama chat
model and asks it to reorganize it (headings, summary, tags) WITHOUT
changing factual content. The model cannot be trusted to obey that
instruction on its own, so output is validated with deterministic checks
before being accepted; any failure (network error, timeout, empty output,
failed validation) returns a result that tells the caller to fall back to
the original Markdown — this module never raises for a "the model behaved
badly" case, only for programmer errors (bad arguments).

PROMPT_VERSION must be bumped whenever SYSTEM_PROMPT or the validation
contract changes in a way that should force existing structured documents
to be regenerated (see ingest/sync.py's incremental-skip logic, which
compares this against the value recorded in saved sync state).
"""

import os
import re

import requests

PROMPT_VERSION = 1

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:12b"
DEFAULT_TIMEOUT_SECONDS = 300

SYSTEM_PROMPT = """You are a knowledge structuring engine for a personal knowledge base.

Reorganize the given Markdown document for clarity. You may:
- Add or improve a title, headings, and subheadings
- Add a short summary section
- Add relevant topic tags
- Reorganize into clearer sections
- Fix obvious Markdown formatting

You must NOT:
- Invent, guess, or add any fact, name, number, date, URL, command, or quote
  that is not already present in the source text
- Remove or omit any meaningful information from the source
- Change any number, date, name, command, URL, quote, or other technical
  value from the source, even slightly
- Change the meaning or certainty of any statement (do not turn a guess
  into a fact, or a fact into a guess)
- Fabricate citations, links, or references
- Alter the content of code blocks or inline code

Preserve all code blocks and technical content exactly as given.
Output only the resulting Markdown document, nothing else — no commentary,
no explanation of what you changed, no wrapping in an extra fenced block.
"""


class StructureResult:
    """ok=True means `text` is validated structured output, safe to index.
    ok=False means structuring failed or was rejected; `text` is the
    ORIGINAL input text (safe fallback) and `reason` explains why."""

    def __init__(self, ok, text, model, reason=None):
        self.ok = ok
        self.text = text
        self.model = model
        self.reason = reason

    def __repr__(self):
        return f"StructureResult(ok={self.ok}, model={self.model!r}, reason={self.reason!r})"


def _significant_tokens(text):
    """Extract tokens from text whose loss/change would mean lost or altered
    information: URLs, code spans/blocks, and numbers that look like real
    values (2+ digits, decimals, or dates) rather than list markers like
    "1." or "2.". This is a pragmatic, deterministic guard — not a
    substitute for the prompt, a second independent check on top of it.
    """
    tokens = set()
    tokens.update(re.findall(r"https?://\S+", text))
    tokens.update(re.findall(r"```.*?```", text, re.S))
    tokens.update(re.findall(r"`[^`\n]+`", text))
    tokens.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))          # dates
    tokens.update(re.findall(r"\b\d+\.\d+\b", text))                   # decimals/versions
    tokens.update(re.findall(r"\b\d{2,}\b", text))                     # multi-digit numbers
    return tokens


def _validate(original, structured, min_ratio=0.3):
    """Return None if structured is acceptable, else a short reason string."""
    if not isinstance(structured, str) or not structured.strip():
        return "empty or non-string output"
    original_stripped = original.strip()
    structured_stripped = structured.strip()
    if len(original_stripped) > 200 and len(structured_stripped) < min_ratio * len(original_stripped):
        return f"output too short ({len(structured_stripped)} chars vs {len(original_stripped)} source chars)"
    missing = _significant_tokens(original) - _significant_tokens(structured)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        return f"missing {len(missing)} source value(s) not found in output: {preview}"
    return None


def structure_markdown(text, model=None, ollama_url=None, http_client=None, timeout=None):
    """Attempt to AI-structure `text`. Always returns a StructureResult;
    never raises for model/network/validation failures (falls back safely).

    model: defaults to AI_STRUCTURE_MODEL env var, then DEFAULT_MODEL.
    http_client: injectable for tests (must expose .post like `requests`).

    timeout: request timeout in seconds. Precedence: explicit argument (if
    given) > AI_STRUCTURE_TIMEOUT_SECONDS env var (if set and a valid
    positive number) > DEFAULT_TIMEOUT_SECONDS (300s — local inference,
    especially a cold model, routinely takes 2+ minutes; a short timeout
    just means every run silently falls back to raw Markdown). An invalid
    env value (non-numeric, zero, negative) is ignored with the default
    used instead, rather than raising — a request-shaping problem should
    never crash ingestion.
    """
    model = model or os.getenv("AI_STRUCTURE_MODEL", DEFAULT_MODEL)
    url = (ollama_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)).rstrip("/")
    http = http_client or requests

    if timeout is None:
        raw_timeout = os.getenv("AI_STRUCTURE_TIMEOUT_SECONDS")
        timeout = DEFAULT_TIMEOUT_SECONDS
        if raw_timeout is not None:
            try:
                parsed = float(raw_timeout)
                if parsed > 0:
                    timeout = parsed
            except ValueError:
                pass  # invalid env value: fall back to the default, don't crash

    if not isinstance(text, str) or not text.strip():
        return StructureResult(False, text, model, reason="empty input, nothing to structure")

    try:
        response = http.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        structured = data.get("message", {}).get("content", "")
    except Exception as exc:  # noqa: BLE001 - any failure here must fall back, not crash the pipeline
        return StructureResult(False, text, model, reason=f"structuring request failed: {exc}")

    reason = _validate(text, structured)
    if reason:
        return StructureResult(False, text, model, reason=f"structuring output rejected: {reason}")
    return StructureResult(True, structured, model, reason=None)
