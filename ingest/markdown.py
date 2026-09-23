import re
from datetime import datetime

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n\n?", re.S)


# Document-body contract (what becomes generated_body / the managed note
# body, before any optional AI structuring):
#   md:  the source text minus its own leading frontmatter block, verbatim.
#        That frontmatter is source metadata (it feeds filter metadata; see
#        ingest/sync.py) and is never part of the body, so the managed note
#        keeps exactly one frontmatter block (Cortex's) and ingest/apply.py
#        can keep it byte-for-byte while replacing only the body.
#   txt: the source text, verbatim. Plain text is already valid Markdown;
#        bullet-prefixing it would make a reverse-applied body re-ingest
#        into something different.
#   pdf/docx/anything else: normalize(), unchanged.
# A leading UTF-8 BOM is dropped for md/txt. For md/txt,
# source -> document_body is deterministic, and a body applied back into
# the source re-ingests to the same body (ingest/apply.py checks this
# before writing).
NATIVE_BODY_TYPES = ("md", "txt")


def document_body(text, ftype=None):
    if ftype in NATIVE_BODY_TYPES:
        text = text[1:] if text.startswith("\ufeff") else text
        return split_source_frontmatter(text)[1] if ftype == "md" else text
    return normalize(text)


def split_source_frontmatter(text):
    """(frontmatter_block, body) for a native Markdown source; the block
    includes its closing '---' line and one following blank line, as matched
    by FRONTMATTER_RE. ("", text) when there is none."""
    match = FRONTMATTER_RE.match(text)
    if not match:
        return "", text
    return match.group(0), text[match.end():]


def to_markdown(text, source, ftype=None):
    """Ingestion frontmatter + document_body(text, ftype). Without ftype,
    every line goes through normalize() (the original behavior)."""
    now = datetime.now().isoformat()

    frontmatter = f"""---
source: {source}
ingested: {now}
---

"""

    return frontmatter + document_body(text, ftype)

def normalize(text):
    lines = text.splitlines()
    return "\n".join(f"- {l}" if len(l) < 120 else l for l in lines)


def split_frontmatter(content):
    """Split a to_markdown()-style leading '---'-delimited frontmatter block
    from the rest of the document. Returns (frontmatter, body):
      frontmatter: the raw '---\\n...\\n---' text (no trailing blank line),
                   or "" if content has no leading frontmatter block.
      body:        everything after the frontmatter block (and the blank
                   line separating it from the body), or the original
                   content unchanged if there was no frontmatter.

    This is the inverse of to_markdown()'s concatenation, used wherever a
    caller needs to inspect or replace the body while preserving (or
    discarding) the frontmatter block verbatim -- e.g. AI structuring must
    never send the ingestion timestamp to the model, and managed vault
    write-back must never nest this frontmatter inside its own.
    """
    match = FRONTMATTER_RE.match(content)
    if not match:
        return "", content
    return match.group(0).rstrip("\n"), content[match.end():]
