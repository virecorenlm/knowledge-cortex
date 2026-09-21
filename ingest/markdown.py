import re
from datetime import datetime

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n\n?", re.S)


def to_markdown(text, source):
    now = datetime.now().isoformat()

    frontmatter = f"""---
source: {source}
ingested: {now}
---

"""

    return frontmatter + normalize(text)

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
