from datetime import datetime

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
