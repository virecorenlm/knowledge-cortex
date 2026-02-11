import subprocess
import json
import tempfile
from pathlib import Path

LLAMA_BIN = "llama"  # or full path: /usr/local/bin/llama
MODEL_PATH = "/path/to/your/model.gguf"

SYSTEM_PROMPT = """
You are a knowledge structuring engine.

Convert raw notes into clean, structured markdown optimized for Obsidian.

You must:
- Infer headings and sections
- Summarize content clearly
- Extract key concepts
- Generate semantic tags
- Suggest wiki-style links

Output valid Markdown only.

Format:

---
title:
tags:
links:
---

# Title

## Summary
...

## Key Concepts
- ...

## Notes
...
"""

def ai_structure(text: str) -> str:
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    cmd = [
        LLAMA_BIN,
        "-m", MODEL_PATH,
        "-f", tmp_path,
        "-p", SYSTEM_PROMPT,
        "-n", "512"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    Path(tmp_path).unlink(missing_ok=True)

    return result.stdout.strip()
