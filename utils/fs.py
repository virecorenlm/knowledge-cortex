from pathlib import Path

def iter_files(root):
    for p in Path(root).rglob("*"):
        if p.is_file():
            yield str(p)

def safe_write(md, src, out_root):
    src = Path(src)
    out = Path(out_root) / src.with_suffix(".md").name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
