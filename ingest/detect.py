from pathlib import Path

def detect_file_type(path):
    ext = Path(path).suffix.lower()
    return {
        ".txt": "txt",
        ".pdf": "pdf",
        ".docx": "docx",
        ".md": "md",
    }.get(ext, "unknown")
