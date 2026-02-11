from pypdf import PdfReader
from docx import Document

def extract_text(path, ftype):
    if ftype == "txt" or ftype == "md":
        return open(path, "r", encoding="utf-8", errors="ignore").read()

    if ftype == "pdf":
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    if ftype == "docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    return ""
