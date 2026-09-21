import hashlib
import uuid


def chunk_text(text, max_chars=1200):
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def build_chunks(path, text, source="local", max_chars=1200):
    """Split text into deterministic, addressable chunks for one document.

    Each chunk gets a stable UUIDv5 id derived from (source, path, sha256, index),
    so re-ingesting identical content produces the same point IDs (upsert,
    not duplicate), while a content change changes every id for that path.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = []
    for index, part in enumerate(chunk_text(text, max_chars)):
        if not part.strip():
            continue
        identity = f"{source}:{path}:{digest}:{index}"
        chunks.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
            "text": part,
            "path": path,
            "sha256": digest,
            "chunk_index": index,
            "source": source,
        })
    return chunks
