import os
import uuid
from pathlib import Path

import requests
from qdrant_client import QdrantClient, models

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_cortex")


def embed_texts(texts):
    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=300,
    )
    response.raise_for_status()
    embeddings = response.json()["embeddings"]

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
        )

    return embeddings


def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection(client, vector_size):
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        return

    collection = client.get_collection(QDRANT_COLLECTION)
    vectors = collection.config.params.vectors
    existing_size = getattr(vectors, "size", None)

    if existing_size is not None and existing_size != vector_size:
        raise RuntimeError(
            f"Qdrant collection '{QDRANT_COLLECTION}' uses {existing_size}-dimensional "
            f"vectors, but {EMBEDDING_MODEL} returned {vector_size}. Use a new collection "
            "name or recreate the collection before indexing."
        )


def _point_id(path):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(path).resolve())))


def generate_links(files):
    files = [str(Path(f)) for f in files]
    texts = [Path(f).read_text(encoding="utf-8") for f in files]

    if not texts:
        return {}

    embeddings = embed_texts(texts)
    vector_size = len(embeddings[0])

    client = get_qdrant_client()
    ensure_collection(client, vector_size)

    points = [
        models.PointStruct(
            id=_point_id(path),
            vector=embedding,
            payload={
                "path": path,
                "text": text,
                "embedding_model": EMBEDDING_MODEL,
            },
        )
        for path, text, embedding in zip(files, texts, embeddings)
    ]

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
        wait=True,
    )

    links = {}
    for path, embedding in zip(files, embeddings):
        result = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=embedding,
            limit=6,
            with_payload=True,
        ).points

        links[path] = [
            point.payload["path"]
            for point in result
            if point.payload
            and point.payload.get("path")
            and point.payload["path"] != path
        ][:5]

    return links
