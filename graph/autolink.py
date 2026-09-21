import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_cortex")

model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


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


def _point_id(path):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(path).resolve())))


def generate_links(files):
    files = [str(Path(f)) for f in files]
    texts = [Path(f).read_text(encoding="utf-8") for f in files]

    if not texts:
        return {}

    embeddings = embed_texts(texts)
    client = get_qdrant_client()
    ensure_collection(client, embeddings.shape[1])

    points = [
        models.PointStruct(
            id=_point_id(path),
            vector=embedding.tolist(),
            payload={"path": path, "text": text},
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
            query=embedding.tolist(),
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
