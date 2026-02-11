from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, convert_to_numpy=True)

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def generate_links(files):
    texts = [Path(f).read_text(encoding="utf-8") for f in files]
    embeddings = embed_texts(texts)

    index = build_faiss_index(embeddings)

    links = {}

    for i, vec in enumerate(embeddings):
        D, I = index.search(vec.reshape(1, -1), 6)
        links[files[i]] = [files[j] for j in I[0][1:]]

    return links
