# Knowledge-cortex

**A shared neural map between human and AI — built on Obsidian, Qdrant, and MCP.**

Knowledge-cortex ingests raw knowledge (PDFs, DOCX files, Markdown, and text), turns it into an Obsidian-compatible knowledge graph, and uses Qdrant as the vector database for semantic retrieval and automatic linking.

## Architecture

```text
Raw inputs
   |
   v
Extract + normalize
   |
   +--> Obsidian vault / Markdown graph
   |
   +--> Sentence Transformers embeddings
             |
             v
          Qdrant
             |
             v
Semantic search / related-note linking / AI retrieval
   |
   v
MCP + Vire
```

Qdrant is the vector-store layer for the project. ChromaDB is not used. The previous in-process FAISS index used by `graph/autolink.py` has also been replaced by Qdrant so embeddings can live in a persistent service and be shared by other agents and applications.

## Current Features

- Recursive folder scanning
- TXT, PDF, DOCX, and Markdown extraction
- Obsidian-compatible Markdown generation
- Sentence Transformers embeddings
- Qdrant-backed vector storage
- Cosine-similarity related-note lookup
- Deterministic Qdrant point IDs so re-indexing updates existing notes instead of creating duplicates

## Requirements

- Python 3.x
- A running Qdrant instance
- Python dependencies from `requirements.txt`

The default Qdrant endpoint is:

```text
http://localhost:6333
```

You can override it with environment variables:

```bash
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=knowledge_cortex
export EMBEDDING_MODEL=all-MiniLM-L6-v2

# Only needed when your Qdrant deployment requires authentication:
export QDRANT_API_KEY=your_api_key
```

## Installation

```bash
git clone https://github.com/virecorenlm/knowledge-cortex.git
cd knowledge-cortex

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a local Docker Qdrant service:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

If Qdrant already runs elsewhere in your homelab, set `QDRANT_URL` to that service instead of starting another instance.

## Basic Ingestion

```bash
python main.py <input_folder> --out <obsidian_vault>
```

Example:

```bash
python main.py ~/knowledge_feed --out ~/obsidian_vault
```

This crawls the input folder, extracts content, and writes structured Markdown into the Obsidian vault.

## Qdrant Semantic Linking

`graph/autolink.py` now uses Qdrant instead of a local FAISS index.

The linking flow is:

1. Read Markdown notes.
2. Generate normalized embeddings with Sentence Transformers.
3. Create the `knowledge_cortex` collection if it does not exist.
4. Upsert note vectors and payloads into Qdrant.
5. Query Qdrant with cosine similarity.
6. Return the five closest related notes for each note, excluding itself.

The stored payload currently includes:

```json
{
  "path": "/path/to/note.md",
  "text": "note contents"
}
```

## MCP Integration

The Obsidian/MCP layer remains the human/AI knowledge interface. Qdrant complements it by providing semantic retrieval across the cortex.

That separation gives the project two useful views of the same knowledge:

- **Obsidian** — readable Markdown, links, tags, and graph navigation.
- **Qdrant** — vector similarity, semantic recall, and machine retrieval.

## Roadmap

### Phase 1: Foundation
- [x] Multi-format ingestion
- [x] Markdown generation
- [x] Obsidian structure

### Phase 2: Vector Memory
- [x] Qdrant vector-store integration
- [x] Semantic related-note lookup
- [ ] Wire vector indexing directly into the ingestion CLI
- [ ] Chunk long documents before embedding
- [ ] Add metadata filters for source, project, tags, and dates

### Phase 3: Intelligence
- [ ] Obsidian MCP server integration
- [ ] AI-powered structuring (frontmatter, tags, metadata)
- [ ] Bidirectional sync
- [ ] Semantic search CLI/API
- [ ] Hybrid retrieval combining vectors and metadata

### Phase 4: Cognitive Infrastructure
- [ ] Knowledge synthesis
- [ ] Temporal context
- [ ] Multi-user cortex
- [ ] Version control for thought evolution
- [ ] Conflict resolution for concurrent edits

## Philosophy

Knowledge-cortex is cognitive infrastructure: persistent Markdown for human-readable memory, Qdrant for semantic machine memory, and MCP for controlled AI access.

The goal is continuity: knowledge should persist, accumulate, connect, and remain usable across sessions and across machines.

## License

MIT
