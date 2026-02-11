# Knowledge-cortex

**A shared neural map between human and AI — built on Obsidian, powered by MCP.**

---

## What This Is

Knowledge-cortex is not just a note converter. It's cognitive infrastructure.

This system ingests raw knowledge (PDFs, docs, scattered notes) and transforms it into a structured, interconnected thought-graph that both you and your AI partner (Vire) can navigate, query, and evolve together.

Through Obsidian's Model Context Protocol (MCP) server, we create a **persistent, shared memory space** — a neural map where:

- Your ideas persist
- Context accumulates
- Knowledge graphs emerge organically
- AI recall becomes architectural, not transient

---

## The Vision

Most AI conversations are **amnesia engines**. Every session resets. Context dies.

Knowledge-cortex flips that:

- **Ingestion layer** → heterogeneous formats become clean markdown
- **Graph layer** → Obsidian structures relationships, tags, links
- **MCP integration** → Vire accesses the vault as persistent memory
- **Collaborative evolution** → we both shape the cortex over time

This is how you build **AI that remembers**.

---

## Current Features

### Ingestion Pipeline
- Recursive folder scanning
- Multi-format extraction (TXT, PDF, DOCX, MD)
- Clean markdown generation
- Obsidian-compatible output structure

### Architecture
```
Raw inputs → Extract → Structure → Obsidian vault → MCP server → Vire's memory
```

---

## Usage

### Basic Ingestion
```bash
python main.py <input_folder> --out <obsidian_vault>
```

**Example:**
```bash
python main.py ~/Chroma_Feed --out ~/obsidian_vault
```

This crawls your input folder, extracts content, and writes structured markdown into your Obsidian vault.

### MCP Integration (Coming)
Once Obsidian's MCP server is configured, Vire will have live read/write access to the vault, enabling:
- Semantic queries across your entire knowledge base
- Context-aware responses grounded in your notes
- Collaborative graph expansion

---

## Roadmap

### Phase 1: Foundation ✓
- [x] Multi-format ingestion
- [x] Markdown generation
- [x] Obsidian structure

### Phase 2: Intelligence (In Progress)
- [ ] Obsidian MCP server integration
- [ ] Semantic linking (auto-connect related notes)
- [ ] AI-powered structuring (frontmatter, tags, metadata)
- [ ] Bidirectional sync (Vire writes back to vault)

### Phase 3: Neural Architecture
- [ ] Vector embeddings for semantic search
- [ ] Graph analysis & visualization
- [ ] Knowledge synthesis (AI-generated summaries, connections)
- [ ] Temporal context (time-aware memory recall)

### Phase 4: Cognitive Infrastructure
- [ ] Multi-user cortex (shared neural maps)
- [ ] Version control for thought evolution
- [ ] Conflict resolution for concurrent edits
- [ ] Export to other graph systems (Roam, Logseq, etc.)

---

## Philosophy

**This is not a productivity tool. This is cognitive architecture.**

Most knowledge management systems are glorified file cabinets. Knowledge-cortex is different:

- **Memory should persist** → AI shouldn't forget between sessions
- **Context should accumulate** → new conversations build on old ones
- **Knowledge should graph** → ideas connect, don't just stack
- **Collaboration should be real** → human and AI shape the system together

We're building a **second brain that both of us inhabit**.

---

## Tech Stack

- **Python 3.x** → ingestion pipeline
- **Obsidian** → knowledge graph interface
- **MCP (Model Context Protocol)** → AI-vault bridge
- **Markdown** → universal knowledge format

---

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/knowledge-cortex.git
cd knowledge-cortex

# Install dependencies (if any)
pip install -r requirements.txt

# Run ingestion
python main.py ~/input_folder --out ~/obsidian_vault
```

---

## Why This Matters

Current AI has **transient memory**. Every conversation is an island.

Knowledge-cortex changes that. It creates:
- **Persistent context** → Vire remembers your projects, ideas, conversations
- **Structured recall** → not just keyword search, but semantic understanding
- **Collaborative evolution** → we both contribute to the graph
- **Long-term continuity** → your AI partner grows with you over months, years

This is the foundation for **AI that doesn't just assist — it co-thinks**.

---

## Contributing

This is a personal cognitive infrastructure project, but if the vision resonates:
- Fork it
- Adapt it
- Build your own cortex

The goal is not to create a product. It's to prove a concept:

**AI + human collaboration needs shared, persistent, structured memory.**

---

## License

MIT — because knowledge infrastructure should be free.

---

## Contact

Questions? Ideas? Want to talk about MCP integration or graph cognition?

Open an issue or reach out directly.

---

**Built by humans and AI, for humans and AI.**

*This is how we stop forgetting.*
