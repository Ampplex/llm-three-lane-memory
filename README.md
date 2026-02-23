# threelane-memory

**Give any LLM a lifelong memory — not chat history, an actual knowledge graph that understands what changed, what matters, and what happened when.**

---

## The Problem

LLMs are stateless. Every conversation starts from zero. The common fixes — chat logs, RAG over documents, vector stores — all share the same blind spots:

| Approach | What it misses |
|---|---|
| **Chat history** | Grows unbounded, no structure, no contradiction handling |
| **Vector stores** (Pinecone, ChromaDB, Weaviate) | Flat document chunks — no entity relationships, no temporal awareness, no state tracking |
| **RAG pipelines** | Retrieve similar text, not *relevant knowledge* — "I moved to NYC" never invalidates "I live in SF" |
| **Context-window managers** (MemGPT/Letta) | Clever paging, but still operating on raw text with no semantic structure |

None of them answer: *"What has changed since last time?"* or *"What's still true about this person?"*

## How Threelane Is Different

Threelane Memory doesn't store text — it **understands** it. Every input is parsed by an LLM into structured semantics (entities, actions, states, roles, emotions, locations) and written into a **Neo4j knowledge graph** with typed relationships.

When you say *"I adopted a dog named Max, he's 3 years old"*, Threelane doesn't just embed the sentence. It creates:

```
(Episode) ──INVOLVES──▶ (Entity: Max)
    │                        │
    ├──HAS_ACTION──▶ (Action: adopted) ──BY_ENTITY──▶ (Entity: you)
    │                                   ──ON_ENTITY──▶ (Entity: Max)
    ├──HAS_STATE──▶ (State: age=3) ──OF_ENTITY──▶ (Entity: Max)
    └──HAS_STATE──▶ (State: species=dog) ──OF_ENTITY──▶ (Entity: Max)
```

Later, when you say *"Max turned 4"*, the old state is **superseded** — not deleted, not duplicated — with a full audit trail:

```
(State: age=4, active=true) ──SUPERSEDES──▶ (State: age=3, active=false)
```

**This is what "memory" actually means.** Not similarity search — structured, evolving knowledge.

### 3-Lane Retrieval

Retrieval isn't just "find the closest vector." Threelane scores every candidate across three lanes:

```
score = 0.65 × vector_similarity + 0.30 × importance + 0.05 × recency
```

- **Vector similarity** — semantic relevance via ANN search
- **Importance** — LLM-assigned at ingestion time (landmark events score higher)
- **Recency** — exponential decay with a 365-day half-life, tuned so a 70-year-old memory still surfaces if it matters

Memories above importance 0.75 **bypass recency decay entirely** — your wedding, your child's birth, your PhD defense never get buried by yesterday's grocery list.

### Entity Resolution

"Max", "my dog Max", "the dog", "him" — Threelane resolves all of these to the same canonical entity through 3-layer matching:

1. **Case-insensitive exact** — instant lookup
2. **Substring containment** — "my dog Max" → Max
3. **Embedding similarity** — cosine ≥ 0.88 catches semantic equivalents

### Built for Decades, Not Chat Sessions

Most memory systems assume weeks of data. Threelane is tuned for a **70-year lifespan**:

- Dynamic candidate pool scales with graph size (2% of episodes, clamped 50–500)
- Consolidation engine LLM-summarizes old low-importance episodes to keep the graph manageable
- Background entity deduplication merges fragmented nodes in 3 passes
- Full JSON backup with timestamped exports for disaster recovery

---

## Features

- **Pluggable LLM providers** — Ollama (local, free) or OpenAI (cloud); switch via one env var
- **Semantic extraction** — LLM parses text into structured entities, actions, states, roles, emotions, importance, and location
- **Neo4j knowledge graph** — Episodes, Entities, Actions, States, Roles, Locations with typed relationships
- **3-layer entity resolution** — case-insensitive → substring → embedding similarity (≥ 0.88)
- **State contradiction handling** — new facts `SUPERSEDE` old ones with full audit trail
- **Vector ANN search** — Provider-matched embeddings indexed in Neo4j
- **70-year recency tuning** — 5% recency weight, 365-day half-life, importance floor bypass
- **Temporal queries** — supports "last 3 months", "in 2024", "March 2025" natively
- **Memory consolidation** — LLM-summarizes old low-importance episodes
- **Backup & export** — full graph dump to timestamped JSON
- **Entity deduplication** — 3-pass background scan and merge

## Installation

```bash
pip install threelane-memory
```

Or install from source:

```bash
git clone https://github.com/ankeshkumar/threelane-memory.git
cd threelane-memory
pip install -e .
```

### Prerequisites

- **Python 3.10+**
- **Neo4j 5.x** (Aura or self-hosted with vector index support)
- **LLM provider** — one of:
  - **Ollama** (default, free, local) — [Install Ollama](https://ollama.com)
  - **OpenAI** — requires API key from [platform.openai.com](https://platform.openai.com/api-keys)

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

#### Option A — Ollama (local, default)

```bash
# Install & start Ollama
brew install ollama
ollama serve          # keep running in a separate terminal

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Your `.env` only needs Neo4j credentials — Ollama settings default automatically:

```env
LLM_PROVIDER=ollama
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

#### Option B — OpenAI (cloud)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

Create the required Neo4j vector index (run once in Neo4j Browser — set dimension to match your provider):

```cypher
-- Ollama (nomic-embed-text): 768 dimensions
-- OpenAI (text-embedding-ada-002): 1536 dimensions
CREATE VECTOR INDEX episode_embedding IF NOT EXISTS
FOR (ep:Episode) ON (ep.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};
```

Verify your setup:

```bash
threelane-memory config
```

## Quick Start

### Python API

```python
from threelane_memory import store, query, close

# Store memories
store("My dog Max is 3 years old", speaker="ankesh")
store("I work at Google as a software engineer", speaker="ankesh")

# Query memories
answer = query("How old is Max?", speaker="ankesh")
print(answer)  # "Max is 3 years old."

close()
```

### CLI

```bash
# Interactive chat
threelane-memory chat --speaker ankesh

# Store a single memory
threelane-memory store "My dog Max is 3 years old" --speaker ankesh

# Query
threelane-memory query "How old is Max?" --speaker ankesh

# Show active provider, models & run health checks
threelane-memory config

# Admin operations
threelane-memory backup --speaker ankesh
threelane-memory dedup
threelane-memory dedup --dry-run
threelane-memory consolidate --speaker ankesh
threelane-memory reindex --speaker ankesh --old-model nomic-embed-text
```

Or via `python -m`:

```bash
python -m threelane_memory chat --speaker ankesh
```

## Architecture

```
User Input
  │
  ├── Statement ──▶ operator.py ──▶ reconciler.py ──▶ Neo4j Graph
  │                  (LLM)          (entity resolution,
  │                                   state contradiction,
  │                                   graph writing)
  │
  └── Question  ──▶ retriever.py ──▶ Neo4j Vector Index
                     (3-lane search)    + Graph Traversal
                    ──▶ llm_interface.py ──▶ Answer
                         (LLM)
```

See [docs/flowcharts.md](docs/flowcharts.md) for detailed architecture diagrams.

## Switching Providers

threelane-memory auto-detects the correct embedding dimension and warns you about mismatches. Run `threelane-memory config` at any time to check your setup.

### Ollama → OpenAI

```bash
# 1. Update .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key

# 2. Install the optional OpenAI dependency
pip install threelane-memory[openai]

# 3. Verify (will warn if Neo4j index dimension doesn't match)
threelane-memory config

# 4. Recreate the vector index (1536-dim for text-embedding-ada-002)
#    In Neo4j Browser:
#      DROP INDEX episode_embedding;
#      CREATE VECTOR INDEX episode_embedding IF NOT EXISTS
#      FOR (ep:Episode) ON (ep.embedding)
#      OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};

# 5. Re-embed all episodes with the new model
threelane-memory reindex --old-model nomic-embed-text
```

### OpenAI → Ollama

```bash
# 1. Install & start Ollama
brew install ollama && ollama serve
ollama pull llama3.2:3b && ollama pull nomic-embed-text

# 2. Update .env
LLM_PROVIDER=ollama

# 3. Verify
threelane-memory config

# 4. Recreate the vector index (768-dim for nomic-embed-text)
#    In Neo4j Browser:
#      DROP INDEX episode_embedding;
#      CREATE VECTOR INDEX episode_embedding IF NOT EXISTS
#      FOR (ep:Episode) ON (ep.embedding)
#      OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};

# 5. Re-embed all episodes
threelane-memory reindex --old-model text-embedding-ada-002
```

### Using a Custom Model

Set these in `.env` to use any Ollama-compatible model:

```env
OLLAMA_CHAT_MODEL=mistral       # any chat model
OLLAMA_EMBED_MODEL=mxbai-embed-large  # any embedding model
EMBEDDING_DIM=1024              # override auto-detection for unknown models
```

## Advanced Usage

Use the lower-level API for fine-grained control:

```python
from threelane_memory.operator import operator_extract
from threelane_memory.reconciler import reconcile, consolidate
from threelane_memory.retriever import retrieve
from threelane_memory.backup import save_backup
from threelane_memory.entity_dedup import deduplicate_entities

# Extract semantics
semantics = operator_extract("Alice presented her paper at MIT")

# Write to graph
episode_id = reconcile(semantics, speaker="ankesh", raw_text="...")

# Retrieve context
context = retrieve("What did Alice do?", speaker="ankesh")

# Admin operations
consolidate("ankesh")
save_backup(speaker="ankesh")
deduplicate_entities(dry_run=False)
```

## Project Structure

```
threelane-memory/
├── threelane_memory/          # Core package
│   ├── __init__.py            # Public API (store, query, close)
│   ├── __main__.py            # python -m threelane_memory
│   ├── cli.py                 # CLI entry point
│   ├── config.py              # Configuration from .env
│   ├── schemas.py             # TypedDict data contracts
│   ├── database.py            # Neo4j driver wrapper
│   ├── embeddings.py          # Embedding wrapper (Ollama / OpenAI)
│   ├── llm_interface.py       # LLM chat wrapper (Ollama / OpenAI)
│   ├── operator.py            # Semantic extraction via LLM
│   ├── reconciler.py          # Graph writer + entity resolution
│   ├── retriever.py           # 3-lane retrieval engine
│   ├── entity_dedup.py        # Entity deduplication
│   ├── backup.py              # Graph export to JSON
│   ├── chat.py                # Interactive chat loop
│   └── utils.py               # Shared helpers
├── examples/                  # Example scripts
│   ├── basic_usage.py         # Store and query
│   ├── advanced_pipeline.py   # Lower-level API usage
│   ├── streamlit_app.py       # Optional Streamlit web UI
│   └── debug_retrieval.py     # Diagnostic tool
├── tests/                     # Test suite
├── docs/                      # Documentation
│   └── flowcharts.md          # Architecture flowcharts
├── pyproject.toml             # Package metadata and dependencies
├── .env.example               # Environment variable template
├── LICENSE                    # MIT License
├── CONTRIBUTING.md            # Contribution guidelines
└── README.md
```

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama` (local) or `openai` (cloud) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_CHAT_MODEL` | `llama3.2:3b` | Ollama chat model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model (768-dim) |
| `OPENAI_API_KEY` | — | OpenAI API key (required when provider=openai) |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | OpenAI chat model |
| `OPENAI_EMBED_MODEL` | `text-embedding-ada-002` | OpenAI embedding model (1536-dim) |
| `EMBEDDING_DIM` | auto | Override auto-detected embedding dimension |
| `WEIGHT_SIMILARITY` | `0.65` | Vector similarity weight in scoring |
| `WEIGHT_IMPORTANCE` | `0.30` | Importance weight in scoring |
| `WEIGHT_RECENCY` | `0.05` | Recency weight (low for 70-year safety) |
| `RECENCY_HALF_LIFE_DAYS` | `365` | Days for recency to decay by 50% |
| `IMPORTANCE_FLOOR` | `0.75` | Episodes above this ignore recency decay |
| `VECTOR_CANDIDATES_MIN` | `50` | Minimum ANN candidates |
| `VECTOR_CANDIDATES_MAX` | `500` | Maximum ANN candidates |
| `VECTOR_CANDIDATES_RATIO` | `0.02` | Fraction of total episodes to search |
| `CONSOLIDATION_AGE_DAYS` | `90` | Episodes older than this are eligible |
| `CONSOLIDATION_BATCH_SIZE` | `50` | Max episodes per consolidation round |
| `CONSOLIDATION_IMPORTANCE_CAP` | `0.3` | Only consolidate below this importance |

## 70-Year Design Decisions

| Problem | Solution |
|---|---|
| Recency bias buries old memories | Recency weight = 5%, half-life = 365 days |
| Important memories forgotten | Importance floor bypass (≥ 0.75 → recency = 1.0) |
| Fixed candidate pool too small | Dynamic pool: 2% of graph, clamped 50–500 |
| Graph grows unbounded | Consolidation engine merges old trivia |
| State contradictions | SUPERSEDES chain — only latest value active |
| Entity fragmentation | 3-layer entity resolution + background dedup |
| Embedding model changes | Model version tag + batch reindex utility |
| Vector index lag | 5-minute recent-episode safety net |
| No disaster recovery | Full JSON export with incremental backup |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check .

# Type check
mypy threelane_memory/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## Tech Stack

- **Python 3.10+**
- **Neo4j 5.x** — Graph database with vector index
- **Ollama** (default) — Local LLM & embeddings (llama3.2, nomic-embed-text)
- **OpenAI** (optional) — GPT-4o (chat) + text-embedding-ada-002 (embeddings)
- **LangChain** — Unified LLM/embedding wrappers
- **NumPy** — Cosine similarity computation

## License

MIT — see [LICENSE](LICENSE).
