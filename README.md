# LLM Three-Lane Memory: A Cognitive Architecture for Long-Term AI Agent Memory
**Short-term → Episodic → Long-term memory consolidation for scalable AI agents**

Lifetime persistent memory for LLMs — a knowledge graph that remembers everything, knows what changed, and never forgets what matters.
Every fact you store today is still retrievable 50 years from now — accurately, instantly, and in context.

```python
from threelane_memory import store, query

store("I love playing football")
store("I just started at Google as a software engineer in San Francisco")

# Months later...
store("I got promoted to senior engineer at Google")
store("I moved from San Francisco to New York")

# A friend invites you to play football — should you go?
query("My friend invited me to play football, should I go?")
# → "Yes, you should go! You love playing football."

query("Where do I work?")        # → "Google, as a senior software engineer." (role updated)
query("Where do I live?")        # → "New York." (not SF — state was superseded)
query("Did I ever live in SF?")  # → "Yes, before moving to New York." (history preserved)
```

The football memory was stored months ago in a completely different conversation. But when you need it, Threelane connects the dots — because it actually *understands* what it stored, not just what's similar.

No context windows. No token limits. No forgetting. **Persistent memory that outlives any single conversation — or any single year.**

---

## Why This Exists

LLMs are stateless. Every conversation starts from zero. The common workarounds all break down over time:

| Approach | Works for a week | Breaks at scale |
|---|---|---|
| **Chat history** | ✓ | Grows unbounded, no structure, can't handle contradictions |
| **Vector stores** (Pinecone, ChromaDB) | ✓ | Flat chunks — "I moved to NYC" never invalidates "I live in SF" |
| **RAG pipelines** | ✓ | Retrieves similar text, not relevant *knowledge* |
| **Context-window managers** (MemGPT) | ✓ | Clever paging, but still raw text — no entity tracking, no state evolution |

Ask any of them: *"What changed about me since last year?"* — silence.

Ask Threelane — it knows, because it actually tracks entities, states, and time.

---

## What Makes Threelane Different

Most agent memory systems are just vector databases with chat history attached.  
Threelane introduces **cognitively inspired memory mechanics** that make long-term memory actually work.

### 🧠 1. 3-Lane Retrieval (Searches Memory Like Humans)

Instead of relying on vector similarity alone, Threelane retrieves memories through **three parallel lanes**:

- **Recent lane** — prevents immediate context loss  
- **Temporal lane** — understands time queries like *“last year”* or *“in 2022”*  
- **Semantic lane** — traditional ANN vector similarity  

These lanes are merged and re-ranked to surface the right memory.

**Result:** Memory retrieval becomes multi-dimensional — not just semantic, but also temporal and contextual.

---

### ⭐ 2. Landmark Memory Protection

Humans don’t forget major life events just because they’re old.  
Threelane mirrors this behavior with an **importance floor**:

importance ≥ 0.75 → recency decay disabled

Your wedding, promotions, diagnoses, or life milestones remain instantly retrievable decades later.

---

### 🔄 3. State Supersession (Memory Evolves, Not Overwrites)

Facts change over time. Threelane tracks this explicitly:

(State: city=New York) ──SUPERSEDES──▶ (State: city=San Francisco)

This enables:

- Accurate current answers  
- Full historical trace  
- Contradiction resolution  
- Versioned facts  

Ask *“Where do I live?”* → New York  
Ask *“Did I ever live in SF?”* → Yes, before moving  

---

### 📈 4. Lifespan-Aware Retrieval Scaling

Most systems search a fixed number of memories.  
Threelane adapts search dynamically:

candidate_pool = 2% of total episodes (clamped 50–500)

This ensures:

- Efficient retrieval at small scale  
- Deep search at large scale  
- Lifelong memory scalability  

---

### 🔮 5. Future-Proof Embedding Migration

Embedding models evolve. Most memory systems silently degrade when they change.

Threelane stores embedding model versions and provides batch re-indexing utilities so memory remains usable across decades of model upgrades.

---

### 🧩 6. Temporal Knowledge Graph Memory

By combining:

- structured episodic graph storage  
- supersession chains  
- entity resolution  
- consolidation  
- hybrid retrieval  

Threelane behaves less like a vector store and more like a **temporal knowledge graph for personal memory**.

---

## How It Works

Threelane doesn't store text. It **understands** it.

Every input is parsed by an LLM into structured semantics — entities, actions, states, roles, emotions, locations — and written into a **Neo4j knowledge graph** with typed relationships. This isn't an embedding dump. It's a living, evolving model of everything you've told it.

### What Happens When You Store a Memory

*"I love playing football"* becomes a structured graph:

```
(Episode) ──INVOLVES──▶ (Entity: you)
    │
    ├──HAS_STATE──▶ (State: hobby=football, sentiment=love) ──OF_ENTITY──▶ (Entity: you)
    └──INVOLVES──▶ (Entity: football)
```

*"I just started at Google as a software engineer in San Francisco"*:

```
(Episode) ──INVOLVES──▶ (Entity: Google)
    │                        │
    ├──HAS_ACTION──▶ (Action: started working) ──BY_ENTITY──▶ (Entity: you)
    │                                          ──ON_ENTITY──▶ (Entity: Google)
    ├──HAS_STATE──▶ (State: role=software engineer) ──OF_ENTITY──▶ (Entity: you)
    └──AT_LOCATION──▶ (Location: San Francisco)
```

Months later, when someone asks *"My friend invited me to play football, should I go?"*, Threelane retrieves the football episode by semantic similarity, sees the `sentiment=love` state, and answers: *"Yes! You love playing football."*

No explicit link was ever made between the question and the stored memory — the knowledge graph makes the connection automatically.

### What Happens When Facts Change

*"I moved from San Francisco to New York"* — the old state is **superseded**, not deleted or duplicated. Full history is preserved:

```
(State: city=New York, active=true) ──SUPERSEDES──▶ (State: city=San Francisco, active=false)
```

Ask *"Where do I live?"* → gets the current answer: New York.  
Ask *"Where was I living when I joined Google?"* → still knows it was San Francisco.

**This is what persistent memory actually means.** Not similarity search — structured, evolving knowledge with a complete audit trail.

### 3-Lane Retrieval: Why the Right Memory Surfaces

Most systems rank by vector similarity alone. That works for a week. Over years, your "I got married" memory gets buried under thousands of newer, more recent entries.

Threelane scores every candidate across **three lanes** simultaneously:

```
score = 0.65 × vector_similarity + 0.30 × importance + 0.05 × recency
```

- **Vector similarity** — semantic relevance via ANN search
- **Importance** — LLM-assigned at ingestion (landmark life events score higher)
- **Recency** — exponential decay with a 365-day half-life

The key: memories above importance 0.75 **bypass recency decay entirely**. Your wedding, your child's birth, a cancer diagnosis — these surface instantly no matter how old they are. Yesterday's lunch order won't outrank them.

### Entity Resolution: One Identity, Many Names

*"Google"*, *"my company"*, *"the office"*, *"work"* — all resolve to the same canonical entity through 3-layer matching:

1. **Case-insensitive exact** — instant lookup
2. **Substring containment** — "my company Google" → Google
3. **Embedding similarity** — cosine ≥ 0.88 catches semantic equivalents

### Designed to Last a Lifetime

Most memory systems assume weeks of data. Threelane is engineered for a **70-year lifespan**:

| What could go wrong | How Threelane handles it |
|---|---|
| Graph grows to millions of nodes | Dynamic candidate pool scales with size (2% of episodes, 50–500) |
| Old memories become noise | Consolidation engine LLM-summarizes old low-importance episodes |
| Entity names fragment over years | Background 3-pass deduplication merges fragmented nodes |
| Facts become outdated | SUPERSEDES chain — current state always queryable, history preserved |
| Embedding models get replaced | Model version tags + batch reindex utility |
| Disaster strikes | Full JSON backup with timestamped exports |

**Store a memory on day one. Query it on day 25,550 (year 70). It's still there, still accurate, still in context.**

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
store("I love playing football", speaker="ankesh")
store("I work at Google as a software engineer", speaker="ankesh")

# Query — even months later, across different conversations
answer = query("My friend invited me to play football, should I go?", speaker="ankesh")
print(answer)  # "Yes, you should! You love playing football."

close()
```

### CLI

```bash
# Interactive chat
threelane-memory chat --speaker ankesh

# Store a single memory
threelane-memory store "I love playing football" --speaker ankesh

# Query — connects to stored knowledge automatically
threelane-memory query "My friend invited me to play football, should I go?" --speaker ankesh

# Show active provider, models & run health checks
threelane-memory config

# Admin operations
threelane-memory backup --speaker ankesh
threelane-memory dedup
threelane-memory dedup --dry-run
threelane-memory consolidate --speaker ankesh
threelane-memory reindex --speaker ankesh --old-model nomic-embed-text
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
