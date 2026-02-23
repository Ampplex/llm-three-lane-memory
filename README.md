# threelane-memory

A personal long-term memory system built on **Neo4j** and **OpenAI**. Stores and recalls episodic memories across a **70-year lifespan** using a hybrid 3-lane retrieval engine with vector similarity, temporal queries, and importance-weighted re-ranking.

Every input is semantically extracted into a rich knowledge graph (entities, actions, states, roles, locations) and retrieved using a scoring formula tuned for decades of data.

## Features

- **Semantic extraction** — GPT-4o parses text into structured entities, actions, states, roles, emotions, importance, and location
- **Neo4j knowledge graph** — Episodes, Entities, Actions, States, Roles, Locations with typed relationships
- **3-layer entity resolution** — case-insensitive → substring → embedding similarity (≥ 0.88)
- **State contradiction handling** — new facts `SUPERSEDE` old ones with full audit trail
- **Vector ANN search** — 1536-dim OpenAI embeddings indexed in Neo4j
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
- **OpenAI API key** (GPT-4o + text-embedding-ada-002)

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

```env
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
OPENAI_API_KEY=sk-your-api-key
```

Create the required Neo4j vector index (run once in Neo4j Browser):

```cypher
CREATE VECTOR INDEX episode_embedding IF NOT EXISTS
FOR (ep:Episode) ON (ep.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
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

# Admin operations
threelane-memory backup --speaker ankesh
threelane-memory dedup
threelane-memory dedup --dry-run
threelane-memory consolidate --speaker ankesh
threelane-memory reindex --speaker ankesh --old-model text-embedding-ada-002-v2
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
  │                  (GPT-4o)        (entity resolution,
  │                                   state contradiction,
  │                                   graph writing)
  │
  └── Question  ──▶ retriever.py ──▶ Neo4j Vector Index
                     (3-lane search)    + Graph Traversal
                    ──▶ llm_interface.py ──▶ Answer
                         (GPT-4o)
```

### Graph Schema

```
(Episode) ──INVOLVES──▶ (Entity) ──ALIAS_OF──▶ (Entity: canonical)
    │                       │
    ├──HAS_ACTION──▶ (Action) ──BY_ENTITY──▶ (Entity: actor)
    │                         ──ON_ENTITY──▶ (Entity: object)
    │
    ├──HAS_STATE──▶ (State) ──OF_ENTITY──▶ (Entity)
    │                  │
    │                  └──SUPERSEDES──▶ (State: old, active=false)
    │
    ├──AT_LOCATION──▶ (Location)
    │
    └── (Entity) ──HAS_ROLE──▶ (Role)
```

### Retrieval Scoring

```
score = 0.65 × similarity + 0.30 × importance + 0.05 × recency

• importance ≥ 0.75 → recency forced to 1.0 (landmark memories never penalised)
• recency = e^(-0.693 × age_days / 365)
• candidate pool = clamp(total_episodes × 2%, 50, 500)
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
│   ├── embeddings.py          # OpenAI embeddings + cosine similarity
│   ├── llm_interface.py       # LLM chat wrapper
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
- **OpenAI** — GPT-4o (chat) + text-embedding-ada-002 (embeddings)
- **LangChain** — ChatOpenAI & OpenAIEmbeddings wrappers
- **NumPy** — Cosine similarity computation

## License

MIT — see [LICENSE](LICENSE).
