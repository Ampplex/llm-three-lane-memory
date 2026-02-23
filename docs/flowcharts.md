# Long-Term Memory System — Detailed Flowcharts

> Complete module-by-module architecture reference with high-level overviews and low-level line-flow diagrams.

---

## Table of Contents

1. [Neo4j Graph Schema](#1-neo4j-graph-schema)
2. [config.py — Configuration](#2-configpy--configuration)
3. [schemas.py — Type Definitions](#3-schemaspy--type-definitions)
4. [database.py — Neo4j Driver](#4-databasepy--neo4j-driver)
5. [embeddings.py — Embedding Wrapper](#5-embeddingspy--embedding-wrapper)
6. [llm_interface.py — LLM Wrapper](#6-llm_interfacepy--llm-wrapper)
7. [operator.py — Semantic Extraction](#7-operatorpy--semantic-extraction)
8. [reconciler.py — Graph Writer](#8-reconcilerpy--graph-writer)
9. [retriever.py — Smart Retrieval](#9-retrieverpy--smart-retrieval)
10. [entity_dedup.py — Entity Deduplication](#10-entity_deduppy--entity-deduplication)
11. [backup.py — Graph Export](#11-backuppy--graph-export)
12. [chat.py — CLI Chat Interface](#12-chatpy--cli-chat-interface)
13. [streamlit_app.py — Streamlit UI](#13-streamlit_apppy--streamlit-ui)
14. [End-to-End Flow: Storing a Memory](#14-end-to-end-flow-storing-a-memory)
15. [End-to-End Flow: Querying a Memory](#15-end-to-end-flow-querying-a-memory)

---

## 1. Neo4j Graph Schema

### High-Level Overview

The graph stores episodic memories as a rich knowledge graph. Each user message becomes an `Episode` node connected to `Entity`, `Action`, `State`, `Role`, and `Location` nodes.

### Schema Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                        NEO4J GRAPH SCHEMA                            │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐  INVOLVES    ┌──────────┐  ALIAS_OF   ┌──────────┐    │
│  │ Episode  │─────────────▶│  Entity  │────────────▶│  Entity  │    │
│  │          │              │ (alias)  │             │ (canon)  │    │
│  │ id       │              │          │             │          │    │
│  │ summary  │              │ name     │             │ name     │    │
│  │ raw_text │              └────┬─────┘             └──────────┘    │
│  │ emotion  │                   │                                    │
│  │ importance│                  │ HAS_ROLE                           │
│  │ timestamp │                  ▼                                    │
│  │ speaker  │              ┌──────────┐                              │
│  │ embedding│              │   Role   │                              │
│  │ embed_   │              │ name     │                              │
│  │  model   │              └──────────┘                              │
│  │consolid- │                                                        │
│  │  ated    │  HAS_ACTION  ┌──────────┐  BY_ENTITY  ┌──────────┐   │
│  │consolid- │─────────────▶│  Action  │────────────▶│  Entity  │   │
│  │ ated_into│              │ verb     │             │ (actor)  │   │
│  │source_   │              │          │──ON_ENTITY─▶│  Entity  │   │
│  │  count   │              └──────────┘             │ (object) │   │
│  └────┬─────┘                                       └──────────┘   │
│       │                                                              │
│       │  HAS_STATE   ┌──────────┐  OF_ENTITY   ┌──────────┐        │
│       ├─────────────▶│  State   │─────────────▶│  Entity  │        │
│       │              │ attribute│              └──────────┘        │
│       │              │ value    │                                    │
│       │              │ active   │  SUPERSEDES   ┌──────────┐        │
│       │              │created_at│──────────────▶│State(old)│        │
│       │              │supersed- │              │active=   │        │
│       │              │  ed_at   │              │  false   │        │
│       │              └──────────┘              └──────────┘        │
│       │                                                              │
│       │  AT_LOCATION ┌──────────┐                                   │
│       └─────────────▶│ Location │                                   │
│                      │ name     │                                   │
│                      └──────────┘                                   │
│                                                                       │
│  Vector Index: episode_embedding ON Episode.embedding (1536-dim)     │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Relationships Summary

| Relationship | From | To | Purpose |
|---|---|---|---|
| `INVOLVES` | Episode | Entity | Which entities are mentioned |
| `HAS_ACTION` | Episode | Action | What happened |
| `BY_ENTITY` | Action | Entity | Who performed the action |
| `ON_ENTITY` | Action | Entity | Who/what received the action |
| `HAS_STATE` | Episode | State | Attribute-value facts |
| `OF_ENTITY` | State | Entity | Which entity the state belongs to |
| `SUPERSEDES` | State(new) | State(old) | Contradiction chain |
| `HAS_ROLE` | Entity | Role | Roles of entities |
| `AT_LOCATION` | Episode | Location | Where it happened |
| `ALIAS_OF` | Entity(alias) | Entity(canon) | Entity deduplication |

---

## 2. config.py — Configuration

### High-Level Overview

Central configuration hub. Loads all settings from `.env` file via `dotenv`. Contains Neo4j credentials, OpenAI credentials, retrieval scoring weights, dynamic candidate pool settings, consolidation thresholds, and backup directory.

### Flow

```
┌─────────────────────────────────────────────────┐
│                   config.py                      │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌───────────────┐                               │
│  │  load_dotenv() │  ◄── reads .env file         │
│  └───────┬───────┘                               │
│          ▼                                        │
│  ┌───────────────────────────────────────┐       │
│  │  Neo4j Config                          │       │
│  │  ├── NEO4J_URI                        │       │
│  │  ├── NEO4J_USER                       │       │
│  │  └── NEO4J_PASSWORD                   │       │
│  └───────────────────────────────────────┘       │
│          ▼                                        │
│  ┌───────────────────────────────────────┐       │
│  │  OpenAI Config                         │       │
│  │  ├── OPENAI_API_KEY                   │       │
│  │  ├── OPENAI_CHAT_MODEL                │ gpt-4o│
│  │  └── OPENAI_EMBED_MODEL               │ada-002│
│  │  ├── EMBEDDING_DIM = 1536             │       │
│  │  └── EMBEDDING_MODEL_VERSION          │       │
│  └───────────────────────────────────────┘       │
│          ▼                                        │
│  ┌───────────────────────────────────────┐       │
│  │  Retrieval Scoring Weights             │       │
│  │  ├── WEIGHT_SIMILARITY    = 0.65      │       │
│  │  ├── WEIGHT_IMPORTANCE    = 0.30      │       │
│  │  ├── WEIGHT_RECENCY       = 0.05      │       │
│  │  ├── RECENCY_HALF_LIFE    = 365 days  │       │
│  │  └── IMPORTANCE_FLOOR     = 0.75      │       │
│  └───────────────────────────────────────┘       │
│          ▼                                        │
│  ┌───────────────────────────────────────┐       │
│  │  Dynamic Candidate Pool                │       │
│  │  ├── VECTOR_CANDIDATES_MIN  = 50      │       │
│  │  ├── VECTOR_CANDIDATES_MAX  = 500     │       │
│  │  └── VECTOR_CANDIDATES_RATIO = 0.02   │       │
│  └───────────────────────────────────────┘       │
│          ▼                                        │
│  ┌───────────────────────────────────────┐       │
│  │  Consolidation Settings                │       │
│  │  ├── CONSOLIDATION_AGE_DAYS     = 90  │       │
│  │  ├── CONSOLIDATION_BATCH_SIZE   = 50  │       │
│  │  └── CONSOLIDATION_IMPORTANCE_CAP=0.3 │       │
│  └───────────────────────────────────────┘       │
│          ▼                                        │
│  ┌───────────────────────────────────────┐       │
│  │  BACKUP_DIR = ../backups               │       │
│  └───────────────────────────────────────┘       │
│                                                   │
└─────────────────────────────────────────────────┘
```

---

## 3. schemas.py — Type Definitions

### High-Level Overview

Defines strict TypedDict schemas for the GSW-style semantic extraction output. Every extracted memory conforms to `SemanticExtraction`.

### Flow

```
┌──────────────────────────────────────────────────┐
│                  schemas.py                       │
├──────────────────────────────────────────────────┤
│                                                    │
│  SemanticExtraction (TypedDict)                   │
│  ├── summary: str                                 │
│  ├── emotion: str                                 │
│  ├── importance: float (0.0 – 1.0)               │
│  ├── entities: List[str]                          │
│  ├── roles: List[EntityRole]                      │
│  │    └── EntityRole                              │
│  │         ├── entity: str                        │
│  │         └── role: str                          │
│  ├── actions: List[ActionItem]                    │
│  │    └── ActionItem                              │
│  │         ├── actor: str                         │
│  │         ├── verb: str                          │
│  │         └── object: Optional[str]              │
│  ├── states: List[StateItem]                      │
│  │    └── StateItem                               │
│  │         ├── entity: str                        │
│  │         ├── attribute: str                     │
│  │         └── value: str                         │
│  ├── location: Optional[str]                      │
│  └── time: Optional[str]                          │
│                                                    │
└──────────────────────────────────────────────────┘
```

---

## 4. database.py — Neo4j Driver

### High-Level Overview

Thin wrapper around the Neo4j Python driver. Provides a singleton driver instance, a `run_query()` helper, and a `close()` function. Suppresses Neo4j property-not-exist warnings.

### Flow

```
┌──────────────────────────────────────────────────────────────┐
│                       database.py                             │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  Module Load (import time)                  │               │
│  │  1. Set neo4j logger → ERROR level          │               │
│  │  2. Create GraphDatabase.driver(            │               │
│  │       NEO4J_URI, auth=(...),                │               │
│  │       notifications_min_severity="OFF"      │               │
│  │     )                                       │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  run_query(cypher: str, params: dict)       │               │
│  │  ┌─────────────────────────────────────┐   │               │
│  │  │  driver.session()                    │   │               │
│  │  │    └── session.run(cypher, params)   │   │               │
│  │  │         └── result.data()            │   │               │
│  │  │              └── return list[dict]   │   │               │
│  │  └─────────────────────────────────────┘   │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  close()                                    │               │
│  │  └── driver.close()                         │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. embeddings.py — Embedding Wrapper

### High-Level Overview

Wraps OpenAI's `text-embedding-ada-002` model via LangChain. Provides `embed()` for generating 1536-dim vectors and `cosine_similarity()` for comparing them.

### Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      embeddings.py                            │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  Module Load                                │               │
│  │  1. Validate OPENAI_API_KEY exists          │               │
│  │     → raise ValueError if missing           │               │
│  │  2. Create OpenAIEmbeddings client          │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  embed(text: str) → list[float]             │               │
│  │                                             │               │
│  │  text ──▶ empty/blank?                      │               │
│  │           ├── YES → return [0.0] * 1536     │               │
│  │           └── NO                            │               │
│  │                ▼                            │               │
│  │           openai_embeddings                 │               │
│  │             .embed_query(text)              │               │
│  │                ▼                            │               │
│  │           ┌── success? ──┐                  │               │
│  │           │ YES          │ NO               │               │
│  │           ▼              ▼                  │               │
│  │      return vector   print error            │               │
│  │      (1536-dim)      return [0.0]*1536      │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  cosine_similarity(vec1, vec2) → float      │               │
│  │                                             │               │
│  │  v1 = np.asarray(vec1)                      │               │
│  │  v2 = np.asarray(vec2)                      │               │
│  │  n1 = norm(v1), n2 = norm(v2)               │               │
│  │                                             │               │
│  │  n1==0 or n2==0?                            │               │
│  │    ├── YES → return 0.0                     │               │
│  │    └── NO  → return dot(v1,v2) / (n1*n2)   │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. llm_interface.py — LLM Wrapper

### High-Level Overview

Wraps OpenAI's `gpt-4o` model via LangChain's `ChatOpenAI`. Provides `invoke_llm()` for raw prompts and `ask_llm()` for memory-context Q&A.

### Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    llm_interface.py                            │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  Module Load                                │               │
│  │  1. Validate credentials                    │               │
│  │  2. Create ChatOpenAI(                      │               │
│  │       deployment = gpt-4o,                  │               │
│  │       temperature = 0.15,                   │               │
│  │       streaming = False                     │               │
│  │     )                                       │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  invoke_llm(prompt: str) → str              │               │
│  │                                             │               │
│  │  prompt ──▶ client.invoke([{                │               │
│  │                "role": "user",              │               │
│  │                "content": prompt            │               │
│  │             }])                             │               │
│  │          ──▶ response.content               │               │
│  │          ──▶ return str                     │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  ask_llm(context: str, question: str) → str │               │
│  │                                             │               │
│  │  Build prompt:                              │               │
│  │    "Answer ONLY using the memory below."    │               │
│  │    + context + question                     │               │
│  │  ──▶ invoke_llm(prompt)                     │               │
│  │  ──▶ return str                             │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. operator.py — Semantic Extraction

### High-Level Overview

The "Semantic Operator" sends user text to GPT-4o with a carefully crafted prompt that extracts structured event semantics (entities, actions, states, roles, emotions, importance, location, time) as JSON conforming to `SemanticExtraction`.

### Flow

```
┌──────────────────────────────────────────────────────────────┐
│                      operator.py                              │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  OPERATOR_PROMPT (constant)                 │               │
│  │  • Defines JSON output schema               │               │
│  │  • CRITICAL RULES:                          │               │
│  │    - summary must have ALL concrete facts   │               │
│  │    - entities must be specific names         │               │
│  │    - states must capture every attr-value    │               │
│  │    - importance: 0.1=casual → 0.9=major     │               │
│  │    - Return ONLY valid JSON                 │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ┌────────────────────────────────────────────┐               │
│  │  operator_extract(text: str)                │               │
│  │         → SemanticExtraction                │               │
│  │                                             │               │
│  │  text                                       │               │
│  │    │                                        │               │
│  │    ▼                                        │               │
│  │  Build prompt = OPERATOR_PROMPT + text      │               │
│  │    │                                        │               │
│  │    ▼                                        │               │
│  │  response = invoke_llm(prompt)              │               │
│  │    │                                        │               │
│  │    ▼                                        │               │
│  │  Strip markdown fences?                     │               │
│  │    │ starts with ```?                       │               │
│  │    ├── YES → remove ``` lines               │               │
│  │    └── NO  → keep as-is                     │               │
│  │    │                                        │               │
│  │    ▼                                        │               │
│  │  json.loads(cleaned)                        │               │
│  │    │                                        │               │
│  │    ├── Success → return SemanticExtraction   │               │
│  │    └── JSONDecodeError                      │               │
│  │         └── raise ValueError(...)           │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. reconciler.py — Graph Writer

### High-Level Overview

The core graph writer. Takes a `SemanticExtraction` and writes it into Neo4j as an interconnected subgraph. Includes 3-layer entity resolution, state contradiction resolution with SUPERSEDES chains, memory consolidation, and embedding re-indexing.

### 8.1 `_build_searchable_text()`

```
┌──────────────────────────────────────────────────────────────┐
│  _build_searchable_text(semantics) → str                      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  parts = [summary]                                            │
│    │                                                           │
│    ├── + each entity name                                     │
│    ├── + "{entity} is {role}" for each role                   │
│    ├── + "{actor} {verb} {object}" for each action            │
│    ├── + "{entity} {attribute} is {value}" for each state     │
│    └── + "at {location}" if present                           │
│    │                                                           │
│    ▼                                                           │
│  return ". ".join(parts)                                      │
│                                                                │
│  Purpose: Creates a single rich text string for embedding     │
│           so the vector captures ALL semantic content.         │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 `create_episode()`

```
┌──────────────────────────────────────────────────────────────┐
│  create_episode(semantics, speaker, raw_text) → episode_id    │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Generate episode_id = "ep_" + uuid[:10]                   │
│  2. searchable = _build_searchable_text(semantics)            │
│  3. vector = embed(searchable)           ◄── 1536-dim         │
│  4. Cypher: CREATE (ep:Episode {                              │
│       id, summary, raw_text, emotion, importance,             │
│       timestamp: datetime(),                                  │
│       speaker, embedding: vector,                             │
│       embedding_model: EMBEDDING_MODEL_VERSION                │
│     })                                                        │
│  5. return episode_id                                         │
│                                                                │
│  ┌──────────────────────────────────────────┐                 │
│  │  embedding_model tag enables future      │                 │
│  │  migration when model changes (70-year   │                 │
│  │  safety via reindex_embeddings())        │                 │
│  └──────────────────────────────────────────┘                 │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 8.3 `merge_entity()` — 3-Layer Entity Resolution

```
┌──────────────────────────────────────────────────────────────┐
│  merge_entity(name: str) → canonical_name                     │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  name ──▶ name_lower = name.strip().lower()                   │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════╗                  │
│  ║  LAYER 1: Exact Match (case-insensitive) ║                  │
│  ╚═══════════════╤══════════════════════════╝                  │
│    Cypher: MATCH (e:Entity)                                   │
│            WHERE toLower(e.name) = $name_lower                │
│            RETURN e.name LIMIT 1                              │
│    │                                                           │
│    ├── MATCH FOUND                                            │
│    │   canonical = existing e.name                            │
│    │   if canonical ≠ name:                                   │
│    │     MERGE (alias:Entity {name}) -[:ALIAS_OF]-> (canon)   │
│    │   return canonical                                       │
│    │                                                           │
│    └── NO MATCH → continue                                    │
│         │                                                      │
│         ▼                                                      │
│  ╔══════════════════════════════════════════╗                  │
│  ║  LAYER 2: Substring Containment          ║                  │
│  ╚═══════════════╤══════════════════════════╝                  │
│    Cypher: MATCH (e:Entity)                                   │
│            WHERE toLower(e.name) CONTAINS $name_lower         │
│               OR $name_lower CONTAINS toLower(e.name)         │
│            RETURN e.name LIMIT 5                              │
│    │                                                           │
│    ├── HITS FOUND                                             │
│    │   canonical = longest name (most specific)               │
│    │   e.g. "Jeff" → matched → "Jeffrey Epstein"              │
│    │   MERGE alias -[:ALIAS_OF]-> canon                      │
│    │   return canonical                                       │
│    │                                                           │
│    └── NO HITS → continue                                     │
│         │                                                      │
│         ▼                                                      │
│  ╔══════════════════════════════════════════╗                  │
│  ║  LAYER 3: Embedding Similarity ≥ 0.88    ║                  │
│  ╚═══════════════╤══════════════════════════╝                  │
│    Fetch all canonical entities (no ALIAS_OF outgoing)        │
│    For each: embed(entity_name), cosine_sim with embed(name)  │
│    │                                                           │
│    ├── best_sim ≥ 0.88                                        │
│    │   e.g. "Mom" → "Sunita Kumar" (sim=0.91)                │
│    │   MERGE alias -[:ALIAS_OF]-> canon                      │
│    │   return canonical                                       │
│    │                                                           │
│    └── best_sim < 0.88                                        │
│         │                                                      │
│         ▼                                                      │
│  ╔══════════════════════════════════════════╗                  │
│  ║  LAYER 4: Create New Canonical Entity     ║                  │
│  ╚═══════════════╤══════════════════════════╝                  │
│    MERGE (e:Entity {name: $name})                             │
│    return name                                                │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 8.4 `create_state()` — Contradiction Resolution

```
┌──────────────────────────────────────────────────────────────┐
│  create_state(state_item, episode_id) → void                  │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  state_item = {entity, attribute, value}                      │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════════╗              │
│  ║  Step 1: Deactivate old conflicting states    ║              │
│  ╚═══════════════╤══════════════════════════════╝              │
│    Cypher:                                                    │
│      MATCH (e:Entity {name})<-[:OF_ENTITY]-(s:State {attr})  │
│      WHERE s.active <> false                                  │
│      SET s.active = false, s.superseded_at = datetime()       │
│      RETURN elementId(s) AS old_id                            │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════════╗              │
│  ║  Step 2: Create new active State node         ║              │
│  ╚═══════════════╤══════════════════════════════╝              │
│    Cypher:                                                    │
│      MATCH (e:Entity {name}), (ep:Episode {id})               │
│      CREATE (s:State {                                        │
│        attribute, value, active: true, created_at: datetime() │
│      })                                                       │
│      MERGE (ep)-[:HAS_STATE]->(s)                             │
│      MERGE (s)-[:OF_ENTITY]->(e)                              │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════════╗              │
│  ║  Step 3: Link SUPERSEDES chain                ║              │
│  ╚═══════════════╤══════════════════════════════╝              │
│    if old_rows exist:                                         │
│      MATCH (new_s:State {attr, active:true})                  │
│        -[:OF_ENTITY]->(e:Entity {name})                       │
│      UNWIND old_ids                                           │
│      MATCH (old_s) WHERE elementId = oid                      │
│      MERGE (new_s)-[:SUPERSEDES]->(old_s)                     │
│                                                                │
│  Result: Full audit trail, only latest value is active        │
│                                                                │
│  Example:                                                     │
│    "age=20" (active) -SUPERSEDES-> "age=19" (inactive)        │
│              -SUPERSEDES-> "age=18" (inactive)                │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 8.5 `reconcile()` — Main Orchestrator

```
┌──────────────────────────────────────────────────────────────┐
│  reconcile(semantics, speaker, raw_text) → episode_id         │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  semantics (SemanticExtraction)                               │
│    │                                                           │
│    ▼                                                           │
│  ┌──────────────────────────────────┐                         │
│  │ 1. create_episode()              │                         │
│  │    → episode_id                  │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 2. For each entity in semantics: │                         │
│  │    canonical = merge_entity(name)│ ◄── 3-layer resolution  │
│  │    alias_map[name] = canonical   │                         │
│  │    link_entity_to_episode(       │                         │
│  │      canonical, episode_id)      │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 3. For each role:                │                         │
│  │    Resolve entity via alias_map  │                         │
│  │    create_role(resolved_role)    │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 4. For each action:              │                         │
│  │    Resolve actor via alias_map   │                         │
│  │    Resolve object via alias_map  │                         │
│  │    create_action(resolved, ep_id)│                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 5. For each state:               │                         │
│  │    Resolve entity via alias_map  │                         │
│  │    create_state(resolved, ep_id) │ ◄── contradiction check │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 6. If location present:          │                         │
│  │    bind_location(loc, ep_id)     │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  return episode_id                                            │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 8.6 `consolidate()` — Memory Consolidation

```
┌──────────────────────────────────────────────────────────────┐
│  consolidate(speaker) → {merged: int, episode_id: str|None}   │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────┐                         │
│  │ 1. _find_consolidation_candidates│                         │
│  │    Criteria:                     │                         │
│  │    • importance ≤ 0.3            │                         │
│  │    • age > 90 days               │                         │
│  │    • NOT already consolidated    │                         │
│  │    • Limit: 50 per batch         │                         │
│  │    ORDER BY timestamp ASC        │                         │
│  └──────────┬───────────────────────┘                         │
│             │                                                  │
│        empty?                                                  │
│         ├── YES → return {merged:0, id:None}                  │
│         └── NO                                                 │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 2. _summarise_batch(candidates)  │                         │
│  │    • Format: "[timestamp] text"  │                         │
│  │    • Prompt: "Combine into ONE   │                         │
│  │      concise summary. Preserve   │                         │
│  │      all facts, names, numbers." │                         │
│  │    → summary_text (from LLM)     │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 3. Collect entities from         │                         │
│  │    candidate episodes            │                         │
│  │    (UNWIND ids, MATCH INVOLVES)  │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 4. Create consolidated Episode   │                         │
│  │    id = "ep_consolidated_" + uuid│                         │
│  │    embedding = embed(summary +   │                         │
│  │      entity_names)               │                         │
│  │    importance = avg + 0.1        │                         │
│  │    consolidated = true           │                         │
│  │    source_count = N              │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 5. Link to same entities         │                         │
│  │    MERGE (ep)-[:INVOLVES]->(e)   │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ 6. Mark originals:               │                         │
│  │    SET ep.consolidated = true    │                         │
│  │    SET ep.consolidated_into =    │                         │
│  │        consolidated_episode_id   │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  return {merged: N, id: ep_consolidated_id}                   │
│                                                                │
│  Note: Originals are kept for audit but excluded from         │
│  retrieval by: WHERE ep.consolidated_into IS NULL             │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 8.7 `reindex_embeddings()` — Embedding Migration

```
┌──────────────────────────────────────────────────────────────┐
│  reindex_embeddings(speaker, old_model, batch_size=100) → int │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  total = 0                                                    │
│    │                                                           │
│    ▼                                                           │
│  ┌────── LOOP ──────────────────────────────────┐             │
│  │  Cypher: MATCH (ep:Episode {speaker})         │             │
│  │  WHERE ep.embedding_model = $old_model        │             │
│  │     OR ep.embedding_model IS NULL             │             │
│  │  RETURN id, summary, raw_text LIMIT batch     │             │
│  │                                               │             │
│  │  empty? ── YES ──▶ break                      │             │
│  │    │                                          │             │
│  │    ▼ NO                                       │             │
│  │  For each episode:                            │             │
│  │    text = raw_text OR summary                 │             │
│  │    new_vec = embed(text)                      │             │
│  │    SET ep.embedding = new_vec                 │             │
│  │    SET ep.embedding_model = CURRENT_VERSION   │             │
│  │                                               │             │
│  │  total += batch_count                         │             │
│  └───────────────────────────────────────────────┘             │
│    │                                                           │
│    ▼                                                           │
│  return total                                                 │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. retriever.py — Smart Retrieval

### High-Level Overview

3-lane retrieval engine designed for 70-year lifespans. Combines recent-episode safety net, temporal date-range queries, and vector ANN search with importance-weighted re-ranking.

### Scoring Formula

```
score = 0.65 × similarity + 0.30 × importance + 0.05 × recency

• If importance ≥ 0.75 (IMPORTANCE_FLOOR) → recency = 1.0 (never penalised)
• Recency decay: e^(-0.693 × age_days / 365)
• 10-year-old memory: recency ≈ 0.001
• But with 5% weight: minimal impact on score
```

### 9.1 `_recency_weight()`

```
┌──────────────────────────────────────────────────────────────┐
│  _recency_weight(ts_iso: str) → float (0-1)                  │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ts_iso                                                       │
│    │                                                           │
│    ├── None? → return 0.5                                     │
│    └── parse ISO string                                       │
│         │                                                      │
│         ├── parse error? → return 0.5                         │
│         └── success                                           │
│              │                                                 │
│              ▼                                                 │
│         age_days = (now - ts) / 86400                         │
│         return e^(-0.693 × age_days / 365)                    │
│                                                                │
│  Examples:                                                    │
│    today     → 1.0                                            │
│    1 year    → 0.5                                            │
│    10 years  → 0.001                                          │
│    70 years  → ~0.0                                           │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 9.2 `_combined_score()`

```
┌──────────────────────────────────────────────────────────────┐
│  _combined_score(sim, importance, recency) → float            │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  importance ≥ IMPORTANCE_FLOOR (0.75)?                        │
│    ├── YES → recency = 1.0 (landmark memory bypass)          │
│    └── NO  → recency unchanged                               │
│                                                                │
│  return 0.65 × sim + 0.30 × importance + 0.05 × recency      │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 9.3 `_dynamic_candidates()`

```
┌──────────────────────────────────────────────────────────────┐
│  _dynamic_candidates(speaker) → int                           │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  total = COUNT(Episode {speaker})                             │
│  candidates = total × 0.02                                    │
│  return clamp(candidates, min=50, max=500)                    │
│                                                                │
│  Examples:                                                    │
│    100 episodes  → 50  (clamped to min)                       │
│    5000 episodes → 100                                        │
│    25000 episodes → 500 (clamped to max)                      │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 9.4 `_extract_time_range()`

```
┌──────────────────────────────────────────────────────────────┐
│  _extract_time_range(text) → (start_iso, end_iso) | None      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  text (lowered)                                               │
│    │                                                           │
│    ▼                                                           │
│  Try: "last/past N day/week/month/year(s)"                    │
│    REGEX: (?:last|past)\s+(\d+)\s+(day|week|month|year)s?    │
│    │                                                           │
│    ├── MATCH → convert to days                                │
│    │   day=1, week=7, month=30, year=365                     │
│    │   start = now - N×unit_days                              │
│    │   return (start, now)                                    │
│    │                                                           │
│    └── NO MATCH                                               │
│         │                                                      │
│         ▼                                                      │
│  Try: Explicit year (19xx or 20xx)                            │
│    REGEX: \b(19|20)\d{2}\b                                    │
│    │                                                           │
│    ├── MATCH → year extracted                                 │
│    │   Check for month name ("january", "feb", etc.)          │
│    │   start = Jan 1 of year (or month 1)                    │
│    │   end = Dec 31 of year (or month end)                   │
│    │   return (start, end)                                    │
│    │                                                           │
│    └── NO MATCH → return None                                 │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 9.5 `find_relevant_episodes()` — 3-Lane Merge

```
┌──────────────────────────────────────────────────────────────┐
│  find_relevant_episodes(question, speaker, top_k=8)           │
│         → list[episode_id]                                    │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  question                                                     │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════╗                  │
│  ║ LANE 0: Recent Safety Net (5 min)        ║                  │
│  ╚═══════════════╤══════════════════════════╝                  │
│    Cypher: ep.timestamp >= datetime() - 5min                  │
│    → ids_recent                                               │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════╗                  │
│  ║ LANE 1: Temporal Query                    ║                  │
│  ╚═══════════════╤══════════════════════════╝                  │
│    time_range = _extract_time_range(question)                 │
│    ├── present? → _temporal_episode_ids(range)                │
│    │             → ids_from_time                              │
│    └── absent?  → ids_from_time = []                          │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════╗                  │
│  ║ LANE 2: Vector ANN Search                ║                  │
│  ╚═══════════════╤══════════════════════════╝                  │
│    q_vec = embed(question)                                    │
│    candidates = _dynamic_candidates(speaker)                  │
│    │                                                           │
│    Cypher: db.index.vector.queryNodes(                        │
│              'episode_embedding', candidates, q_vec)          │
│    YIELD node, score WHERE node.speaker = speaker             │
│    │                                                           │
│    For each result:                                           │
│      skip if sim < 0.35 (SIMILARITY_THRESHOLD × 0.5)         │
│      recency = _recency_weight(ts)                            │
│      combined = _combined_score(sim, importance, recency)      │
│      keep if combined ≥ 0.20 (MIN_SCORE)                      │
│    │                                                           │
│    Sort by combined score DESC                                │
│    → ids_from_vector[:top_k]                                  │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════╗                  │
│  ║ MERGE (deduplicated, recent first)        ║                  │
│  ╚═══════════════╤══════════════════════════╝                  │
│    merged = deduplicate(                                      │
│      ids_recent + ids_from_time + ids_from_vector             │
│    )                                                          │
│    return merged[:top_k]                                      │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 9.6 `expand_episodes()` — Subgraph Expansion

```
┌──────────────────────────────────────────────────────────────┐
│  expand_episodes(episode_ids) → formatted context string      │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  episode_ids                                                  │
│    │                                                           │
│    ▼                                                           │
│  ┌── 6 Cypher Queries (in parallel) ─────────────────────┐   │
│  │                                                         │   │
│  │  Q1: Episodes                                          │   │
│  │      WHERE consolidated_into IS NULL                   │   │
│  │      → id, summary, raw_text, emotion, importance, ts  │   │
│  │                                                         │   │
│  │  Q2: Entities (follows ALIAS_OF → canonical)           │   │
│  │      OPTIONAL MATCH (e)-[:ALIAS_OF]->(canon)           │   │
│  │      → ep_id, coalesce(canon.name, e.name) AS entity   │   │
│  │                                                         │   │
│  │  Q3: Actions                                           │   │
│  │      HAS_ACTION → BY_ENTITY, ON_ENTITY                │   │
│  │      → ep_id, actor, verb, object                     │   │
│  │                                                         │   │
│  │  Q4: States (active only, alias-aware)                 │   │
│  │      WHERE coalesce(s.active, true) <> false           │   │
│  │      OPTIONAL MATCH (e)-[:ALIAS_OF]->(canon)           │   │
│  │      → ep_id, entity, attribute, value                 │   │
│  │                                                         │   │
│  │  Q5: Roles (for involved entities + their aliases)     │   │
│  │      → entity, role                                    │   │
│  │                                                         │   │
│  │  Q6: Locations                                         │   │
│  │      AT_LOCATION → loc.name                            │   │
│  │      → ep_id, location                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│    │                                                           │
│    ▼                                                           │
│  ┌── Format Output ──────────────────────────────────────┐    │
│  │  For each episode:                                     │    │
│  │    [timestamp] raw_text or summary                     │    │
│  │    (summary=..., emotion=..., importance=...)          │    │
│  │    Entities: entity1, entity2                          │    │
│  │    Action: actor verb → object                         │    │
│  │    State: entity.attr = value                          │    │
│  │    Location: loc_name                                  │    │
│  │                                                        │    │
│  │  Roles:                                                │    │
│  │    entity → role                                       │    │
│  └────────────────────────────────────────────────────────┘    │
│    │                                                           │
│    ▼                                                           │
│  return "\n".join(lines)                                      │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### 9.7 `retrieve()` — Public API

```
┌──────────────────────────────────────────────────────────────┐
│  retrieve(question, speaker, top_k=8) → str                   │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  question ──▶ find_relevant_episodes(question, speaker, top_k)│
│            ──▶ episode_ids                                    │
│            ──▶ expand_episodes(episode_ids)                   │
│            ──▶ return formatted context string                │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. entity_dedup.py — Entity Deduplication

### High-Level Overview

Background utility that scans all Entity nodes and merges duplicates found via 3 passes: case-insensitive exact match, substring containment, and embedding similarity. Migrates all relationships from alias to canonical entity.

### Flow

```
┌──────────────────────────────────────────────────────────────┐
│  deduplicate_entities(dry_run=True) → {found, merged}         │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  entities = all canonical entities                            │
│    (WHERE NOT exists((e)-[:ALIAS_OF]->()))                    │
│  names = [e.name for each]                                    │
│  already_merged = set()                                       │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════════╗              │
│  ║  Pass 1: Case-Insensitive Exact Match        ║              │
│  ╚═══════════════╤══════════════════════════════╝              │
│    Group names by name.strip().lower()                        │
│    For groups with >1 member:                                 │
│      canonical = longest name                                 │
│      others → aliases                                         │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════════╗              │
│  ║  Pass 2: Substring Containment               ║              │
│  ╚═══════════════╤══════════════════════════════╝              │
│    For remaining (not yet merged) names:                      │
│      Pairwise check: a.lower() in b.lower()                  │
│        or b.lower() in a.lower()                              │
│      canonical = longer name                                  │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════════╗              │
│  ║  Pass 3: Embedding Similarity ≥ 0.88         ║              │
│  ╚═══════════════╤══════════════════════════════╝              │
│    For remaining names:                                       │
│      Embed each name                                          │
│      Pairwise cosine similarity                               │
│      If sim ≥ 0.88: longer name = canonical                   │
│    │                                                           │
│    ▼                                                           │
│  ╔══════════════════════════════════════════════╗              │
│  ║  Apply Merges                                 ║              │
│  ╚═══════════════╤══════════════════════════════╝              │
│    For each (alias, canonical) pair:                          │
│      _create_alias(alias, canonical, dry_run)                 │
│    │                                                           │
│    ▼                                                           │
│  return {duplicates_found, merged}                            │
│                                                                │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  _create_alias(alias_name, canonical_name, dry_run) → void    │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  dry_run?                                                     │
│  ├── YES → print "[DRY-RUN] Would alias ..."                 │
│  └── NO                                                       │
│       │                                                        │
│       ▼                                                        │
│    1. MERGE (alias)-[:ALIAS_OF]->(canon)                      │
│    2. Migrate INVOLVES: Episode→alias → also Episode→canon   │
│    3. Migrate BY_ENTITY: Action→alias → also Action→canon    │
│    4. Migrate ON_ENTITY: Action→alias → also Action→canon    │
│    5. Migrate OF_ENTITY: State→alias → also State→canon      │
│    6. Migrate HAS_ROLE: alias→Role → also canon→Role         │
│                                                                │
│    print "✅ Aliased 'X' → 'Y' (relationships migrated)"     │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 11. backup.py — Graph Export

### High-Level Overview

Full graph export utility. Dumps all nodes and relationships to timestamped JSON files. Supports filtering by speaker and since-date.

### Flow

```
┌──────────────────────────────────────────────────────────────┐
│  save_backup(speaker?, since?) → filepath                     │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────┐                         │
│  │ _ensure_dir(BACKUP_DIR)          │                         │
│  │ os.makedirs(path, exist_ok=True) │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ full_export(speaker, since)      │                         │
│  │                                  │                         │
│  │  ┌─ export_episodes() ──────────┐│                         │
│  │  │ Optional WHERE:              ││                         │
│  │  │   ep.speaker = $speaker      ││                         │
│  │  │   ep.timestamp >= $since     ││                         │
│  │  │ Returns: id, summary,       ││                         │
│  │  │   raw_text, emotion,        ││                         │
│  │  │   importance, timestamp,     ││                         │
│  │  │   speaker, embedding_model,  ││                         │
│  │  │   consolidated,              ││                         │
│  │  │   consolidated_into,         ││                         │
│  │  │   source_count               ││                         │
│  │  └──────────────────────────────┘│                         │
│  │  ┌─ export_entities() ──────────┐│                         │
│  │  │ MATCH (e:Entity)             ││                         │
│  │  │ → name                       ││                         │
│  │  └──────────────────────────────┘│                         │
│  │  ┌─ export_states() ───────────┐│                         │
│  │  │ MATCH (s:State)-[:OF_ENTITY]│││                         │
│  │  │ coalesce for active          ││                         │
│  │  │ OPTIONAL SUPERSEDES          ││                         │
│  │  │ → attr, value, active,      ││                         │
│  │  │   entity, created_at,       ││                         │
│  │  │   superseded_at,            ││                         │
│  │  │   superseded_value          ││                         │
│  │  └──────────────────────────────┘│                         │
│  │  ┌─ export_roles() ────────────┐│                         │
│  │  │ MATCH (e)-[:HAS_ROLE]->(r)  ││                         │
│  │  │ → entity, role              ││                         │
│  │  └──────────────────────────────┘│                         │
│  │  ┌─ export_actions() ──────────┐│                         │
│  │  │ MATCH (ep)-[:HAS_ACTION]→   ││                         │
│  │  │   (a)-[:BY_ENTITY]->(actor) ││                         │
│  │  │ OPTIONAL ON_ENTITY → obj    ││                         │
│  │  │ → episode_id, actor,        ││                         │
│  │  │   verb, object              ││                         │
│  │  └──────────────────────────────┘│                         │
│  │  ┌─ export_locations() ────────┐│                         │
│  │  │ MATCH (ep)-[:AT_LOCATION]→  ││                         │
│  │  │ → episode_id, location      ││                         │
│  │  └──────────────────────────────┘│                         │
│  │  ┌─ export_involves() ─────────┐│                         │
│  │  │ MATCH (ep)-[:INVOLVES]->(e) ││                         │
│  │  │ → episode_id, entity        ││                         │
│  │  └──────────────────────────────┘│                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  ┌──────────────────────────────────┐                         │
│  │ Save to JSON:                    │                         │
│  │   backup_{speaker}_{timestamp}   │                         │
│  │   .json                          │                         │
│  │                                  │                         │
│  │ Print summary:                   │                         │
│  │   episodes: N                    │                         │
│  │   entities: N                    │                         │
│  │   states: N                      │                         │
│  │   roles: N                       │                         │
│  │   actions: N                     │                         │
│  │   locations: N                   │                         │
│  └──────────┬───────────────────────┘                         │
│             ▼                                                  │
│  return filepath                                              │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 12. chat.py — CLI Chat Interface

### High-Level Overview

Interactive terminal chat loop. Classifies user input as question vs. statement, routes to retrieval+LLM or extraction+storage accordingly. Supports slash commands for admin operations.

### Flow

```
┌──────────────────────────────────────────────────────────────┐
│                       chat.py                                 │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  main()                                                       │
│    │                                                           │
│    ▼                                                           │
│  Print banner                                                 │
│    │                                                           │
│    ▼                                                           │
│  ┌═══════════ MAIN LOOP ═══════════════════════════════┐      │
│  │                                                       │      │
│  │  user_input = input("You: ")                         │      │
│  │    │                                                  │      │
│  │    ├── empty?        → continue                      │      │
│  │    ├── quit/exit/q?  → break                         │      │
│  │    │                                                  │      │
│  │    ├── "/consolidate"                                │      │
│  │    │    └── consolidate(SPEAKER)                     │      │
│  │    │         ├── merged > 0 → print success          │      │
│  │    │         └── merged = 0 → print "nothing"        │      │
│  │    │                                                  │      │
│  │    ├── "/backup"                                     │      │
│  │    │    └── save_backup(speaker=SPEAKER)             │      │
│  │    │                                                  │      │
│  │    ├── "/dedup"                                      │      │
│  │    │    └── deduplicate_entities(dry_run=False)      │      │
│  │    │         ├── merged > 0 → print success          │      │
│  │    │         └── merged = 0 → print "no dupes"       │      │
│  │    │                                                  │      │
│  │    └── Regular input                                 │      │
│  │         │                                             │      │
│  │         ▼                                             │      │
│  │    is_question(text)?                                │      │
│  │    │                                                  │      │
│  │    ├── YES (Question Mode)                           │      │
│  │    │    ctx = retrieve(question, SPEAKER)             │      │
│  │    │    answer = invoke_llm(ctx + question)           │      │
│  │    │    print answer                                  │      │
│  │    │                                                  │      │
│  │    └── NO (Store Mode)                               │      │
│  │         semantics = operator_extract(text)            │      │
│  │         episode_id = reconcile(semantics, SPEAKER,    │      │
│  │                                raw_text=text)          │      │
│  │         print episode_id + summary + entities         │      │
│  │                                                       │      │
│  └═══════════════════════════════════════════════════════┘      │
│    │                                                           │
│    ▼                                                           │
│  close()  ← shutdown Neo4j driver                             │
│                                                                │
│  ┌──────────────────────────────────────────┐                 │
│  │  is_question(text) → bool                │                 │
│  │    • Ends with "?"       → True          │                 │
│  │    • Starts with:                        │                 │
│  │      what, who, where, when, why, how,   │                 │
│  │      do, did, does, is, are, was, were,  │                 │
│  │      can, could, tell me, recall,        │                 │
│  │      remember, show me   → True          │                 │
│  │    • Otherwise           → False         │                 │
│  └──────────────────────────────────────────┘                 │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 13. streamlit_app.py — Streamlit UI

### High-Level Overview

Web-based chat interface using Streamlit. Same core logic as chat.py but with a modern UI. Sidebar contains admin tools (Consolidate, Backup, Dedup, Clear History). Chat messages are persisted in `st.session_state`.

### Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   streamlit_app.py                             │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌── Page Config ───────────────────────────────────────┐     │
│  │  title: "Memory Chat"  icon: 🧠  layout: centered    │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  ┌── Session State Init ────────────────────────────────┐     │
│  │  if "messages" not in st.session_state:              │     │
│  │    st.session_state.messages = []                     │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  ┌── Sidebar ───────────────────────────────────────────┐     │
│  │  ┌─────────────────────────────────────────────┐     │     │
│  │  │ 🔄 Consolidate Memories (button)            │     │     │
│  │  │  → consolidate(SPEAKER)                     │     │     │
│  │  │  → st.success / st.info                     │     │     │
│  │  ├─────────────────────────────────────────────┤     │     │
│  │  │ 📦 Backup Graph (button)                    │     │     │
│  │  │  → save_backup(speaker=SPEAKER)             │     │     │
│  │  │  → st.success(path)                         │     │     │
│  │  ├─────────────────────────────────────────────┤     │     │
│  │  │ 🔗 Deduplicate Entities (button)            │     │     │
│  │  │  → deduplicate_entities(dry_run=False)      │     │     │
│  │  │  → st.success / st.info                     │     │     │
│  │  ├─────────────────────────────────────────────┤     │     │
│  │  │ 🗑️ Clear Chat History (button)              │     │     │
│  │  │  → st.session_state.messages = []           │     │     │
│  │  │  → st.rerun()                               │     │     │
│  │  └─────────────────────────────────────────────┘     │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  ┌── Render Chat History ───────────────────────────────┐     │
│  │  for msg in st.session_state.messages:               │     │
│  │    st.chat_message(msg["role"]).markdown(content)     │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  ┌── Chat Input ────────────────────────────────────────┐     │
│  │  user_input = st.chat_input(...)                      │     │
│  │    │                                                   │     │
│  │    ▼                                                   │     │
│  │  Append to session_state.messages                     │     │
│  │  Display user message                                 │     │
│  │    │                                                   │     │
│  │    ▼                                                   │     │
│  │  is_question(user_input)?                             │     │
│  │    │                                                   │     │
│  │    ├── YES (🔍 Searching memory…)                     │     │
│  │    │    answer = answer_question(user_input)           │     │
│  │    │    st.markdown(answer)                            │     │
│  │    │    Append to session_state.messages               │     │
│  │    │                                                   │     │
│  │    └── NO (📥 Extracting semantics…)                  │     │
│  │         semantics = operator_extract(user_input)       │     │
│  │         episode_id = reconcile(semantics, SPEAKER,     │     │
│  │                                raw_text=user_input)     │     │
│  │         Display: episode_id, summary, entities,        │     │
│  │                  emotion, importance, location          │     │
│  │         Append to session_state.messages               │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 14. End-to-End Flow: Storing a Memory

### Worked Example

**Input**: `"I met Jeffrey Epstein at a party in Manhattan last night"`

```
┌──────────────────────────────────────────────────────────────┐
│           END-TO-END: STORING A MEMORY                        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  User: "I met Jeffrey Epstein at a party in Manhattan"        │
│    │                                                           │
│    ▼                                                           │
│  ┌─ is_question() ─┐                                         │
│  │ Starts with "I"  │                                         │
│  │ No "?" at end    │                                         │
│  │ → False (STORE)  │                                         │
│  └────────┬─────────┘                                         │
│           ▼                                                    │
│  ┌─ operator_extract() ──────────────────────────────────┐    │
│  │  Input → GPT-4o with OPERATOR_PROMPT                   │    │
│  │  Output JSON:                                          │    │
│  │  {                                                     │    │
│  │    "summary": "Speaker met Jeffrey Epstein at a        │    │
│  │               party in Manhattan",                     │    │
│  │    "emotion": "neutral",                               │    │
│  │    "importance": 0.6,                                  │    │
│  │    "entities": ["Jeffrey Epstein", "Manhattan"],       │    │
│  │    "roles": [],                                        │    │
│  │    "actions": [{"actor":"speaker","verb":"met",        │    │
│  │                 "object":"Jeffrey Epstein"}],           │    │
│  │    "states": [],                                       │    │
│  │    "location": "Manhattan",                            │    │
│  │    "time": null                                        │    │
│  │  }                                                     │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           ▼                                                    │
│  ┌─ reconcile() ─────────────────────────────────────────┐    │
│  │                                                        │    │
│  │  1. create_episode()                                   │    │
│  │     • Build searchable text:                           │    │
│  │       "Speaker met Jeffrey Epstein at a party in       │    │
│  │        Manhattan. Jeffrey Epstein. Manhattan.           │    │
│  │        speaker met Jeffrey Epstein. at Manhattan"      │    │
│  │     • embed() → 1536-dim vector                        │    │
│  │     • CREATE Episode node with embedding               │    │
│  │     → ep_abc1234567                                    │    │
│  │                                                        │    │
│  │  2. Entity resolution:                                 │    │
│  │     merge_entity("Jeffrey Epstein")                    │    │
│  │       Layer 1: exact? → check toLower                  │    │
│  │         If "jeffrey epstein" exists → found!           │    │
│  │         → canonical = "Jeffrey Epstein"                │    │
│  │       OR if new → CREATE Entity node                   │    │
│  │     link_entity_to_episode("Jeffrey Epstein", ep_id)   │    │
│  │                                                        │    │
│  │     merge_entity("Manhattan")                          │    │
│  │       → canonical = "Manhattan"                        │    │
│  │     link_entity_to_episode("Manhattan", ep_id)         │    │
│  │                                                        │    │
│  │  3. Roles: (none)                                      │    │
│  │                                                        │    │
│  │  4. Actions:                                           │    │
│  │     create_action({actor:"speaker", verb:"met",        │    │
│  │                     object:"Jeffrey Epstein"}, ep_id)  │    │
│  │     • CREATE Action {verb:"met"}                       │    │
│  │     • BY_ENTITY → Entity("speaker")                    │    │
│  │     • ON_ENTITY → Entity("Jeffrey Epstein")            │    │
│  │     • Episode -[:HAS_ACTION]-> Action                  │    │
│  │                                                        │    │
│  │  5. States: (none)                                     │    │
│  │                                                        │    │
│  │  6. bind_location("Manhattan", ep_id)                  │    │
│  │     • MERGE Location {name:"Manhattan"}                │    │
│  │     • Episode -[:AT_LOCATION]-> Location               │    │
│  │                                                        │    │
│  │  return ep_abc1234567                                  │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           ▼                                                    │
│  Display: "✅ Stored episode ep_abc1234567"                   │
│           "Summary: Speaker met Jeffrey Epstein..."           │
│           "Entities: Jeffrey Epstein, Manhattan"              │
│                                                                │
│  ┌─ Resulting Subgraph ─────────────────────────────────┐    │
│  │                                                        │    │
│  │  (Episode:ep_abc1234567)                               │    │
│  │    ├──INVOLVES──▶ (Entity: Jeffrey Epstein)            │    │
│  │    ├──INVOLVES──▶ (Entity: Manhattan)                  │    │
│  │    ├──HAS_ACTION──▶ (Action: met)                      │    │
│  │    │                  ├──BY_ENTITY──▶ (Entity: speaker) │    │
│  │    │                  └──ON_ENTITY──▶ (Entity: J.E.)   │    │
│  │    └──AT_LOCATION──▶ (Location: Manhattan)             │    │
│  │                                                        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 15. End-to-End Flow: Querying a Memory

### Worked Example

**Input**: `"Who did I meet in Manhattan?"`

```
┌──────────────────────────────────────────────────────────────┐
│           END-TO-END: QUERYING A MEMORY                       │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  User: "Who did I meet in Manhattan?"                         │
│    │                                                           │
│    ▼                                                           │
│  ┌─ is_question() ─┐                                         │
│  │ Starts with "Who"│                                         │
│  │ Ends with "?"    │                                         │
│  │ → True (QUERY)   │                                         │
│  └────────┬─────────┘                                         │
│           ▼                                                    │
│  ┌─ retrieve() ──────────────────────────────────────────┐    │
│  │                                                        │    │
│  │  ┌─ find_relevant_episodes() ─────────────────────┐   │    │
│  │  │                                                  │   │    │
│  │  │  Lane 0: Recent Safety Net                      │   │    │
│  │  │    episodes from last 5 minutes                 │   │    │
│  │  │    → ids_recent (may be empty)                  │   │    │
│  │  │                                                  │   │    │
│  │  │  Lane 1: Temporal Query                         │   │    │
│  │  │    _extract_time_range("who did i meet...")      │   │    │
│  │  │    → None (no temporal cue)                     │   │    │
│  │  │    → ids_from_time = []                         │   │    │
│  │  │                                                  │   │    │
│  │  │  Lane 2: Vector ANN Search                      │   │    │
│  │  │    q_vec = embed("Who did I meet in Manhattan?")│   │    │
│  │  │    candidates = _dynamic_candidates(speaker)    │   │    │
│  │  │    → e.g. 50                                    │   │    │
│  │  │                                                  │   │    │
│  │  │    db.index.vector.queryNodes(                  │   │    │
│  │  │      'episode_embedding', 50, q_vec)            │   │    │
│  │  │    → Returns episodes ranked by cosine sim      │   │    │
│  │  │                                                  │   │    │
│  │  │    For each result:                             │   │    │
│  │  │      ep_abc1234567: sim=0.85                    │   │    │
│  │  │      recency = e^(-0.693 × 1/365) ≈ 0.998      │   │    │
│  │  │      importance = 0.6                           │   │    │
│  │  │      score = 0.65×0.85 + 0.30×0.6 + 0.05×0.998 │   │    │
│  │  │           = 0.5525 + 0.18 + 0.0499              │   │    │
│  │  │           = 0.782                               │   │    │
│  │  │      score(0.782) ≥ MIN_SCORE(0.20) ✓           │   │    │
│  │  │                                                  │   │    │
│  │  │    Sort by score DESC                           │   │    │
│  │  │    → [ep_abc1234567, ...]                       │   │    │
│  │  └───────────┬──────────────────────────────────────┘   │    │
│  │              ▼                                          │    │
│  │  ┌─ expand_episodes() ─────────────────────────────┐   │    │
│  │  │                                                  │   │    │
│  │  │  6 Cypher queries on [ep_abc1234567]:           │   │    │
│  │  │    Q1: Episode data (filtered: not consolidated) │   │    │
│  │  │    Q2: Entities → Jeffrey Epstein, Manhattan     │   │    │
│  │  │    Q3: Actions → speaker met Jeffrey Epstein    │   │    │
│  │  │    Q4: States → (none)                          │   │    │
│  │  │    Q5: Roles → (none)                           │   │    │
│  │  │    Q6: Locations → Manhattan                    │   │    │
│  │  │                                                  │   │    │
│  │  │  Format:                                        │   │    │
│  │  │    "[2026-02-22T...] I met Jeffrey Epstein..."  │   │    │
│  │  │    "  Entities: Jeffrey Epstein, Manhattan"     │   │    │
│  │  │    "  Action: speaker met → Jeffrey Epstein"    │   │    │
│  │  │    "  Location: Manhattan"                      │   │    │
│  │  └───────────┬──────────────────────────────────────┘   │    │
│  │              ▼                                          │    │
│  │  return formatted context string                       │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           ▼                                                    │
│  ┌─ invoke_llm() ────────────────────────────────────────┐    │
│  │  Prompt:                                               │    │
│  │    "You are a personal memory assistant.               │    │
│  │     Use ONLY the memory context below..."              │    │
│  │                                                        │    │
│  │    Memory Context: [formatted subgraph]                │    │
│  │    Question: "Who did I meet in Manhattan?"            │    │
│  │                                                        │    │
│  │  → GPT-4o response:                                    │    │
│  │    "You met Jeffrey Epstein at a party in Manhattan."  │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           ▼                                                    │
│  Display: "You met Jeffrey Epstein at a party in Manhattan."  │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

---

## Summary Table: Module Dependencies

```
┌──────────────────┬──────────────────────────────────────────────┐
│ Module           │ Depends On                                    │
├──────────────────┼──────────────────────────────────────────────┤
│ config.py        │ dotenv, os                                    │
│ schemas.py       │ typing                                        │
│ database.py      │ config, neo4j                                 │
│ embeddings.py    │ config, langchain_openai, numpy               │
│ llm_interface.py │ config, langchain_openai                      │
│ operator.py      │ llm_interface, schemas                        │
│ reconciler.py    │ database, embeddings, llm_interface,          │
│                  │ schemas, config                                │
│ retriever.py     │ database, embeddings, config                  │
│ entity_dedup.py  │ database, embeddings                          │
│ backup.py        │ database, config                               │
│ chat.py          │ operator, reconciler, retriever,              │
│                  │ llm_interface, backup, entity_dedup, database │
│ streamlit_app.py │ operator, reconciler, retriever,              │
│                  │ llm_interface, backup, entity_dedup           │
└──────────────────┴──────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────┐
│               DATA FLOW (HIGH LEVEL)                          │
│                                                                │
│   User Input                                                  │
│     │                                                          │
│     ├── Statement ──▶ operator.py ──▶ reconciler.py ──▶ Neo4j │
│     │                                                          │
│     └── Question  ──▶ retriever.py ──▶ Neo4j                  │
│                       ──▶ llm_interface.py                     │
│                       ──▶ Answer                               │
│                                                                │
│   Admin Commands                                              │
│     ├── /consolidate ──▶ reconciler.consolidate()             │
│     ├── /backup      ──▶ backup.save_backup()                 │
│     └── /dedup       ──▶ entity_dedup.deduplicate_entities()  │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```
