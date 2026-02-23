"""GSW-style Reconciler – writes semantic extractions into the Neo4j workspace.

Includes:
  • **State contradiction resolution**: new states SUPERSEDE old conflicting
    ones so only the latest value is active for each entity+attribute pair.
  • **Memory consolidation**: old low-importance episodes are periodically
    merged into compact summary episodes so the graph stays manageable
    across decades of data.
  • **Embedding versioning**: stores model version on every Episode for future
    migration safety.
"""

import uuid, json
from datetime import datetime, timezone, timedelta
from threelane_memory.database import run_query
from threelane_memory.embeddings import embed, cosine_similarity
from threelane_memory.llm_interface import invoke_llm
from threelane_memory.schemas import SemanticExtraction
from threelane_memory.config import (
    CONSOLIDATION_AGE_DAYS,
    CONSOLIDATION_BATCH_SIZE,
    CONSOLIDATION_IMPORTANCE_CAP,
    EMBEDDING_MODEL_VERSION,
)


def _build_searchable_text(semantics: SemanticExtraction) -> str:
    """Combine all semantic fields into a single string for embedding."""
    parts = [semantics["summary"]]
    for ent in semantics.get("entities", []):
        parts.append(ent)
    for role in semantics.get("roles", []):
        parts.append(f"{role['entity']} is {role['role']}")
    for act in semantics.get("actions", []):
        obj = f" {act['object']}" if act.get("object") else ""
        parts.append(f"{act['actor']} {act['verb']}{obj}")
    for st in semantics.get("states", []):
        parts.append(f"{st['entity']} {st['attribute']} is {st['value']}")
    if semantics.get("location"):
        parts.append(f"at {semantics['location']}")
    return ". ".join(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_episode(semantics: SemanticExtraction, speaker: str, raw_text: str = "") -> str:
    """Create an Episode node with an embedding vector and return its id.

    Stores `embedding_model` so embeddings can be re-generated when the
    model is changed in the future (70-year migration safety).
    """
    episode_id = f"ep_{uuid.uuid4().hex[:10]}"
    searchable = _build_searchable_text(semantics)
    vector = embed(searchable)
    run_query(
        """
        CREATE (ep:Episode {
            id:              $id,
            summary:         $summary,
            raw_text:        $raw_text,
            emotion:         $emotion,
            importance:      $importance,
            timestamp:       datetime(),
            speaker:         $speaker,
            embedding:       $embedding,
            embedding_model: $embedding_model
        })
        """,
        {
            "id": episode_id,
            "summary": semantics["summary"],
            "raw_text": raw_text,
            "emotion": semantics["emotion"],
            "importance": semantics["importance"],
            "speaker": speaker,
            "embedding": vector,
            "embedding_model": EMBEDDING_MODEL_VERSION,
        },
    )
    return episode_id


def merge_entity(name: str) -> str:
    """Ensure an Entity node exists, resolving to a canonical entity if one
    matches.  Returns the canonical entity name to use for all subsequent
    linking (actions, states, roles, etc.).

    Resolution order (first match wins):
      1. Exact match (case-insensitive)
      2. One name is a substring of the other (e.g. "Jeff" vs "Jeffrey Epstein")
      3. Embedding cosine similarity ≥ 0.88

    If a match is found the new name becomes an alias of the canonical entity
    via an ALIAS_OF relationship.  If no match, a new canonical Entity is created.
    """
    name_lower = name.strip().lower()

    # ── 1. Exact match (case-insensitive) ────────────────────────────────
    exact = run_query(
        "MATCH (e:Entity) WHERE toLower(e.name) = $name_lower "
        "RETURN e.name AS name LIMIT 1",
        {"name_lower": name_lower},
    )
    if exact:
        canonical = exact[0]["name"]
        if canonical != name:
            # Create alias node + ALIAS_OF if name differs in casing / form
            run_query(
                "MERGE (alias:Entity {name:$alias}) "
                "WITH alias "
                "MATCH (canon:Entity {name:$canon}) "
                "MERGE (alias)-[:ALIAS_OF]->(canon)",
                {"alias": name, "canon": canonical},
            )
        return canonical

    # ── 2. Substring containment ─────────────────────────────────────────
    #    "Jeff" ⊂ "Jeffrey Epstein", or "Jeffrey Epstein" ⊃ "Jeff"
    substring_hits = run_query(
        "MATCH (e:Entity) "
        "WHERE toLower(e.name) CONTAINS $name_lower "
        "   OR $name_lower CONTAINS toLower(e.name) "
        "RETURN e.name AS name "
        "LIMIT 5",
        {"name_lower": name_lower},
    )
    if substring_hits:
        # Pick the longest name as canonical (most specific)
        best = max(substring_hits, key=lambda r: len(r["name"]))
        canonical = best["name"]
        if canonical != name:
            run_query(
                "MERGE (alias:Entity {name:$alias}) "
                "WITH alias "
                "MATCH (canon:Entity {name:$canon}) "
                "MERGE (alias)-[:ALIAS_OF]->(canon)",
                {"alias": name, "canon": canonical},
            )
        return canonical

    # ── 3. Embedding similarity ──────────────────────────────────────────
    #    Compare against all existing entity names via embedding.
    #    Only triggered when steps 1+2 didn't match (rare but handles
    #    nicknames like "Mom" vs "Sunita Kumar").
    all_entities = run_query(
        "MATCH (e:Entity) WHERE NOT exists((e)-[:ALIAS_OF]->()) "
        "RETURN e.name AS name"
    )
    if all_entities:
        name_vec = embed(name)
        best_sim, best_name = 0.0, None
        for row in all_entities:
            ent_vec = embed(row["name"])
            sim = cosine_similarity(name_vec, ent_vec)
            if sim > best_sim:
                best_sim = sim
                best_name = row["name"]
        if best_sim >= 0.88 and best_name and best_name != name:
            run_query(
                "MERGE (alias:Entity {name:$alias}) "
                "WITH alias "
                "MATCH (canon:Entity {name:$canon}) "
                "MERGE (alias)-[:ALIAS_OF]->(canon)",
                {"alias": name, "canon": best_name},
            )
            return best_name

    # ── 4. No match — create new canonical entity ────────────────────────
    run_query("MERGE (e:Entity {name:$name})", {"name": name})
    return name


def link_entity_to_episode(entity: str, episode_id: str) -> None:
    """Connect an Entity to an Episode via INVOLVES.

    Uses the canonical name so all episodes cluster under one Entity.
    """
    run_query(
        """
        MATCH (e:Entity {name:$entity}),
              (ep:Episode {id:$ep})
        MERGE (ep)-[:INVOLVES]->(e)
        """,
        {"entity": entity, "ep": episode_id},
    )


def create_action(action: dict, episode_id: str) -> None:
    """Create an Action node linked to its actor, optional object, and episode."""
    run_query(
        """
        CREATE (a:Action {verb:$verb})
        WITH a
        MATCH (actor:Entity {name:$actor})
        MERGE (a)-[:BY_ENTITY]->(actor)
        WITH a
        OPTIONAL MATCH (obj:Entity {name:$object})
        FOREACH (_ IN CASE WHEN obj IS NOT NULL THEN [1] ELSE [] END |
            MERGE (a)-[:ON_ENTITY]->(obj)
        )
        WITH a
        MATCH (ep:Episode {id:$ep})
        MERGE (ep)-[:HAS_ACTION]->(a)
        """,
        {
            "verb": action["verb"],
            "actor": action["actor"],
            "object": action.get("object"),
            "ep": episode_id,
        },
    )


def create_role(role_item: dict) -> None:
    """Bind a Role to an Entity."""
    run_query(
        """
        MATCH (e:Entity {name:$entity})
        MERGE (r:Role {name:$role})
        MERGE (e)-[:HAS_ROLE]->(r)
        """,
        {"entity": role_item["entity"], "role": role_item["role"]},
    )


def create_state(state_item: dict, episode_id: str) -> None:
    """Create a State node tied to an Entity and Episode.

    **Contradiction resolution**: if the same (entity, attribute) already
    has an active State node, the old state is marked `active:false` and a
    `SUPERSEDES` edge is created from the new state to the old one.  This
    keeps a full audit trail while ensuring only the latest value is live.
    """
    # 1. Mark any existing active state for this entity+attribute as inactive
    #    and collect its internal id so we can link SUPERSEDES.
    old_rows = run_query(
        """
        MATCH (e:Entity {name:$entity})<-[:OF_ENTITY]-(s:State {attribute:$attribute})
        WHERE s.active <> false
        SET s.active = false, s.superseded_at = datetime()
        RETURN elementId(s) AS old_id
        """,
        {"entity": state_item["entity"], "attribute": state_item["attribute"]},
    )

    # 2. Create the new (active) state
    run_query(
        """
        MATCH (e:Entity {name:$entity}),
              (ep:Episode {id:$ep})
        CREATE (s:State {
            attribute: $attribute,
            value:     $value,
            active:    true,
            created_at: datetime()
        })
        MERGE (ep)-[:HAS_STATE]->(s)
        MERGE (s)-[:OF_ENTITY]->(e)
        """,
        {
            "entity": state_item["entity"],
            "attribute": state_item["attribute"],
            "value": state_item["value"],
            "ep": episode_id,
        },
    )

    # 3. Link SUPERSEDES from new state to each old state
    if old_rows:
        old_ids = [r["old_id"] for r in old_rows]
        run_query(
            """
            MATCH (new_s:State {attribute:$attribute, active:true})-[:OF_ENTITY]->(e:Entity {name:$entity})
            UNWIND $old_ids AS oid
            MATCH (old_s) WHERE elementId(old_s) = oid
            MERGE (new_s)-[:SUPERSEDES]->(old_s)
            """,
            {
                "entity": state_item["entity"],
                "attribute": state_item["attribute"],
                "old_ids": old_ids,
            },
        )


def bind_location(location: str, episode_id: str) -> None:
    """Attach a Location node to an Episode."""
    run_query(
        """
        MERGE (loc:Location {name:$location})
        WITH loc
        MATCH (ep:Episode {id:$ep})
        MERGE (ep)-[:AT_LOCATION]->(loc)
        """,
        {"location": location, "ep": episode_id},
    )


# ── Main entry point ─────────────────────────────────────────────────────────

def reconcile(semantics: SemanticExtraction, speaker: str, raw_text: str = "") -> str:
    """Write a full SemanticExtraction into Neo4j. Returns the episode id."""

    # 1. Episode
    episode_id = create_episode(semantics, speaker, raw_text)

    # 2. Entities — resolve to canonical names
    #    Build a mapping {extracted_name → canonical_name} so actions/states/roles
    #    use the resolved canonical entity.
    alias_map: dict[str, str] = {}
    for entity in semantics["entities"]:
        canonical = merge_entity(entity)
        alias_map[entity] = canonical
        link_entity_to_episode(canonical, episode_id)

    # 3. Roles (use canonical names)
    for role in semantics["roles"]:
        resolved_role = dict(role)
        resolved_role["entity"] = alias_map.get(role["entity"], role["entity"])
        create_role(resolved_role)

    # 4. Actions (use canonical names)
    for action in semantics["actions"]:
        resolved_action = dict(action)
        resolved_action["actor"] = alias_map.get(action["actor"], action["actor"])
        if action.get("object"):
            resolved_action["object"] = alias_map.get(action["object"], action["object"])
        create_action(resolved_action, episode_id)

    # 5. States (use canonical names)
    for state in semantics["states"]:
        resolved_state = dict(state)
        resolved_state["entity"] = alias_map.get(state["entity"], state["entity"])
        create_state(resolved_state, episode_id)

    # 6. Location
    if semantics.get("location"):
        bind_location(semantics["location"], episode_id)

    return episode_id


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY CONSOLIDATION – keeps the graph manageable over decades
# ══════════════════════════════════════════════════════════════════════════════

def _find_consolidation_candidates(speaker: str) -> list[dict]:
    """Return old, low-importance episodes eligible for consolidation."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=CONSOLIDATION_AGE_DAYS)).isoformat()
    rows = run_query(
        "MATCH (ep:Episode {speaker:$speaker}) "
        "WHERE ep.importance <= $cap "
        "  AND ep.timestamp <= datetime($cutoff) "
        "  AND NOT EXISTS(ep.consolidated) "
        "RETURN ep.id AS id, ep.summary AS summary, ep.raw_text AS raw_text, "
        "       ep.importance AS importance, toString(ep.timestamp) AS ts "
        "ORDER BY ep.timestamp ASC "
        "LIMIT $limit",
        {
            "speaker": speaker,
            "cap": CONSOLIDATION_IMPORTANCE_CAP,
            "cutoff": cutoff,
            "limit": CONSOLIDATION_BATCH_SIZE,
        },
    )
    return rows


def _summarise_batch(episodes: list[dict]) -> str:
    """Use the LLM to produce a compact summary of a batch of episodes."""
    texts = []
    for ep in episodes:
        display = ep.get("raw_text") or ep["summary"]
        texts.append(f"[{ep.get('ts','?')}] {display}")
    prompt = (
        "Combine the following memory episodes into ONE concise summary "
        "paragraph.  Preserve all concrete facts, names, and numbers.\n\n"
        + "\n".join(texts)
    )
    return invoke_llm(prompt)


def consolidate(speaker: str) -> dict:
    """Run one round of memory consolidation for *speaker*.

    1. Find old low-importance episodes.
    2. Summarise them into a single consolidated episode.
    3. Mark originals as consolidated (keep for audit, but exclude from
       future retrieval by default).

    Returns {"merged": int, "consolidated_episode_id": str | None}.
    """
    candidates = _find_consolidation_candidates(speaker)
    if not candidates:
        return {"merged": 0, "consolidated_episode_id": None}

    # Build consolidated summary via LLM
    summary_text = _summarise_batch(candidates)
    episode_id = f"ep_consolidated_{uuid.uuid4().hex[:8]}"

    # Collect all involved entity names from the candidate episodes
    candidate_ids = [c["id"] for c in candidates]
    ent_rows = run_query(
        "UNWIND $ids AS eid "
        "MATCH (ep:Episode {id:eid})-[:INVOLVES]->(e:Entity) "
        "RETURN DISTINCT e.name AS entity",
        {"ids": candidate_ids},
    )
    entity_names = [r["entity"] for r in ent_rows]

    # Build searchable text and embed
    searchable = summary_text + ". " + ". ".join(entity_names)
    vector = embed(searchable)

    # Average importance of originals
    avg_importance = sum(c["importance"] for c in candidates) / len(candidates)

    # Create consolidated episode
    run_query(
        """
        CREATE (ep:Episode {
            id:              $id,
            summary:         $summary,
            raw_text:        '',
            emotion:         'neutral',
            importance:      $importance,
            timestamp:       datetime(),
            speaker:         $speaker,
            embedding:       $embedding,
            embedding_model: $embedding_model,
            consolidated:    true,
            source_count:    $source_count
        })
        """,
        {
            "id": episode_id,
            "summary": summary_text,
            "importance": round(min(avg_importance + 0.1, 1.0), 2),
            "speaker": speaker,
            "embedding": vector,
            "embedding_model": EMBEDDING_MODEL_VERSION,
            "source_count": len(candidates),
        },
    )

    # Link consolidated episode to same entities
    for ename in entity_names:
        run_query(
            "MATCH (e:Entity {name:$entity}), (ep:Episode {id:$ep}) "
            "MERGE (ep)-[:INVOLVES]->(e)",
            {"entity": ename, "ep": episode_id},
        )

    # Mark originals as consolidated
    run_query(
        "UNWIND $ids AS eid "
        "MATCH (ep:Episode {id:eid}) "
        "SET ep.consolidated = true, ep.consolidated_into = $target",
        {"ids": candidate_ids, "target": episode_id},
    )

    return {"merged": len(candidates), "consolidated_episode_id": episode_id}


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING RE-INDEXING – migrate old embeddings to a new model
# ══════════════════════════════════════════════════════════════════════════════

def reindex_embeddings(speaker: str, old_model: str, batch_size: int = 100) -> int:
    """Re-embed all episodes that were embedded with *old_model*.

    Call this after changing EMBEDDING_MODEL_VERSION in config.  Processes
    in batches to avoid rate-limit issues.

    Returns the number of episodes re-indexed.
    """
    total = 0
    while True:
        rows = run_query(
            "MATCH (ep:Episode {speaker:$speaker}) "
            "WHERE ep.embedding_model = $old_model OR ep.embedding_model IS NULL "
            "RETURN ep.id AS id, ep.summary AS summary, ep.raw_text AS raw_text "
            "LIMIT $limit",
            {"speaker": speaker, "old_model": old_model, "limit": batch_size},
        )
        if not rows:
            break
        for r in rows:
            text = r.get("raw_text") or r["summary"]
            new_vec = embed(text)
            run_query(
                "MATCH (ep:Episode {id:$id}) "
                "SET ep.embedding = $vec, ep.embedding_model = $model",
                {"id": r["id"], "vec": new_vec, "model": EMBEDDING_MODEL_VERSION},
            )
        total += len(rows)
    return total
