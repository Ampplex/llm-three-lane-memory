"""Smart retriever – Neo4j vector index + graph traversal for targeted recall.

Designed for 70-year lifespans:
  • Recency is a light tiebreaker (5%), not a dominant signal.
  • High-importance episodes ignore recency entirely (importance floor).
  • Candidate pool scales dynamically with graph size.
  • Temporal queries (date-range) are supported as a first-class path.
"""

import math, re
from datetime import datetime, timezone
from threelane_memory.database import run_query
from threelane_memory.embeddings import embed
from threelane_memory.config import (
    SIMILARITY_THRESHOLD,
    WEIGHT_SIMILARITY,
    WEIGHT_IMPORTANCE,
    WEIGHT_RECENCY,
    RECENCY_HALF_LIFE_DAYS,
    IMPORTANCE_FLOOR,
    VECTOR_CANDIDATES_MIN,
    VECTOR_CANDIDATES_MAX,
    VECTOR_CANDIDATES_RATIO,
)

# ── Tunables ──────────────────────────────────────────────────────────────────
TOP_K = 8                         # max episodes to retrieve
MIN_SCORE = 0.20                  # drop episodes below this combined score
RECENT_WINDOW_MINUTES = 5         # always include episodes this fresh
                                  # (covers vector index lag)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _recency_weight(ts_iso: str | None) -> float:
    """Exponential decay based on age.  Returns 0‑1.

    With RECENCY_HALF_LIFE_DAYS=365 a 10-year-old memory still scores ~0.001
    (instead of ~0 with 30-day half-life).
    """
    if not ts_iso:
        return 0.5
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
    except Exception:
        return 0.5
    return math.exp(-0.693 * age_days / RECENCY_HALF_LIFE_DAYS)


def _combined_score(sim: float, importance: float, recency: float) -> float:
    """Weighted combination with importance-floor bypass.

    Episodes with importance >= IMPORTANCE_FLOOR get recency=1.0 so that
    major life events are never penalised for being old.
    """
    if importance >= IMPORTANCE_FLOOR:
        recency = 1.0  # never penalise landmark memories
    return (WEIGHT_SIMILARITY * sim
            + WEIGHT_IMPORTANCE * importance
            + WEIGHT_RECENCY * recency)


def _dynamic_candidates(speaker: str) -> int:
    """Choose ANN candidate count based on total episodes for this speaker."""
    rows = run_query(
        "MATCH (ep:Episode {speaker:$speaker}) RETURN count(ep) AS cnt",
        {"speaker": speaker},
    )
    total = rows[0]["cnt"] if rows else 0
    candidates = int(total * VECTOR_CANDIDATES_RATIO)
    return max(VECTOR_CANDIDATES_MIN, min(candidates, VECTOR_CANDIDATES_MAX))


# ── Temporal query helpers ────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_RELATIVE_RE = re.compile(
    r"(?:last|past)\s+(\d+)\s+(day|week|month|year)s?", re.IGNORECASE
)


def _extract_time_range(text: str):
    """Try to pull a (start_dt, end_dt) from the question. Returns None if no temporal cue."""
    lower = text.lower()
    now = datetime.now(timezone.utc)

    # "last N days/weeks/months/years"
    m = _RELATIVE_RE.search(lower)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days_map = {"day": 1, "week": 7, "month": 30, "year": 365}
        delta_days = n * days_map.get(unit, 1)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = start - __import__("datetime").timedelta(days=delta_days)
        return start.isoformat(), now.isoformat()

    # Explicit year mentions like "in 2030"
    year_match = _YEAR_RE.search(text)
    if year_match:
        year = int(year_match.group())
        # Check for month
        month_start, month_end = 1, 12
        for name, num in _MONTH_NAMES.items():
            if name in lower:
                month_start = month_end = num
                break
        from datetime import timedelta
        start = datetime(year, month_start, 1, tzinfo=timezone.utc)
        if month_end == 12:
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
        else:
            end = datetime(year, month_end + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
        return start.isoformat(), end.isoformat()

    return None


def _temporal_episode_ids(speaker: str, start_iso: str, end_iso: str,
                          limit: int = TOP_K) -> list[str]:
    """Fetch episode IDs within a date range, ordered by importance DESC."""
    rows = run_query(
        "MATCH (ep:Episode {speaker:$speaker}) "
        "WHERE ep.timestamp >= datetime($start) AND ep.timestamp <= datetime($end) "
        "RETURN ep.id AS id "
        "ORDER BY ep.importance DESC, ep.timestamp DESC "
        "LIMIT $limit",
        {"speaker": speaker, "start": start_iso, "end": end_iso, "limit": limit},
    )
    return [r["id"] for r in rows]


def _recent_episode_ids(speaker: str, minutes: int = 5) -> list[str]:
    """Fetch episodes created in the last *minutes* minutes.

    This is a safety net: Neo4j vector indexes update asynchronously so a
    brand-new episode might not appear in ANN search yet.  By always
    including very recent episodes we guarantee the user's latest input is
    available for retrieval immediately.
    """
    rows = run_query(
        "MATCH (ep:Episode {speaker:$speaker}) "
        "WHERE ep.timestamp >= datetime() - duration({minutes: $mins}) "
        "RETURN ep.id AS id "
        "ORDER BY ep.timestamp DESC",
        {"speaker": speaker, "mins": minutes},
    )
    return [r["id"] for r in rows]


# ── Core retrieval (Neo4j vector index) ──────────────────────────────────────

def find_relevant_episodes(question: str, speaker: str, top_k: int = TOP_K) -> list[str]:
    """Return up to *top_k* episode IDs most relevant to *question*.

    Pipeline:
      0. Always fetch very recent episodes (last 5 min) to beat index lag.
      1. Check for temporal cues → date-range query if found.
      2. Otherwise: dynamic ANN search → re-rank with importance + recency.
      3. Merge all result sets (recent + temporal + semantic), deduplicated.
    """
    # ── 0. Recent-episodes safety net (beats vector index lag) ───────────
    ids_recent = _recent_episode_ids(speaker, minutes=RECENT_WINDOW_MINUTES)

    ids_from_time: list[str] = []
    time_range = _extract_time_range(question)
    if time_range:
        ids_from_time = _temporal_episode_ids(speaker, *time_range, limit=top_k)

    # ── Semantic / vector path ───────────────────────────────────────────
    q_vec = embed(question)
    candidates = _dynamic_candidates(speaker)

    rows = run_query(
        "CALL db.index.vector.queryNodes('episode_embedding', $candidates, $vec) "
        "YIELD node, score "
        "WHERE node.speaker = $speaker "
        "RETURN node.id AS id, score AS sim, "
        "       node.importance AS imp, toString(node.timestamp) AS ts",
        {"candidates": candidates, "vec": q_vec, "speaker": speaker},
    )

    scored = []
    for r in rows:
        sim = r["sim"]
        if sim < SIMILARITY_THRESHOLD * 0.5:
            continue
        recency = _recency_weight(r.get("ts"))
        importance = r.get("imp", 0.5)
        score = _combined_score(sim, importance, recency)
        if score >= MIN_SCORE:
            scored.append((r["id"], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    ids_from_vector = [eid for eid, _ in scored[:top_k]]

    # ── Merge recent + temporal + semantic (deduplicated, recent first) ───
    seen = set()
    merged: list[str] = []
    for eid in ids_recent + ids_from_time + ids_from_vector:
        if eid not in seen:
            seen.add(eid)
            merged.append(eid)
    return merged[:top_k]


# ── Subgraph expansion ───────────────────────────────────────────────────────

def expand_episodes(episode_ids: list[str]) -> str:
    """Given episode IDs, traverse their subgraphs and return formatted context."""
    if not episode_ids:
        return ""

    # Episodes (exclude originals that were consolidated into another episode)
    episodes = run_query(
        "UNWIND $ids AS eid "
        "MATCH (ep:Episode {id:eid}) "
        "WHERE ep.consolidated_into IS NULL "
        "RETURN ep.id AS id, ep.summary AS summary, "
        "       ep.raw_text AS raw_text, ep.emotion AS emotion, "
        "       ep.importance AS importance, toString(ep.timestamp) AS ts "
        "ORDER BY ep.timestamp DESC",
        {"ids": episode_ids},
    )

    # Entities involved (follow ALIAS_OF to show canonical names)
    entities = run_query(
        "UNWIND $ids AS eid "
        "MATCH (ep:Episode {id:eid})-[:INVOLVES]->(e:Entity) "
        "OPTIONAL MATCH (e)-[:ALIAS_OF]->(canon:Entity) "
        "RETURN DISTINCT ep.id AS ep_id, coalesce(canon.name, e.name) AS entity",
        {"ids": episode_ids},
    )

    # Actions
    actions = run_query(
        "UNWIND $ids AS eid "
        "MATCH (ep:Episode {id:eid})-[:HAS_ACTION]->(a:Action)-[:BY_ENTITY]->(actor:Entity) "
        "OPTIONAL MATCH (a)-[:ON_ENTITY]->(obj:Entity) "
        "RETURN ep.id AS ep_id, actor.name AS actor, a.verb AS verb, obj.name AS object",
        {"ids": episode_ids},
    )

    # States (only active — superseded states are excluded)
    # coalesce handles old State nodes that don't have the 'active' property yet
    # Also follows ALIAS_OF to resolve canonical entity names
    states = run_query(
        "UNWIND $ids AS eid "
        "MATCH (ep:Episode {id:eid})-[:HAS_STATE]->(s:State)-[:OF_ENTITY]->(e:Entity) "
        "OPTIONAL MATCH (e)-[:ALIAS_OF]->(canon:Entity) "
        "WHERE coalesce(s.active, true) <> false "
        "RETURN ep.id AS ep_id, coalesce(canon.name, e.name) AS entity, s.attribute AS attr, s.value AS val",
        {"ids": episode_ids},
    )

    # Roles (for involved entities + their aliases)
    involved_names = list({e["entity"] for e in entities})
    roles = []
    if involved_names:
        roles = run_query(
            "UNWIND $names AS n "
            "MATCH (e:Entity {name:n}) "
            "OPTIONAL MATCH (alias:Entity)-[:ALIAS_OF]->(e) "
            "WITH collect(e) + collect(alias) AS all_ents "
            "UNWIND all_ents AS ent "
            "MATCH (ent)-[:HAS_ROLE]->(r:Role) "
            "RETURN DISTINCT ent.name AS entity, r.name AS role",
            {"names": involved_names},
        )

    # Locations
    locations = run_query(
        "UNWIND $ids AS eid "
        "MATCH (ep:Episode {id:eid})-[:AT_LOCATION]->(loc:Location) "
        "RETURN ep.id AS ep_id, loc.name AS location",
        {"ids": episode_ids},
    )

    # ── Format ────────────────────────────────────────────────────────────
    lines = []
    for ep in episodes:
        # Show raw text if available (contains the actual user words)
        display = ep.get('raw_text') or ep['summary']
        lines.append(
            f"[{ep.get('ts', '?')}] {display}  "
            f"(summary: {ep['summary']}, emotion={ep['emotion']}, importance={ep['importance']})"
        )

        ep_entities = [e["entity"] for e in entities if e["ep_id"] == ep["id"]]
        if ep_entities:
            lines.append(f"  Entities: {', '.join(ep_entities)}")

        ep_actions = [a for a in actions if a["ep_id"] == ep["id"]]
        for a in ep_actions:
            obj = f" → {a['object']}" if a.get("object") else ""
            lines.append(f"  Action: {a['actor']} {a['verb']}{obj}")

        ep_states = [s for s in states if s["ep_id"] == ep["id"]]
        for s in ep_states:
            lines.append(f"  State: {s['entity']}.{s['attr']} = {s['val']}")

        ep_locs = [loc["location"] for loc in locations if loc["ep_id"] == ep["id"]]
        if ep_locs:
            lines.append(f"  Location: {', '.join(ep_locs)}")

        lines.append("")

    if roles:
        lines.append("Roles:")
        for r in roles:
            lines.append(f"  {r['entity']} → {r['role']}")

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(question: str, speaker: str, top_k: int = TOP_K) -> str:
    """End-to-end: embed question → find top-k episodes → expand subgraphs."""
    episode_ids = find_relevant_episodes(question, speaker, top_k)
    return expand_episodes(episode_ids)
