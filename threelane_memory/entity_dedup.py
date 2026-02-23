"""Entity deduplication utility – scan and merge duplicate Entity nodes.

Runs three passes:
  1. Case-insensitive exact matches  ("jeff" ↔ "Jeff")
  2. Substring containment           ("Jeff" ↔ "Jeffrey Epstein")
  3. Embedding similarity ≥ 0.88     ("Mom" ↔ "Sunita Kumar" — if close enough)

For each duplicate pair the longer / more-specific name is kept as canonical
and the other becomes an alias via ALIAS_OF.  All relationships (INVOLVES,
BY_ENTITY, ON_ENTITY, OF_ENTITY, HAS_ROLE) on the alias are migrated to the
canonical entity.

Usage:
    python -m threelane_memory.entity_dedup              # dry-run (show matches)
    python -m threelane_memory.entity_dedup --apply      # actually merge
"""

from __future__ import annotations

from threelane_memory.database import run_query, close
from threelane_memory.embeddings import embed, cosine_similarity

SIMILARITY_THRESHOLD = 0.88


def _all_canonical_entities() -> list[dict]:
    """Return all Entity nodes that are NOT already aliases."""
    return run_query(
        "MATCH (e:Entity) "
        "WHERE NOT exists((e)-[:ALIAS_OF]->()) "
        "RETURN e.name AS name"
    )


def _pick_canonical(name_a: str, name_b: str) -> tuple[str, str]:
    """Return (canonical, alias).  Longer/more specific name wins."""
    if len(name_a) >= len(name_b):
        return name_a, name_b
    return name_b, name_a


def _create_alias(alias_name: str, canonical_name: str, dry_run: bool) -> None:
    """Link alias to canonical and migrate all relationships."""
    if dry_run:
        print(f"  [DRY-RUN] Would alias '{alias_name}' → '{canonical_name}'")
        return

    # 1. Create ALIAS_OF relationship
    run_query(
        "MATCH (alias:Entity {name:$alias}), (canon:Entity {name:$canon}) "
        "MERGE (alias)-[:ALIAS_OF]->(canon)",
        {"alias": alias_name, "canon": canonical_name},
    )

    # 2. Migrate INVOLVES edges: episodes linked to alias → also link to canonical
    run_query(
        "MATCH (ep:Episode)-[:INVOLVES]->(alias:Entity {name:$alias}) "
        "MATCH (canon:Entity {name:$canon}) "
        "MERGE (ep)-[:INVOLVES]->(canon)",
        {"alias": alias_name, "canon": canonical_name},
    )

    # 3. Migrate BY_ENTITY (Actions)
    run_query(
        "MATCH (a:Action)-[:BY_ENTITY]->(alias:Entity {name:$alias}) "
        "MATCH (canon:Entity {name:$canon}) "
        "MERGE (a)-[:BY_ENTITY]->(canon)",
        {"alias": alias_name, "canon": canonical_name},
    )

    # 4. Migrate ON_ENTITY (Actions)
    run_query(
        "MATCH (a:Action)-[:ON_ENTITY]->(alias:Entity {name:$alias}) "
        "MATCH (canon:Entity {name:$canon}) "
        "MERGE (a)-[:ON_ENTITY]->(canon)",
        {"alias": alias_name, "canon": canonical_name},
    )

    # 5. Migrate OF_ENTITY (States)
    run_query(
        "MATCH (s:State)-[:OF_ENTITY]->(alias:Entity {name:$alias}) "
        "MATCH (canon:Entity {name:$canon}) "
        "MERGE (s)-[:OF_ENTITY]->(canon)",
        {"alias": alias_name, "canon": canonical_name},
    )

    # 6. Migrate HAS_ROLE
    run_query(
        "MATCH (alias:Entity {name:$alias})-[:HAS_ROLE]->(r:Role) "
        "MATCH (canon:Entity {name:$canon}) "
        "MERGE (canon)-[:HAS_ROLE]->(r)",
        {"alias": alias_name, "canon": canonical_name},
    )

    print(f"  ✅ Aliased '{alias_name}' → '{canonical_name}' (relationships migrated)")


def deduplicate_entities(dry_run: bool = True) -> dict:
    """Run entity deduplication.  Returns summary stats."""
    entities = _all_canonical_entities()
    names = [e["name"] for e in entities]
    already_merged = set()
    merges = []

    # ── Pass 1: Case-insensitive exact match ─────────────────────────────
    lower_map: dict[str, list[str]] = {}
    for n in names:
        key = n.strip().lower()
        lower_map.setdefault(key, []).append(n)

    for key, group in lower_map.items():
        if len(group) > 1:
            canon = max(group, key=len)
            for alias in group:
                if alias != canon and alias not in already_merged:
                    merges.append((alias, canon, "case-match"))
                    already_merged.add(alias)

    # ── Pass 2: Substring containment ────────────────────────────────────
    remaining = [n for n in names if n not in already_merged]
    for i, a in enumerate(remaining):
        for b in remaining[i + 1:]:
            if b in already_merged or a in already_merged:
                continue
            a_lower, b_lower = a.lower(), b.lower()
            if a_lower in b_lower or b_lower in a_lower:
                canon, alias = _pick_canonical(a, b)
                merges.append((alias, canon, "substring"))
                already_merged.add(alias)

    # ── Pass 3: Embedding similarity ─────────────────────────────────────
    remaining = [n for n in names if n not in already_merged]
    if len(remaining) > 1:
        vecs = {n: embed(n) for n in remaining}
        for i, a in enumerate(remaining):
            for b in remaining[i + 1:]:
                if b in already_merged or a in already_merged:
                    continue
                sim = cosine_similarity(vecs[a], vecs[b])
                if sim >= SIMILARITY_THRESHOLD:
                    canon, alias = _pick_canonical(a, b)
                    merges.append((alias, canon, f"embedding-sim={sim:.3f}"))
                    already_merged.add(alias)

    # ── Apply ────────────────────────────────────────────────────────────
    if not merges:
        print("  ℹ️  No duplicate entities found.")
        return {"duplicates_found": 0, "merged": 0}

    print(f"\n  Found {len(merges)} duplicate pair(s):\n")
    for alias, canon, reason in merges:
        print(f"    '{alias}' → '{canon}'  ({reason})")
        _create_alias(alias, canon, dry_run)

    return {"duplicates_found": len(merges), "merged": 0 if dry_run else len(merges)}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate Entity nodes in Neo4j")
    parser.add_argument("--apply", action="store_true",
                        help="Actually merge duplicates (default is dry-run)")
    args = parser.parse_args()

    try:
        result = deduplicate_entities(dry_run=not args.apply)
        mode = "APPLIED" if args.apply else "DRY-RUN"
        print(f"\n  [{mode}] Duplicates found: {result['duplicates_found']}, "
              f"Merged: {result['merged']}")
    finally:
        close()
