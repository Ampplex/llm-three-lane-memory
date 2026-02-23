"""Diagnostic: check episode data and vector search results.

Useful for debugging retrieval issues.
"""

from threelane_memory.database import run_query
from threelane_memory.embeddings import embed


def main() -> None:
    # 1. All episodes
    eps = run_query(
        "MATCH (ep:Episode) "
        "RETURN ep.id AS id, ep.summary AS summary, ep.speaker AS speaker, "
        "       ep.embedding IS NOT NULL AS has_emb "
        "ORDER BY ep.timestamp DESC"
    )
    print("=== All Episodes ===")
    for e in eps:
        print(f"  [{e['speaker']}] {e['id']}: {e['summary']}  (emb={e['has_emb']})")

    # 2. Vector search tests
    queries = [
        "what is Max age",
        "what is my age",
        "what is my dog name",
        "Max is 3 years old",
    ]
    for q in queries:
        print(f"\n=== Vector search: '{q}' ===")
        qv = embed(q)
        rows = run_query(
            "CALL db.index.vector.queryNodes('episode_embedding', 5, $vec) "
            "YIELD node, score "
            "RETURN node.id AS id, node.summary AS summary, "
            "       node.speaker AS speaker, score "
            "ORDER BY score DESC",
            {"vec": qv},
        )
        for r in rows:
            print(f"  score={r['score']:.4f} [{r['speaker']}] {r['summary']}")


if __name__ == "__main__":
    main()
