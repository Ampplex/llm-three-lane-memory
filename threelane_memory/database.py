"""Neo4j driver and query helper."""

from __future__ import annotations

import logging
import sys

from neo4j import GraphDatabase

from threelane_memory.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    EMBEDDING_DIM,
    LLM_PROVIDER,
)

# Suppress Neo4j property-not-exist warnings for new properties on old nodes
logging.getLogger("neo4j").setLevel(logging.ERROR)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    notifications_min_severity="OFF",
)


def run_query(query: str, params: dict | None = None) -> list[dict]:
    """Execute a Cypher query and return the result rows as dicts."""
    with driver.session() as session:
        result = session.run(query, params or {})
        return result.data()


def close():
    """Shut down the Neo4j driver cleanly."""
    driver.close()


# ── Vector index dimension check ─────────────────────────────────────────────

def get_index_dimension() -> int | None:
    """Return the dimension of the ``episode_embedding`` vector index, or None."""
    try:
        rows = run_query(
            "SHOW INDEXES YIELD name, type, options "
            "WHERE name = 'episode_embedding' AND type = 'VECTOR' "
            "RETURN options"
        )
        if rows:
            opts = rows[0].get("options", {})
            cfg = opts.get("indexConfig", {})
            dim = cfg.get("vector.dimensions")
            return int(dim) if dim is not None else None
    except Exception:
        return None
    return None


def check_index_dimension(quiet: bool = False) -> bool:
    """Compare Neo4j index dimension with configured EMBEDDING_DIM.

    Returns True if they match (or if the index doesn't exist yet).
    Prints a warning to stderr on mismatch unless *quiet* is True.
    """
    index_dim = get_index_dimension()
    if index_dim is None:
        # Index doesn't exist yet — will be created on first store
        return True
    if index_dim != EMBEDDING_DIM:
        if not quiet:
            print(
                f"\n⚠  DIMENSION MISMATCH: Neo4j vector index is {index_dim}-dim "
                f"but your {LLM_PROVIDER} embedding model expects {EMBEDDING_DIM}-dim.\n"
                f"   Queries will fail or return bad results until you fix this.\n"
                f"\n"
                f"   To fix, run these steps:\n"
                f"   1. Drop the old index in Neo4j Browser:\n"
                f"        DROP INDEX episode_embedding;\n"
                f"   2. Create a new index with the correct dimension:\n"
                f"        CREATE VECTOR INDEX episode_embedding IF NOT EXISTS\n"
                f"        FOR (ep:Episode) ON (ep.embedding)\n"
                f"        OPTIONS {{indexConfig: {{\n"
                f"          `vector.dimensions`: {EMBEDDING_DIM},\n"
                f"          `vector.similarity_function`: 'cosine'\n"
                f"        }}}};\n"
                f"   3. Re-embed all episodes:\n"
                f"        threelane-memory reindex --old-model <previous-model>\n",
                file=sys.stderr,
            )
        return False
    return True
