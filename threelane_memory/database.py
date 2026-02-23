"""Neo4j driver and query helper."""

import logging
from neo4j import GraphDatabase
from threelane_memory.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

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
