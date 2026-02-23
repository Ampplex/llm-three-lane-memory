"""threelane-memory — Personal long-term memory on Neo4j + OpenAI.

Quick-start::

    from threelane_memory import store, query, close

    store("My dog Max is 3 years old", speaker="ankesh")
    print(query("How old is Max?", speaker="ankesh"))
    close()
"""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy-import public symbols so ``import threelane_memory`` works without API keys."""
    _lazy = {
        "close": ("threelane_memory.database", "close"),
        "embed": ("threelane_memory.embeddings", "embed"),
        "cosine_similarity": ("threelane_memory.embeddings", "cosine_similarity"),
        "invoke_llm": ("threelane_memory.llm_interface", "invoke_llm"),
        "operator_extract": ("threelane_memory.operator", "operator_extract"),
        "reconcile": ("threelane_memory.reconciler", "reconcile"),
        "consolidate": ("threelane_memory.reconciler", "consolidate"),
        "reindex_embeddings": ("threelane_memory.reconciler", "reindex_embeddings"),
        "retrieve": ("threelane_memory.retriever", "retrieve"),
        "save_backup": ("threelane_memory.backup", "save_backup"),
        "deduplicate_entities": ("threelane_memory.entity_dedup", "deduplicate_entities"),
    }
    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(module_path)
        val = getattr(mod, attr)
        globals()[name] = val  # cache for next access
        return val
    raise AttributeError(f"module 'threelane_memory' has no attribute {name!r}")


# ── Convenience helpers ──────────────────────────────────────────────────────


def store(text: str, *, speaker: str = "default") -> str:
    """Extract semantics from *text* and persist to the knowledge graph.

    Returns the episode ID of the newly created node.
    """
    from threelane_memory.operator import operator_extract
    from threelane_memory.reconciler import reconcile

    semantics = operator_extract(text)
    return reconcile(semantics, speaker=speaker, raw_text=text)


def query(question: str, *, speaker: str = "default") -> str:
    """Retrieve relevant memories and answer *question* via LLM.

    Returns the LLM-generated answer string.
    """
    from threelane_memory.retriever import retrieve
    from threelane_memory.llm_interface import invoke_llm

    ctx = retrieve(question, speaker=speaker)
    if not ctx.strip():
        return "I don't have any relevant memories for that question."
    prompt = (
        "You are a personal memory assistant. Use ONLY the memory context below "
        "to answer the user's question. If the answer isn't in the context, say so.\n\n"
        f"Memory Context:\n{ctx}\n\n"
        f"Question: {question}"
    )
    return invoke_llm(prompt)


__all__ = [
    "__version__",
    # High-level API
    "store",
    "query",
    "close",
    # Core pipeline
    "operator_extract",
    "reconcile",
    "retrieve",
    "invoke_llm",
    # Utilities
    "embed",
    "cosine_similarity",
    "consolidate",
    "reindex_embeddings",
    "save_backup",
    "deduplicate_entities",
]
