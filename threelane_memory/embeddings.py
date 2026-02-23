"""Embedding wrapper — supports Ollama (local) and OpenAI providers."""

from __future__ import annotations

import numpy as np

from threelane_memory.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    EMBEDDING_DIM,
)

# ── Build the embeddings client based on provider ────────────────────────────

if LLM_PROVIDER == "ollama":
    from langchain_ollama import OllamaEmbeddings

    _embeddings = OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
else:
    from langchain_openai import OpenAIEmbeddings

    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not set. Add to .env:\n"
            "  OPENAI_API_KEY=sk-your-api-key"
        )
    _embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL,
        api_key=OPENAI_API_KEY,
    )


def embed(text: str) -> list[float]:
    """Return an embedding vector for *text*."""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM
    try:
        return _embeddings.embed_query(text)
    except Exception as e:
        print(f"[embeddings] Error generating embedding: {e}")
        return [0.0] * EMBEDDING_DIM


def cosine_similarity(vec1, vec2) -> float:
    """Cosine similarity between two vectors."""
    v1, v2 = np.asarray(vec1), np.asarray(vec2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))
