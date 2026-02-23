"""OpenAI embeddings wrapper and cosine-similarity helper."""

import numpy as np
from langchain_openai import OpenAIEmbeddings
from threelane_memory.config import (
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_VERSION,
)

# ── Validate credentials at import time ───────────────────────────────────────
if not OPENAI_API_KEY:
    raise ValueError(
        "OpenAI API key not set. Add to .env:\n"
        "  OPENAI_API_KEY=sk-your-api-key"
    )

openai_embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBED_MODEL,
    api_key=OPENAI_API_KEY,
)


def embed(text: str) -> list[float]:
    """Return an embedding vector for *text* (1536-dim for ada-002)."""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM
    try:
        return openai_embeddings.embed_query(text)
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
