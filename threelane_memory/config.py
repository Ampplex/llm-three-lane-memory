import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ── Supported Providers ──────────────────────────────────────────────────────
SUPPORTED_PROVIDERS = ("ollama", "openai")

# ── Neo4j Configuration ──────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://your-instance.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── LLM Provider ──────────────────────────────────────────────────────────────
# Set LLM_PROVIDER to "ollama" for local models or "openai" for OpenAI.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower().strip()

if LLM_PROVIDER not in SUPPORTED_PROVIDERS:
    print(
        f"[config] WARNING: Unknown LLM_PROVIDER='{LLM_PROVIDER}'. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}. Falling back to 'ollama'.",
        file=sys.stderr,
    )
    LLM_PROVIDER = "ollama"

# Ollama settings (default)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# OpenAI settings (used when LLM_PROVIDER=openai)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")

# Embedding dimension — auto-set based on provider
_EMBED_DIMS = {"nomic-embed-text": 768, "text-embedding-ada-002": 1536}
_active_embed = OLLAMA_EMBED_MODEL if LLM_PROVIDER == "ollama" else OPENAI_EMBED_MODEL
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", str(_EMBED_DIMS.get(_active_embed, 768))))
EMBEDDING_MODEL_VERSION = os.getenv("EMBEDDING_MODEL_VERSION", _active_embed)


# ── Provider Info (used by CLI `config` and startup checks) ──────────────────

def get_provider_summary() -> dict:
    """Return a dict summarising the active provider configuration."""
    if LLM_PROVIDER == "ollama":
        chat_model = OLLAMA_CHAT_MODEL
        embed_model = OLLAMA_EMBED_MODEL
    else:
        chat_model = OPENAI_CHAT_MODEL
        embed_model = OPENAI_EMBED_MODEL
    return {
        "provider": LLM_PROVIDER,
        "chat_model": chat_model,
        "embed_model": embed_model,
        "embedding_dim": EMBEDDING_DIM,
        "neo4j_uri": NEO4J_URI,
    }

# ── Memory / Retrieval Thresholds ────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.70
FALLBACK_SIMILARITY_THRESHOLD = 0.60

# ── Retrieval Scoring Weights ────────────────────────────────────────────────
# 70-year-safe: recency is a light tiebreaker, not a dominant signal.
WEIGHT_SIMILARITY = float(os.getenv("WEIGHT_SIMILARITY", "0.65"))
WEIGHT_IMPORTANCE = float(os.getenv("WEIGHT_IMPORTANCE", "0.30"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.05"))
RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "365.0"))
IMPORTANCE_FLOOR = float(os.getenv("IMPORTANCE_FLOOR", "0.75"))
# Episodes with importance >= IMPORTANCE_FLOOR ignore recency entirely.

# ── Dynamic Candidate Pool ───────────────────────────────────────────────────
VECTOR_CANDIDATES_MIN = int(os.getenv("VECTOR_CANDIDATES_MIN", "50"))
VECTOR_CANDIDATES_MAX = int(os.getenv("VECTOR_CANDIDATES_MAX", "500"))
VECTOR_CANDIDATES_RATIO = float(os.getenv("VECTOR_CANDIDATES_RATIO", "0.02"))
# candidates = clamp(total_episodes * RATIO, MIN, MAX)

# ── Consolidation Settings ───────────────────────────────────────────────────
CONSOLIDATION_AGE_DAYS = int(os.getenv("CONSOLIDATION_AGE_DAYS", "90"))
CONSOLIDATION_BATCH_SIZE = int(os.getenv("CONSOLIDATION_BATCH_SIZE", "50"))
CONSOLIDATION_IMPORTANCE_CAP = float(os.getenv("CONSOLIDATION_IMPORTANCE_CAP", "0.3"))
# Only consolidate episodes older than AGE_DAYS **and** importance <= CAP.

# ── Backup / Export ──────────────────────────────────────────────────────────
BACKUP_DIR = os.getenv("BACKUP_DIR", os.path.join(os.path.dirname(__file__), "..", "backups"))
