"""OpenAI LLM wrapper for chat / reasoning calls."""

from langchain_openai import ChatOpenAI
from threelane_memory.config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)

# ── Validate credentials at import time ───────────────────────────────────────
if not OPENAI_API_KEY:
    raise ValueError(
        "OpenAI API key not set. Add to .env:\n"
        "  OPENAI_API_KEY=sk-your-api-key"
    )

client = ChatOpenAI(
    model=OPENAI_CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.15,
    streaming=False,
)


def invoke_llm(prompt: str) -> str:
    """Send a single prompt to the LLM and return the raw text response."""
    response = client.invoke([{"role": "user", "content": prompt}])
    return response.content


def ask_llm(context: str, question: str) -> str:
    """Answer a question using the provided memory context."""
    prompt = (
        "Answer ONLY using the memory below.\n\n"
        f"Memory:\n{context}\n\n"
        f"Question:\n{question}\n"
    )
    return invoke_llm(prompt)
