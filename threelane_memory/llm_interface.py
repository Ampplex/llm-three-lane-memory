"""LLM wrapper — supports Ollama (local) and OpenAI providers."""

from __future__ import annotations

from threelane_memory.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)

# ── Build the LLM client based on provider ───────────────────────────────────

if LLM_PROVIDER == "ollama":
    from langchain_ollama import ChatOllama

    client = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.15,
    )
else:
    from langchain_openai import ChatOpenAI

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
