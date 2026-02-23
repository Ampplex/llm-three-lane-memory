"""Interactive chat loop – stores memories and answers questions from the graph."""

from __future__ import annotations

from threelane_memory.operator import operator_extract
from threelane_memory.reconciler import reconcile, consolidate
from threelane_memory.retriever import retrieve
from threelane_memory.llm_interface import invoke_llm
from threelane_memory.backup import save_backup
from threelane_memory.entity_dedup import deduplicate_entities
from threelane_memory.database import close


# ── Retrieval + answering ────────────────────────────────────────────────────

def answer_question(question: str, speaker: str) -> str:
    """Retrieve the most relevant subgraph context and answer via LLM."""
    ctx = retrieve(question, speaker=speaker)
    if not ctx.strip():
        return "I don't have any relevant memories. Tell me something first!"

    prompt = (
        "You are a personal memory assistant. Use ONLY the memory context below "
        "to answer the user's question. If the answer isn't in the context, say so.\n\n"
        f"Memory Context:\n{ctx}\n\n"
        f"Question: {question}"
    )
    return invoke_llm(prompt)


# ── Intent classification ────────────────────────────────────────────────────

def is_question(text: str) -> bool:
    """Simple heuristic: is the user asking a question or stating a fact?"""
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    starters = ("what", "who", "where", "when", "why", "how", "do ", "did ",
                "does ", "is ", "are ", "was ", "were ", "can ", "could ",
                "tell me", "recall", "remember", "show me")
    return any(t.startswith(s) for s in starters)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main(speaker: str = "default") -> None:
    print("╔══════════════════════════════════════════════╗")
    print("║       Memory Chat  (type 'quit' to exit)    ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  • Tell me facts → stored in the graph      ║")
    print("║  • Ask questions → answered from the graph   ║")
    print("║  • /consolidate  → merge old low-importance  ║")
    print("║  • /backup       → export graph to JSON      ║")
    print("║  • /dedup        → merge duplicate entities   ║")
    print("╚══════════════════════════════════════════════╝\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # ── Slash commands ──
        if user_input.lower() == "/consolidate":
            print("  🔄 Running consolidation …")
            try:
                result = consolidate(speaker)
                if result["merged"]:
                    print(f"  ✅ Merged {result['merged']} episodes → {result['consolidated_episode_id']}")
                else:
                    print("  ℹ️  Nothing to consolidate right now.")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            print()
            continue

        if user_input.lower().startswith("/backup"):
            print("  📦 Exporting graph …")
            try:
                save_backup(speaker=speaker)
            except Exception as e:
                print(f"  ❌ Error: {e}")
            print()
            continue

        if user_input.lower() == "/dedup":
            print("  🔗 Scanning for duplicate entities …")
            try:
                result = deduplicate_entities(dry_run=False)
                if result["merged"]:
                    print(f"  ✅ Merged {result['merged']} duplicate entity pair(s)")
                else:
                    print("  ℹ️  No duplicate entities found.")
            except Exception as e:
                print(f"  ❌ Error: {e}")
            print()
            continue

        if is_question(user_input):
            # ── Answer mode ──
            print("  🔍 Searching memory …")
            answer = answer_question(user_input, speaker)
            print(f"  🧠 {answer}\n")
        else:
            # ── Store mode ──
            print("  📥 Extracting semantics …")
            try:
                semantics = operator_extract(user_input)
                episode_id = reconcile(semantics, speaker=speaker, raw_text=user_input)
                print(f"  ✅ Stored episode {episode_id}")
                print(f"     Summary: {semantics['summary']}")
                print(f"     Entities: {', '.join(semantics['entities'])}")
                if semantics.get("location"):
                    print(f"     Location: {semantics['location']}")
                print()
            except Exception as e:
                print(f"  ❌ Error: {e}\n")

    close()


if __name__ == "__main__":
    main()
