"""Optional Streamlit web interface for threelane-memory.

Install the streamlit extra first::

    pip install -e ".[streamlit]"

Then run::

    streamlit run examples/streamlit_app.py
"""

import streamlit as st

from threelane_memory.operator import operator_extract
from threelane_memory.reconciler import reconcile, consolidate
from threelane_memory.retriever import retrieve
from threelane_memory.llm_interface import invoke_llm
from threelane_memory.backup import save_backup
from threelane_memory.entity_dedup import deduplicate_entities

SPEAKER = "ankesh"


def is_question(text: str) -> bool:
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    starters = (
        "what", "who", "where", "when", "why", "how", "do ", "did ",
        "does ", "is ", "are ", "was ", "were ", "can ", "could ",
        "tell me", "recall", "remember", "show me",
    )
    return any(t.startswith(s) for s in starters)


def answer_question(question: str) -> str:
    ctx = retrieve(question, speaker=SPEAKER)
    if not ctx.strip():
        return "I don't have any relevant memories. Tell me something first!"
    prompt = (
        "You are a personal memory assistant. Use ONLY the memory context below "
        "to answer the user's question. If the answer isn't in the context, say so.\n\n"
        f"Memory Context:\n{ctx}\n\n"
        f"Question: {question}"
    )
    return invoke_llm(prompt)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Memory Chat", page_icon="🧠", layout="centered")
st.title("🧠 Memory Chat")
st.caption("Tell me facts → stored in the graph  •  Ask questions → answered from the graph")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar: admin tools ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Admin Tools")

    if st.button("🔄 Consolidate Memories", use_container_width=True):
        with st.spinner("Running consolidation…"):
            try:
                result = consolidate(SPEAKER)
                if result["merged"]:
                    st.success(f"Merged {result['merged']} episodes")
                else:
                    st.info("Nothing to consolidate right now.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("📦 Backup Graph", use_container_width=True):
        with st.spinner("Exporting graph…"):
            try:
                path = save_backup(speaker=SPEAKER)
                st.success(f"Saved to {path}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🔗 Deduplicate Entities", use_container_width=True):
        with st.spinner("Scanning for duplicates…"):
            try:
                result = deduplicate_entities(dry_run=False)
                if result["merged"]:
                    st.success(f"Merged {result['merged']} duplicate pair(s)")
                else:
                    st.info("No duplicate entities found.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Render chat history ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ───────────────────────────────────────────────────────────────

if user_input := st.chat_input("Tell me something or ask a question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if is_question(user_input):
            with st.spinner("🔍 Searching memory…"):
                answer = answer_question(user_input)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.spinner("📥 Extracting semantics…"):
                try:
                    semantics = operator_extract(user_input)
                    episode_id = reconcile(semantics, speaker=SPEAKER, raw_text=user_input)
                    parts = [
                        f"✅ **Stored episode** `{episode_id}`",
                        f"- **Summary:** {semantics['summary']}",
                        f"- **Entities:** {', '.join(semantics['entities'])}",
                        f"- **Emotion:** {semantics['emotion']}",
                        f"- **Importance:** {semantics['importance']}",
                    ]
                    if semantics.get("location"):
                        parts.append(f"- **Location:** {semantics['location']}")
                    response = "\n".join(parts)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
