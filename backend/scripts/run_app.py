"""
app.py
GLA Admission Chatbot — Streamlit UI

Run karo:
    streamlit run app.py
"""

import sys
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatbot import GLAChatbot
from embedder import index_exists

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="GLA Admission Chatbot",
    page_icon="🎓",
    layout="centered",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.stChatMessage { border-radius: 12px; }
.stTextInput > div > div > input { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────
st.title("🎓 GLA University")
st.subheader("Admission Assistant")
st.caption("Ask anything about fees, eligibility, courses, hostel, scholarships")

# ── Check: Ingestion done hai? ──────────────────────────────
if not index_exists("vector_store"):
    st.error("""
    **Vector store not found!**

    Please run ingestion first:
    ```bash
    python scripts/ingest.py
    ```
    """)
    st.stop()

# ── Chatbot initialize ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading chatbot...")
def load_chatbot():
    return GLAChatbot()

bot = load_chatbot()

# ── Session state ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm the GLA University admission assistant.\n\n"
                "Ask me about:\n"
                "- B.Tech / MBA / MCA fees\n"
                "- Eligibility criteria\n"
                "- Scholarships\n"
                "- Hostel facility\n"
                "- Admission dates\n\n"
                "What would you like to know?"
            ),
        }
    ]

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")

    show_sources = st.toggle("Show sources", value=False,
                              help="Show which PDF chunk the answer came from")

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        bot.reset_history()
        st.rerun()

    st.divider()
    st.markdown("**Quick questions:**")
    quick_questions = [
        "What are the B.Tech fees?",
        "What is the eligibility criteria?",
        "Is scholarship available?",
        "Is hostel available?",
        "What is the admission deadline?",
        "What courses are offered?",
    ]
    for q in quick_questions:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            st.session_state.pending_query = q

    st.divider()
    st.caption("Powered by Groq + LangChain + FAISS")

# ── Chat display ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " | ".join(f"📄 {s}" for s in msg["sources"]))

# ── Handle quick question ────────────────────────────────────
pending = st.session_state.pop("pending_query", None)

# ── Chat input ───────────────────────────────────────────────
user_input = st.chat_input("Type your question here...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if show_sources:
                result = bot.ask_with_sources(user_input)
                answer = result["answer"]
                sources = result["sources"]
            else:
                answer = bot.ask(user_input)
                sources = []

        st.markdown(answer)
        if sources:
            st.caption("Sources: " + " | ".join(f"📄 {s}" for s in sources))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })