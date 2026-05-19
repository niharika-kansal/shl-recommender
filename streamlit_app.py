"""
streamlit_app.py
----------------
Production Streamlit app for SHL Assessment Recommender.
Calls run_agent() directly — no FastAPI/uvicorn needed.
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# ── Load secrets (Streamlit Cloud injects these) ──────────────────────────────
load_dotenv(Path(__file__).resolve().parent / ".env")

for key in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
    if hasattr(st, "secrets") and key in st.secrets:
        os.environ[key] = st.secrets[key]

# ── Import agent AFTER env vars are set ──────────────────────────────────────
from app.agent import run_agent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 SHL Assessment Recommender")
st.caption("Find the right SHL Individual Test Solutions for your hiring needs.")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ended" not in st.session_state:
    st.session_state.ended = False

# ── Test type label map ───────────────────────────────────────────────────────
TYPE_LABELS = {
    "A": "🧠 Ability",
    "P": "🙂 Personality",
    "K": "📚 Knowledge",
    "S": "⚖️ Situational Judgment",
    "C": "🏆 Competency",
    "E": "📝 Exercise",
    "B": "📋 Biodata",
    "D": "📈 Development",
}

MAX_TURNS = 8

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("recommendations"):
            st.markdown("#### 📋 Recommended Assessments")
            for r in msg["recommendations"]:
                label = TYPE_LABELS.get(r["test_type"], r["test_type"])
                st.markdown(f"- [{r['name']}]({r['url']}) — {label}")

# ── Turn limit warning ────────────────────────────────────────────────────────
user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
if user_turns >= MAX_TURNS - 1:
    st.warning(f"⚠️ Approaching the {MAX_TURNS}-turn conversation limit.")

# ── End of conversation ───────────────────────────────────────────────────────
if st.session_state.ended:
    st.success("✅ Conversation complete!")
    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        st.session_state.ended = False
        st.rerun()
    st.stop()

# ── Chat input ────────────────────────────────────────────────────────────────
if user_turns >= MAX_TURNS:
    st.error("Conversation limit reached. Please start a new conversation.")
    if st.button("🔄 Start New Conversation"):
        st.session_state.messages = []
        st.session_state.ended = False
        st.rerun()
    st.stop()

prompt = st.chat_input("Describe the role you're hiring for...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Finding the best assessments..."):
            try:
                result = run_agent(history)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        st.write(result["reply"])

        if result.get("recommendations"):
            st.markdown("#### 📋 Recommended Assessments")
            for r in result["recommendations"]:
                label = TYPE_LABELS.get(r.get("test_type", "A"), r.get("test_type", ""))
                st.markdown(f"- [{r['name']}]({r['url']}) — {label}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["reply"],
        "recommendations": result.get("recommendations", []),
    })

    if result.get("end_of_conversation"):
        st.session_state.ended = True
        st.rerun()