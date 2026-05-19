"""
streamlit_app.py
----------------
A clean chat interface for the SHL Assessment Recommender.
Talks to the FastAPI backend at localhost:8000.

Run (from project root, with uvicorn already running):
    streamlit run streamlit_app.py

Or run both at once (two terminals):
    Terminal 1: uvicorn app.main:app --port 8000
    Terminal 2: streamlit run streamlit_app.py
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"
HEALTH_URL = "http://127.0.0.1:8000/health"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="centered",
)

st.title("🎯 SHL Assessment Recommender")
st.caption("Find the right SHL Individual Test Solutions for your hiring needs.")

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "ended" not in st.session_state:
    st.session_state.ended = False

# ── Health check ──────────────────────────────────────────────────────────────

def check_health():
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

if not check_health():
    st.error(
        "⚠️ Cannot reach the API at `localhost:8000`. "
        "Make sure uvicorn is running:\n\n"
        "```\nuvicorn app.main:app --reload --port 8000\n```"
    )
    st.stop()

# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Recommendations panel ─────────────────────────────────────────────────────

if st.session_state.recommendations:
    st.divider()
    st.subheader("📋 Recommended Assessments")
    for rec in st.session_state.recommendations:
        type_labels = {
            "A": "🧠 Ability", "P": "🧬 Personality", "K": "📚 Knowledge",
            "S": "⚖️ Situational Judgment", "C": "🏅 Competency",
            "B": "📋 Biodata", "E": "🎭 Exercise", "D": "🌱 Development",
        }
        label = type_labels.get(rec["test_type"], rec["test_type"])
        st.markdown(f"**[{rec['name']}]({rec['url']})** — {label}")

# ── Input ─────────────────────────────────────────────────────────────────────

if st.session_state.ended:
    st.info("Conversation complete. Refresh the page to start over.")
else:
    user_input = st.chat_input("Describe the role you're hiring for...")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Call the API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"messages": st.session_state.messages},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    reply = data["reply"]
                    recs = data.get("recommendations", [])
                    ended = data.get("end_of_conversation", False)

                    st.markdown(reply)

                    # Save assistant message and recommendations
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                    if recs:
                        st.session_state.recommendations = recs
                    if ended:
                        st.session_state.ended = True

                    st.rerun()

                except requests.exceptions.Timeout:
                    st.error("Request timed out (>30s). The model may be loading. Try again.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error {e.response.status_code}: {e.response.text}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# ── Reset button ──────────────────────────────────────────────────────────────

if st.session_state.messages:
    if st.button("🔄 Start new conversation"):
        st.session_state.messages = []
        st.session_state.recommendations = []
        st.session_state.ended = False
        st.rerun()