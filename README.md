# 🎯 SHL Assessment Recommender

> **What** · A conversational AI agent that recommends SHL assessments from vague hiring intent  
> **Why** · Keyword search in assessment catalogs is slow and shallow — dialogue is faster and smarter  
> **Result** · Multi-turn agent with semantic retrieval, deployed live on Streamlit + FastAPI  
> **Demo** · [🚀 Live App](https://shl-recommender-zn87idffvs57lnhppyfvym.streamlit.app)

---

## ✨ What it does

Takes a hiring manager from vague intent → grounded shortlist in under 4 turns:

```
User:  "I need to hire a Java developer"
Agent: "What seniority level are you targeting?"
User:  "Mid-level, around 4 years"
Agent: Here are 5 assessments that fit → [Java 8 (New), OPQ32r, ...]
```

**Four conversational behaviors:**
- 🔍 **Clarify** — asks for role context before recommending
- 📋 **Recommend** — returns 1–10 assessments with SHL catalog URLs
- ✏️ **Refine** — updates shortlist when user changes constraints mid-conversation
- ⚖️ **Compare** — answers "what's the difference between X and Y?" from catalog data only

**Safety guardrails:**
- Refuses general hiring advice, legal questions, salary questions
- Blocks prompt injection attempts
- Every URL validated against scraped SHL catalog — no hallucinated links

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Streamlit UI            │  ← Conversational frontend
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         FastAPI /chat           │  ← Stateless REST API
└──────────────┬──────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
FAISS Retriever      LLM (Groq)
(k=15 semantic       LLaMA 3.1 8B
 search over         with structured
 120+ assessments)   JSON output
    │                     │
    └──────────┬──────────┘
               ▼
    Validated Recommendations
    (URLs checked against catalog)
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Groq + LLaMA 3.1 8B | Free tier, fast inference (<2s) |
| Embeddings | `all-MiniLM-L6-v2` | Lightweight, strong semantic similarity |
| Vector Store | FAISS (CPU) | No infra needed, loads in <1s |
| API | FastAPI | Async, schema validation via Pydantic |
| Frontend | Streamlit | Rapid deployment, zero frontend code |
| Scraper | Playwright + BeautifulSoup | JS-rendered SHL catalog |

---

## 📁 Project Structure

```
shl-recommender/
├── app/
│   ├── agent.py        # Core agent: retrieval + LLM + response parsing
│   ├── main.py         # FastAPI endpoints: GET /health, POST /chat
│   ├── scraper.py      # Playwright scraper for SHL catalog
│   └── build_index.py  # FAISS index builder
├── faiss_index/        # Pre-built vector index (committed to repo)
├── catalog.json        # Scraped SHL catalog (~120 assessments)
├── streamlit_app.py    # Streamlit chat UI
├── evaluate.py         # Evaluation script: Recall@10 + behavior probes
├── render.yaml         # Render.com deployment config
└── requirements.txt
```

---

## 🚀 Run Locally

**One-command setup:**

```bash
git clone https://github.com/niharika-kansal/shl-recommender.git
cd shl-recommender
pip install -r requirements.txt
```

Add your Groq API key (free at [console.groq.com](https://console.groq.com)):
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

**Run the Streamlit app:**
```bash
streamlit run streamlit_app.py
```

**Run the FastAPI backend:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Run evaluation:**
```bash
python evaluate.py --api http://localhost:8000
```

---

## 📊 API Specification

### `POST /chat`

Stateless — send full conversation history every call.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level are you targeting?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java developer.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

**Test type codes:** A=Ability · P=Personality · K=Knowledge · S=Situational Judgment · C=Competency

### `GET /health`
```json
{"status": "ok"}
```

---

## 📈 Evaluation

Evaluated on 10 conversation traces across 4 categories:

| Metric | Description |
|--------|-------------|
| Schema compliance | Every response matches the API schema |
| Turn cap | Recommendations within 8 turns |
| Catalog-only URLs | No hallucinated assessment links |
| Recall@10 | Fraction of relevant assessments in top-10 |
| Behavior probes | Vague query → clarify, off-topic → refuse, refinement → update |

```bash
python evaluate.py --api https://your-deployment-url.onrender.com
```

---

## 🎨 Design Choices & Trade-offs

**Why FAISS over ChromaDB?**  
FAISS loads from disk in <1s with no server required — critical for cold-start free-tier deployments.

**Why Groq + LLaMA over Gemini?**  
Groq's free tier has no daily quota exhaustion. LLaMA 3.1 8B follows structured JSON output reliably.

**Why stateless API?**  
The assignment requires stateless design — full conversation history sent per request. This scales horizontally with zero session storage.

**Why enforce vague-query guard in code?**  
LLM prompts alone are non-deterministic. A Python `is_vague()` heuristic enforces clarification on turn 1 regardless of LLM behavior.

**What didn't work:**  
- `gemini-1.5-flash` → deprecated, replaced with Groq
- Pushing `venv/` to GitHub → fixed with `.gitignore`
- `faiss-cpu==1.8.0` → incompatible with Python 3.14, pinned to latest

---

## 🔗 Links

- 📺 **Live Demo:** [Streamlit App](https://shl-recommender-zn87idffvs57lnhppyfvym.streamlit.app)
- 📦 **SHL Catalog:** [shl.com/products/product-catalog](https://www.shl.com/products/product-catalog/)
- 🤖 **LLM:** [Groq Console](https://console.groq.com)
