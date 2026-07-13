# Real Estate AI Assistant — RAG

A retrieval-augmented assistant over a 92-document real estate knowledge base. It answers from the documents,
cites every source, holds a multi-turn conversation, and refuses when the corpus has nothing to say.

**Python 3.11 · Streamlit · LangChain · FAISS · Sentence-Transformers · BM25 · Groq (Llama 3.3 70B)**

Runs end to end on free infrastructure: Groq's free API tier (no credit card), local embeddings that cost
nothing per query, and Streamlit Community Cloud. Total cost: ₹0.

| | |
|---|---|
| Live app | _add your Streamlit Cloud URL_ |
| Demo login | `demo` / `realestate2026` |
| Architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Deployment | [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |

---

## The problem this actually solves

A RAG demo is easy. A RAG system you would put in front of a customer has to be right about three things:

**1. It has to find the answer.** Dense retrieval alone fails on the things real estate questions are *made of* —
a RERA number (`P52100034899`), a price written as "1.65 crore", a project code. Keyword search alone fails on
paraphrase. So both run, and their rankings are fused with reciprocal rank fusion. Verified: the correct document
lands in the top 5 for every test query, and a bare RERA number resolves to exactly the right project.

**2. It has to know when it doesn't know.** Refusal here is a *retrieval* decision, not a prompt instruction.
A question reaches the model only if a chunk clears a semantic floor, or the question quotes an identifier that
literally occurs in the corpus. If neither holds, the model is never called — there is nothing to hallucinate
from. "Who won the cricket world cup in 2011?" returns zero chunks and a straight refusal.

**3. It has to be checkable.** Every answer cites numbered context blocks, and the citation panel shows the source
file, its format, its category, the project, and the passage that was actually used. A user can audit any claim.

## Features

- **Multi-turn memory** — follow-ups are rewritten into standalone questions before retrieval. "What about *its*
  payment plan?" becomes "What is the payment plan for Meridian Lakeview Villas?", which is a question the
  retriever can actually answer.
- **Streaming answers** with a typing cursor.
- **All four corpus formats**, plus TXT and CSV: PDF, DOCX, HTML, Markdown.
- **Metadata-aware chunking** — every chunk carries its project, builder, document category and section heading.
- **Login screen**, session chat history, one-click re-ask, clear conversation, dark mode.
- **Runs without any API key at all** in extractive mode: retrieval and citations still work, and the app returns
  the source passages instead of a summary. A keyless deployment demonstrates the retrieval system rather than
  showing an error screen.
- **Zero running cost** — embeddings run locally on CPU (no per-query charge), and Groq's free tier covers the
  generation.
- **20 offline tests**, `ruff`-clean, Docker-ready.

---

## Run it

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # optional — add an API key for generated answers
streamlit run app.py        # http://localhost:8501
```

Sign in with `demo` / `realestate2026`.

The first run embeds the corpus (~30 s) and writes a FAISS index to `data/faiss_index/`. Subsequent runs load it
in under a second; the index is rebuilt only when a document is added, removed or edited.

### Connecting the LLM (free)

**Groq is the default.** It has a genuinely free tier — no credit card, no trial clock — serving Llama on an
OpenAI-compatible endpoint.

1. Sign up at [console.groq.com](https://console.groq.com) with email or Google. No card.
2. **API Keys → Create API Key**, copy it.
3. Put it in `.env`:

```bash
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
```

That is the whole setup. Free-tier limits are ~30 requests/minute and ~1,000/day on the 70B model — far more than
a demo or an interview walkthrough needs. A 429 is retried with exponential backoff rather than failing the turn.

| Model | Quality | Free daily limit |
|---|---|---|
| `llama-3.3-70b-versatile` *(default)* | Best | ~1,000 requests |
| `llama-3.1-8b-instant` | Good, very fast | ~14,400 requests |

Switching provider is two variables — `LLM_BASE_URL` and `LLM_MODEL`. OpenAI, Together, OpenRouter and a local
Ollama server all work unchanged, because nothing in the codebase knows which provider is behind the endpoint.

### Tests and lint

```bash
pip install -r requirements-dev.txt
pytest                                    # 20 tests, ~1s, no network needed
ruff check src app.py tests
```

The suite swaps the neural embedding for a deterministic TF-IDF stand-in with the same vector geometry, so
retrieval, fusion, the relevance gate and citation assembly are all exercised offline in about a second.

### Docker

```bash
docker build -t realestate-rag .
docker run -p 8501:8501 --env-file .env realestate-rag
```

The embedding model is baked into the image, so a cold container answers immediately.

---

## Project structure

```
RealEstateRAG/
├── app.py                  Streamlit UI — login, chat, streaming, citations, sidebar, dark mode
├── src/
│   ├── config.py           Typed settings, all from the environment
│   ├── auth.py             Session auth, constant-time credential comparison
│   ├── loader.py           PDF · DOCX · HTML · MD · TXT · CSV → chunks with project/builder/category metadata
│   ├── embedding.py        Sentence-Transformers, normalised vectors
│   ├── vectorstore.py      FAISS build / persist / load, fingerprint-based invalidation
│   ├── retriever.py        Hybrid dense + BM25, reciprocal rank fusion, the relevance gate
│   ├── llm.py              Any OpenAI-compatible endpoint, retries, streaming
│   ├── memory.py           Conversation window + history-aware question condensation
│   ├── rag_chain.py        Orchestration, grounded prompt, citation assembly
│   └── utils.py            Logging, retry/backoff, corpus fingerprint, text normalisation
├── tests/                  20 tests — loader, retrieval, the gate, citations, memory
├── data/knowledge_base/    92 documents: 21 PDF · 21 DOCX · 23 HTML · 27 MD
├── docs/                   ARCHITECTURE.md · DEPLOYMENT.md · DATASET.md
├── Dockerfile · requirements.txt · .env.example · .streamlit/config.toml
```

## Try these

| Question | What it exercises |
|---|---|
| *What is the payment plan for Skyline Horizon Towers?* | Retrieval across a PDF payment schedule |
| *Which projects are under ₹1 crore, and where are they?* | Cross-document reasoning over six brochures |
| *And what is the possession date for that one?* | Multi-turn memory — the pronoun is resolved before retrieval |
| *P52100034899* | The identifier path: a bare RERA number resolves to its project |
| *What is the cancellation policy at Urban Nest?* | DOCX policy retrieval with citations |
| *Who won the cricket world cup in 2011?* | Refusal — zero chunks retrieved, model never called |

## Known limits

- The semantic floor (`MAX_DISTANCE=1.25`, i.e. cosine ≥ 0.22) is tuned for `all-MiniLM-L6-v2`. Swap the
  embedding model and this needs re-tuning — it is one environment variable.
- Conversation memory is per-session and in-process; it does not survive a restart. A persistent store is a
  drop-in behind `ConversationMemory`, but the brief asked for session history and that is what this does.
- Citations point at the source document, not a page or line number. PDF page anchors would require carrying page
  offsets through the splitter.
