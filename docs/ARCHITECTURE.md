# Architecture

## Pipeline

```
  question
     │
     ▼
  condense ──────────── memory (last 6 turns)
     │  "what about its payment plan?" → "what is the payment plan for Meridian Lakeview Villas?"
     ▼
  ┌──────────────── HybridRetriever ────────────────┐
  │  dense: FAISS over MiniLM  ──┐                  │
  │                              ├─► RRF fusion ──► │ top-5 chunks
  │  sparse: BM25 over tokens  ──┘                  │
  │                                                 │
  │  RELEVANCE GATE: pass only if a chunk clears    │
  │  the semantic floor, or the question quotes an  │
  │  identifier present in the corpus.              │
  └─────────────────────────────────────────────────┘
     │                                   │
     │ nothing passed                    │ chunks
     ▼                                   ▼
  refusal (model never called)     numbered context blocks
                                         │
                                         ▼
                                   grounded prompt → stream → answer + citations
```

## The three decisions that matter

### 1. Hybrid retrieval, fused on rank

Real estate questions are full of tokens that embeddings handle badly: `P52100034899`, `1.65 crore`, `PRM/KA/RERA/1251`.
They are also full of paraphrase: "what does it cost" vs "price range". Neither retriever covers both, so both run.

Their scores are *not* comparable — FAISS returns an L2 distance, BM25 returns an unbounded relevance score — so
fusion is done on **rank**, not score:

```
score(d) = Σᵢ weightᵢ / (K + rankᵢ(d))          K = 60
```

Dense is weighted 0.6, sparse 0.4 (`DENSE_WEIGHT`). Measured on the corpus: the correct document is in the top 5
for every test query, and the bare RERA number `P52100034899` returns only Meridian Lakeview Villas documents.

### 2. Refusal is a retrieval decision

The usual approach — "say 'I don't know' if the context is irrelevant" in the system prompt — hands the decision
to the model, which is exactly the component that hallucinates. So the gate sits *before* generation:

```python
if not dense_hits and not identifier_hit(query):
    return []          # the LLM is never called
```

- **Semantic floor** — FAISS returns L2 distance over normalised vectors, so `distance = √(2 − 2·cos)`.
  `MAX_DISTANCE = 1.25` means cosine ≥ 0.22. For `all-MiniLM-L6-v2`, relevant question/passage pairs sit around
  0.35–0.65 and unrelated ones near 0.0–0.15, so the floor lands in the gap.
- **Identifier escape hatch** — the one case where embeddings are unreliable but a literal match is decisive.
  At index time the retriever collects every corpus token that is ≥ 6 characters and contains a digit; a question
  quoting one of them passes the gate regardless of vector distance.

The prompt *also* forbids ungrounded claims — but that is the second line of defence, not the first.

### 3. Follow-ups are rewritten before retrieval, not after

"What about its payment plan?" is unretrievable: it contains no project, no document type, nothing to match on.
`memory.condense()` rewrites it against the conversation into a standalone question, and *that* is what hits the
index. If the rewrite fails, or no LLM is configured, it falls back to the original question — a degraded rewrite
always beats a failed turn.

## Chunking and metadata

`RecursiveCharacterTextSplitter`, 900 characters with 150 of overlap, splitting on headings before paragraphs
before sentences. 92 documents → 187 chunks.

The corpus follows a strict `<code>_<category>.<ext>` convention, so `loader.classify()` derives structured
metadata from the filename alone:

| Field | Example | Why it exists |
|---|---|---|
| `project` | `Meridian Lakeview Villas` | Lets a citation say *which* project a passage is about |
| `builder` | `Meridian Greens Realty` | Groups the six projects under three builders |
| `category` | `Payment Plan` | Shown in the citation panel; disambiguates 6 near-identical brochures |
| `section` | `Construction-Linked Plan` | The nearest heading above the chunk |

Each chunk is prefixed with `[project · category]` before embedding. This measurably improves recall on questions
that name a project but not a document type — the commonest way people actually ask.

## Index lifecycle

The FAISS index is written to `data/faiss_index/` alongside a fingerprint (name + size + mtime of every source
file). On boot the fingerprint is recomputed and the index is rebuilt **only if it changed**. A cold start on an
unchanged corpus loads in under a second instead of re-embedding 187 chunks. A corrupt index is caught and
rebuilt rather than crashing the app.

## Failure modes, and what happens

| Failure | Behaviour |
|---|---|
| No `GROQ_API_KEY` | Extractive mode — retrieval and citations work; answers are the retrieved passages, clearly labelled |
| Groq 429 (free tier is 30 req/min) | Exponential backoff, 3 attempts (`utils.retry`) |
| LLM unreachable after retries | The answer slot explains it; the retrieved citations are still shown |
| A document fails to parse | Logged and skipped; the other 91 still load |
| Question condensation fails | Falls back to the raw question |
| BM25 or FAISS errors | The other retriever carries the query |
| Corrupt FAISS index | Detected on load, rebuilt |

## Module map

| Module | Owns | Depends on |
|---|---|---|
| `config.py` | Every tunable, read from env | — |
| `utils.py` | Logging, retry, fingerprint, text cleanup | config |
| `loader.py` | Reading 6 formats → chunks + metadata | config, utils |
| `embedding.py` | The sentence-transformer | config |
| `vectorstore.py` | FAISS build / save / load / invalidate | loader, embedding |
| `retriever.py` | Hybrid search, fusion, the relevance gate | config, vectorstore |
| `llm.py` | OpenAI-compatible client, retries, streaming | config, utils |
| `memory.py` | Conversation window, question condensation | config, llm |
| `rag_chain.py` | Orchestration, grounded prompt, citations | retriever, memory, llm |
| `auth.py` | Session login | config |
| `app.py` | Streamlit UI | all of the above |

Dependencies point one way. `retriever.py` does not know an LLM exists; `llm.py` does not know what a document is.
