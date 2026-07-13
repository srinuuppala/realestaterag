# Deployment guide

## Streamlit Community Cloud (free, recommended)

1. Push this folder to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at the repo, branch `main`, main file `app.py`.
4. **Advanced settings → Secrets**, paste (see `.streamlit/secrets.toml.example`):

   ```toml
   GROQ_API_KEY = "gsk_..."
   LLM_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"
   AUTH_USERNAME = "demo"
   AUTH_PASSWORD = "realestate2026"
   ```

   Streamlit exposes secrets as environment variables, which is exactly what `config.py` reads — no code change.

   Get the Groq key at [console.groq.com](https://console.groq.com) → API Keys. Free, no credit card.

5. Deploy. The first boot installs the dependencies, downloads the 90 MB embedding model and builds the FAISS
   index; expect 3–5 minutes. Afterwards the index is cached on disk and cold starts are fast.

**No API key?** Deploy anyway. The app runs in extractive mode: login, retrieval, citations and the refusal path
all work, and the assistant returns the retrieved passages instead of a generated summary — clearly labelled as
such in the sidebar and in each answer.

### Memory

Community Cloud gives about 1 GB. `all-MiniLM-L6-v2` (90 MB) plus a 187-chunk FAISS index fits comfortably. If
you swap in a larger embedding model, check the footprint first.

## Docker (any host)

```bash
docker build -t realestate-rag .
docker run -p 8501:8501 --env-file .env realestate-rag
```

The image bakes in the embedding model, so the container answers its first question without downloading anything.
Health check: `GET /_stcore/health`.

Works unchanged on Render (Web Service → Docker), Fly.io, Railway or Hugging Face Spaces.

## Hugging Face Spaces

Create a Space with the **Docker** SDK, push this folder, and add `OPENAI_API_KEY` under *Settings → Variables and
secrets*. The `Dockerfile` needs no changes.

## Free-tier limits worth knowing

| Service | Free allowance | What it means here |
|---|---|---|
| Groq | ~30 req/min, ~1,000 req/day on Llama 3.3 70B | Each chat turn costs 2 calls (condense + answer). Plenty for a demo |
| Streamlit Cloud | ~1 GB RAM, sleeps when idle | MiniLM (90 MB) + a 187-chunk index fits comfortably |
| Embeddings | Local CPU, unlimited | No per-query cost — retrieval is free forever |

## Configuration reference

Every setting is an environment variable; the defaults in `.env.example` are the ones the app was tuned with.

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | Free key from console.groq.com. Blank ⇒ extractive mode |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Point at OpenAI, Together, OpenRouter, Ollama… |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Or `llama-3.1-8b-instant` for 14× the daily quota |
| `TEMPERATURE` | `0.0` | Grounded answers should not be creative |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Changing this means re-tuning `MAX_DISTANCE` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | |
| `TOP_K` / `FETCH_K` | `5` / `20` | Chunks sent to the model / candidates fused |
| `MAX_DISTANCE` | `1.25` | The semantic floor — cosine ≥ 0.22. Lower it to refuse more readily |
| `DENSE_WEIGHT` | `0.6` | Dense vs BM25 weight in rank fusion |
| `MEMORY_WINDOW` | `6` | Turns replayed to the model |
| `AUTH_USERNAME` / `AUTH_PASSWORD` | `demo` / `realestate2026` | **Change these before any real deployment** |

## Post-deploy checklist

- [ ] The login screen appears and rejects a wrong password
- [ ] *"What is the payment plan for Skyline Horizon Towers?"* returns an answer with citations
- [ ] A follow-up — *"and when is possession?"* — resolves against the previous question
- [ ] *"Who won the cricket world cup in 2011?"* is refused, with no citations
- [ ] **Clear conversation** empties the transcript
- [ ] The theme toggle works and the sidebar reports 92 documents
