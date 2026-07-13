"""Embedding model.

Sentence-Transformers runs locally, so retrieval works with no API key and no
per-query cost. `all-MiniLM-L6-v2` is 384-dimensional and ~90 MB, which keeps
the app inside the memory budget of free hosting tiers.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_settings
from src.utils import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Loads the model once per process — it is ~90 MB and slow to initialise."""
    settings = get_settings()
    log.info("Loading embedding model: %s", settings.embedding_model)

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        # Normalised vectors turn FAISS's L2 distance into a monotonic function
        # of cosine similarity, which is what the relevance threshold assumes.
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
