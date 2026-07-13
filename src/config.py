"""Central configuration.

Every tunable lives here and is sourced from the environment, so the same image
runs locally, in Docker and on Streamlit Community Cloud without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- Paths -------------------------------------------------------------
    knowledge_base_dir: Path = field(
        default_factory=lambda: Path(_env("KNOWLEDGE_BASE_DIR") or PROJECT_ROOT / "data" / "knowledge_base")
    )
    index_dir: Path = field(
        default_factory=lambda: Path(_env("INDEX_DIR") or PROJECT_ROOT / "data" / "faiss_index")
    )

    # --- Chunking ----------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Retrieval ---------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    fetch_k: int = field(default_factory=lambda: _env_int("FETCH_K", 20))
    # Cosine distance above which a chunk is considered irrelevant. FAISS returns
    # L2 distance on normalised vectors, so 0 = identical and 2 = opposite.
    max_distance: float = field(default_factory=lambda: _env_float("MAX_DISTANCE", 1.25))
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", 0.6))

    # --- LLM ---------------------------------------------------------------
    # Groq by default: a free, no-credit-card tier on an OpenAI-compatible
    # endpoint. GROQ_API_KEY is checked first, then OPENAI_API_KEY, so pointing
    # this at OpenAI, Together, OpenRouter or a local Ollama server is a
    # two-variable change and nothing else in the codebase notices.
    llm_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY") or _env("OPENAI_API_KEY"))
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL")
        or _env("OPENAI_BASE_URL")
        or "https://api.groq.com/openai/v1"
    )
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL") or "llama-3.3-70b-versatile")
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.0))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 800))
    request_timeout: int = field(default_factory=lambda: _env_int("REQUEST_TIMEOUT", 60))
    # Groq's free tier allows 30 requests/minute; a 429 is normal under load, so
    # retries back off rather than failing the turn.
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))

    # --- Memory ------------------------------------------------------------
    # How many previous turns are replayed to the model for follow-up questions.
    memory_window: int = field(default_factory=lambda: _env_int("MEMORY_WINDOW", 6))

    # --- Auth --------------------------------------------------------------
    auth_username: str = field(default_factory=lambda: _env("AUTH_USERNAME") or "demo")
    auth_password: str = field(default_factory=lambda: _env("AUTH_PASSWORD") or "realestate2026")

    # --- Logging -----------------------------------------------------------
    log_level: str = field(default_factory=lambda: (_env("LOG_LEVEL") or "INFO").upper())

    @property
    def llm_configured(self) -> bool:
        """False on a keyless deployment — the app then answers in extractive mode."""
        return bool(self.llm_api_key)

    @property
    def provider(self) -> str:
        """Human-readable provider name, for the sidebar."""
        if "groq.com" in self.llm_base_url:
            return "Groq"
        if "openai.com" in self.llm_base_url:
            return "OpenAI"
        if "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url:
            return "Local"
        return "Custom"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
