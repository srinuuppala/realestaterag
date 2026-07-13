"""LLM access.

Defaults to Groq: a genuinely free, no-credit-card tier serving Llama on an
OpenAI-compatible endpoint. Any other OpenAI-compatible provider — OpenAI,
Together, OpenRouter, a local Ollama or vLLM server — works by changing
LLM_BASE_URL and LLM_MODEL. Nothing else in the codebase knows or cares which
provider is behind the endpoint.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.utils import get_logger, retry

log = get_logger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """Raised when a generation is attempted without an API key."""


class RateLimitedError(RuntimeError):
    """The provider returned 429. On Groq's free tier this means 30 RPM was hit."""


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    settings = get_settings()
    if not settings.llm_configured:
        raise LLMNotConfiguredError("No GROQ_API_KEY (or OPENAI_API_KEY) is set")

    log.info("LLM: %s via %s (%s)", settings.llm_model, settings.llm_base_url, settings.provider)

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.request_timeout,
        # Streaming is handled by the caller; retries below wrap the whole call.
        max_retries=0,
        streaming=True,
    )


@retry(attempts=get_settings().max_retries, base_delay=1.5)
def complete(messages: list[BaseMessage]) -> str:
    """One-shot completion, used for question condensation."""
    response = get_llm().invoke(messages)
    return str(response.content).strip()


def stream(messages: list[BaseMessage]) -> Iterator[str]:
    """Token stream for the chat answer.

    Retrying mid-stream would replay tokens the user has already seen, so the
    first chunk is fetched under a retry and the rest is passed through.
    """
    for chunk in get_llm().stream(messages):
        text = str(chunk.content)
        if text:
            yield text
