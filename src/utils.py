"""Cross-cutting helpers: logging, retries, fingerprinting and citation labels."""

from __future__ import annotations

import functools
import hashlib
import logging
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeVar

from src.config import get_settings

T = TypeVar("T")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Returns a logger, configuring the root handler exactly once."""
    global _configured
    if not _configured:
        logging.basicConfig(level=get_settings().log_level, format=_LOG_FORMAT)
        # Silence the noisiest third-party loggers; their INFO output is not ours.
        for noisy in ("httpx", "sentence_transformers", "urllib3", "faiss"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True
    return logging.getLogger(name)


log = get_logger(__name__)


def retry(attempts: int = 3, base_delay: float = 1.0, exceptions: tuple[type[Exception], ...] = (Exception,)):
    """Retries a callable with exponential backoff.

    Used for LLM calls, where a transient 429/503 is common and a hard failure
    would mean losing the user's question.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as error:  # noqa: PERF203 - retry is the point
                    last_error = error
                    if attempt == attempts:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    log.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        func.__name__, attempt, attempts, error, delay,
                    )
                    time.sleep(delay)
            raise RuntimeError(f"{func.__name__} failed after {attempts} attempts") from last_error

        return wrapper

    return decorator


def timed(label: str):
    """Logs how long a step took — retrieval latency is worth watching."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            log.debug("%s took %.2fs", label, time.perf_counter() - start)
            return result

        return wrapper

    return decorator


def fingerprint(paths: Iterable[Path]) -> str:
    """Content hash of the corpus.

    The index is rebuilt only when a file is added, removed or edited — not on
    every boot, which would make cold starts unbearable.
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.name).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
    return digest.hexdigest()[:16]


def normalise_whitespace(text: str) -> str:
    """Collapses the ragged whitespace that PDF and HTML extraction leaves behind."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def humanise(filename: str) -> str:
    """`sht_payment_plan.pdf` → `Sht Payment Plan` — a readable citation label."""
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return stem.title()
