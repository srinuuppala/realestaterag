"""Test fixtures.

The production embedding model is a 90 MB neural network that must be
downloaded. The suite instead uses a deterministic hashed TF-IDF embedding built
from the corpus itself. It has the same geometry as the real one — L2-normalised
vectors, so FAISS distance is a monotonic function of cosine similarity — which
means retrieval, fusion and the relevance gate can all be exercised offline in
about a second.

Its *distance scale* is different (a bag-of-words model is far less confident
than a sentence transformer), so the fixture widens the semantic floor for the
tests. The gate itself is tested directly rather than through the stub, in
`test_retriever.py::test_gate_*`.
"""

from __future__ import annotations

import math
import os
import zlib
from collections import Counter

os.environ.setdefault("MAX_DISTANCE", "1.45")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import pytest  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

import src.vectorstore as vectorstore  # noqa: E402
from src.loader import build_corpus  # noqa: E402
from src.retriever import HybridRetriever, tokenize  # noqa: E402

DIMENSIONS = 512


class TfidfEmbeddings(Embeddings):
    """Hashed TF-IDF vectors, L2-normalised. Deterministic, offline, instant."""

    def __init__(self, corpus: list[str]) -> None:
        documents = [set(tokenize(text)) for text in corpus]
        total = len(documents) or 1
        frequencies = Counter(token for document in documents for token in document)
        self._idf = {token: math.log(total / (1 + count)) + 1.0 for token, count in frequencies.items()}

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for token, count in Counter(tokenize(text)).items():
            # crc32, not the built-in hash(): PYTHONHASHSEED is randomised per
            # process, and a fixture that changes between runs is not a fixture.
            bucket = zlib.crc32(token.encode()) % DIMENSIONS
            vector[bucket] += (1 + math.log(count)) * self._idf.get(token, 0.0)

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture(scope="session")
def chunks():
    return build_corpus()


@pytest.fixture(scope="session")
def retriever(chunks, monkeypatch_session):
    embeddings = TfidfEmbeddings([chunk.page_content for chunk in chunks])
    monkeypatch_session.setattr(vectorstore, "get_embeddings", lambda: embeddings)
    return HybridRetriever(vectorstore.build_index(chunks), chunks)


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    yield patcher
    patcher.undo()
