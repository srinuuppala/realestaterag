"""Retrieval.

Dense search alone misses exact tokens that matter enormously here — a RERA
number, a price written as "1.65 crore", a project code. Keyword search alone
misses paraphrase ("what does it cost?" vs "price range"). So both run, and their
*rankings* are fused with reciprocal rank fusion (their raw scores live on
incomparable scales, so fusing those directly would be meaningless).

Refusal is a retrieval decision, not a prompting one. A question only reaches the
generator if it clears a relevance gate: either a chunk is within the semantic
floor, or the question contains an identifier (a RERA number, a project code)
that literally occurs in the corpus — the one case where embeddings are weak and
a verbatim match is decisive. If neither holds, the chain answers "I don't know"
and never calls the model, so there is nothing to hallucinate from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.utils import get_logger, timed

log = get_logger(__name__)

RRF_K = 60  # Reciprocal-rank-fusion damping constant (the value from the original paper).

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens — keeps `p52100034899` and `1.65` searchable."""
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk plus the evidence for why it was chosen."""

    document: Document
    score: float
    rank: int

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def title(self) -> str:
        return self.document.metadata.get("title", self.source)

    @property
    def category(self) -> str:
        return self.document.metadata.get("category", "General")

    @property
    def project(self) -> str:
        return self.document.metadata.get("project", "All projects")

    @property
    def body(self) -> str:
        """Chunk text without the provenance line prepended at index time."""
        return self.document.page_content.split("]\n", 1)[-1].strip()

    @property
    def snippet(self) -> str:
        text = self.body
        return text[:300].rsplit(" ", 1)[0] + "…" if len(text) > 300 else text


class HybridRetriever:
    """Dense (FAISS) + sparse (BM25), fused with reciprocal rank fusion."""

    def __init__(self, store: FAISS, chunks: list[Document]) -> None:
        settings = get_settings()
        self._store = store
        self._chunks = chunks
        self._top_k = settings.top_k
        self._fetch_k = settings.fetch_k
        self._max_distance = settings.max_distance
        self._dense_weight = settings.dense_weight

        tokenized = [tokenize(chunk.page_content) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)
        # Vocabulary of "identifier-like" tokens actually present in the corpus:
        # long, digit-bearing strings such as `p52100034899` or `006712`.
        self._identifiers = {
            token
            for document in tokenized
            for token in document
            if len(token) >= 6 and any(character.isdigit() for character in token)
        }

        log.info(
            "Retriever ready — %d chunks, top_k=%d, fetch_k=%d, dense_weight=%.2f",
            len(chunks), self._top_k, self._fetch_k, self._dense_weight,
        )

    @timed("retrieve")
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        dense = self._dense(query)
        sparse = self._sparse(query)

        # The relevance gate. Everything downstream assumes it has passed.
        if not dense and not self.identifier_hit(query):
            log.info("Out of scope — no semantic match and no identifier: %r", query[:70])
            return []

        fused = self._fuse(dense, sparse)
        return [
            RetrievedChunk(document=document, score=score, rank=rank)
            for rank, (document, score) in enumerate(fused[: self._top_k], start=1)
        ]

    # -- individual strategies ----------------------------------------------
    def _dense(self, query: str) -> list[Document]:
        """Vector search, cut off at the semantic floor.

        FAISS returns L2 distance over normalised vectors, so 0 is identical and
        2 is opposite; anything past `max_distance` is noise.
        """
        try:
            scored = self._store.similarity_search_with_score(query, k=self._fetch_k)
        except Exception as error:  # noqa: BLE001 - fall back to keywords, never crash
            log.error("Dense retrieval failed: %s", error)
            return []

        kept = [document for document, distance in scored if distance <= self._max_distance]
        if scored and not kept:
            log.debug("Every dense hit was past the floor (best=%.3f)", scored[0][1])
        return kept

    def _sparse(self, query: str) -> list[Document]:
        """BM25 keyword ranking. Contributes to ordering, never to the gate."""
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[: self._fetch_k]

        best = float(ranked[0][1]) if ranked else 0.0
        # Drop the long tail of chunks that merely share a common word with the query.
        cutoff = max(best * 0.25, 1.0)
        return [self._chunks[index] for index, score in ranked if score >= cutoff]

    def identifier_hit(self, query: str) -> bool:
        """True when the question quotes an identifier that exists in the corpus.

        Embeddings are unreliable on bare strings like `P52100034899`, but a
        literal match on one is about as strong a relevance signal as there is.
        """
        return any(token in self._identifiers for token in tokenize(query))

    # -- fusion --------------------------------------------------------------
    def _fuse(self, dense: list[Document], sparse: list[Document]) -> list[tuple[Document, float]]:
        """score(d) = Σᵢ weightᵢ / (K + rankᵢ(d)) across both rankings."""
        scores: dict[str, float] = {}
        documents: dict[str, Document] = {}

        for weight, ranking in (
            (self._dense_weight, dense),
            (1.0 - self._dense_weight, sparse),
        ):
            for rank, document in enumerate(ranking, start=1):
                key = self._key(document)
                documents[key] = document
                scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(documents[key], score) for key, score in ordered]

    @staticmethod
    def _key(document: Document) -> str:
        return f"{document.metadata.get('source')}#{document.metadata.get('chunk_id')}"
