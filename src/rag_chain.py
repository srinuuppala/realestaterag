"""The RAG chain.

    question ─► condense (history-aware)
             ─► hybrid retrieve
             ─► guard: nothing relevant? say so, and stop
             ─► grounded prompt (numbered context blocks)
             ─► stream answer
             ─► attach citations

The prompt is the anti-hallucination control: the model is given numbered
context blocks and told to cite them, and is explicitly instructed to refuse
rather than fill gaps from memory. The retriever's relevance floor is the second
control — if nothing clears it, the model is never called at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.llm import LLMNotConfiguredError, stream
from src.memory import ConversationMemory, condense
from src.retriever import HybridRetriever, RetrievedChunk
from src.utils import get_logger

log = get_logger(__name__)

NO_ANSWER = (
    "I could not find anything about that in the real estate knowledge base. "
    "It covers Skyline Horizon Developers, Meridian Greens Realty and Urban Nest "
    "Infrastructures — their six projects, brochures, payment plans, RERA filings, "
    "policies, FAQs and buying guides. Try asking about one of those, or rephrase "
    "with the project name."
)

SYSTEM_PROMPT = """You are the Real Estate AI Assistant for a property company.

You answer ONLY from the numbered context blocks supplied with each question.

Rules, in order of priority:
1. Never state a fact that is not in the context. No prices, dates, RERA numbers,
   areas or policy terms from memory or inference — if it is not in the context,
   you do not know it.
2. Cite every factual claim inline with the block number it came from, like [1] or [2][3].
3. If the context does not answer the question, say exactly what is missing and
   name what you *can* help with. Do not guess and do not pad.
4. If the context partially answers it, give what you have and state plainly
   which part is not covered.
5. If the question is ambiguous (e.g. it names no project but several are
   relevant), ask one short clarifying question, or answer for each candidate
   and label them clearly.
6. Quote figures exactly as written in the context. Do not convert, round or
   recalculate them.

Style: direct and factual, like a well-briefed sales desk. Short paragraphs or a
tight list. No sales language, no invented reassurance."""


@dataclass
class RagResponse:
    """The result of one turn."""

    answer: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    standalone_question: str = ""
    grounded: bool = True

    @property
    def citations(self) -> list[dict]:
        """Unique source documents, in the order the context presented them."""
        seen: dict[str, dict] = {}
        for index, chunk in enumerate(self.chunks, start=1):
            if chunk.source in seen:
                seen[chunk.source]["blocks"].append(index)
                continue
            seen[chunk.source] = {
                "blocks": [index],
                "source": chunk.source,
                "title": chunk.title,
                "category": chunk.category,
                "project": chunk.project,
                "file_type": chunk.document.metadata.get("file_type", ""),
                "snippet": chunk.snippet,
            }
        return list(seen.values())


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Numbered blocks, each carrying its own provenance so the model can cite it."""
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        body = chunk.document.page_content.split("]\n", 1)[-1]
        blocks.append(
            f"[{index}] source: {chunk.source} | project: {chunk.project} | type: {chunk.category}\n{body}"
        )
    return "\n\n---\n\n".join(blocks)


class RagChain:
    """Owns retrieval + generation for one knowledge base."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever
        self._llm_available = get_settings().llm_configured

    def answer(self, question: str, memory: ConversationMemory) -> Iterator[tuple[str, RagResponse]]:
        """Streams `(token, response)` pairs; the final response carries the citations.

        Yielding the response object alongside each token lets the UI render
        citations the moment retrieval finishes, before generation completes.
        """
        standalone = condense(question, memory)
        chunks = self._retriever.retrieve(standalone)

        response = RagResponse(chunks=chunks, standalone_question=standalone)

        if not chunks:
            log.info("No chunk cleared the relevance floor for %r", standalone)
            response.answer = NO_ANSWER
            response.grounded = False
            yield NO_ANSWER, response
            return

        if not self._llm_available:
            response.answer = self._extractive_answer(standalone, chunks)
            yield response.answer, response
            return

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *memory.as_messages(),
            HumanMessage(
                content=(
                    f"Context blocks:\n\n{build_context(chunks)}\n\n"
                    f"Question: {standalone}\n\n"
                    "Answer using only the blocks above, citing them inline."
                )
            ),
        ]

        try:
            for token in stream(messages):
                response.answer += token
                yield token, response
        except LLMNotConfiguredError:
            response.answer = self._extractive_answer(standalone, chunks)
            yield response.answer, response
        except Exception as error:  # noqa: BLE001 - surface, never crash the app
            log.error("Generation failed: %s", error)
            message = (
                "The language model could not be reached, so I cannot compose an answer right now. "
                "The passages I retrieved for your question are cited below — they contain the information."
            )
            response.answer = message
            response.grounded = False
            yield message, response

    @staticmethod
    def _extractive_answer(question: str, chunks: list[RetrievedChunk]) -> str:
        """Keyless fallback: return the evidence verbatim rather than nothing.

        This is not a stand-in for generation and says so — but it means a
        deployment without an API key still demonstrates real retrieval instead
        of an error screen.
        """
        lines = [
            "**No language model is configured, so I cannot summarise — here are the exact passages "
            "the knowledge base returned for your question.**",
            "",
        ]
        for index, chunk in enumerate(chunks[:3], start=1):
            body = chunk.document.page_content.split("]\n", 1)[-1].strip()
            lines.append(f"**[{index}] {chunk.title}** — {chunk.category}")
            lines.append(f"> {body[:700]}{'…' if len(body) > 700 else ''}")
            lines.append("")
        return "\n".join(lines)
