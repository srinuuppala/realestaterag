"""Conversation memory.

Two jobs:
  1. Hold the transcript for the UI.
  2. Give the retriever a *self-contained* question. "What about its payment
     plan?" is unretrievable on its own; rewritten against the history it
     becomes "What is the payment plan for Meridian Lakeview Villas?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.config import get_settings
from src.llm import LLMNotConfiguredError, complete
from src.utils import get_logger

log = get_logger(__name__)

Role = Literal["user", "assistant"]

CONDENSE_SYSTEM = SystemMessage(
    content=(
        "You rewrite a follow-up question so it can be understood on its own.\n"
        "Resolve every pronoun and implicit reference using the conversation.\n"
        "Return ONLY the rewritten question — no preamble, no explanation.\n"
        "If the question already stands alone, return it unchanged."
    )
)


@dataclass
class Turn:
    """One message, plus the citations that supported it."""

    role: Role
    content: str
    citations: list[dict] = field(default_factory=list)


@dataclass
class ConversationMemory:
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: Role, content: str, citations: list[dict] | None = None) -> None:
        self.turns.append(Turn(role=role, content=content, citations=citations or []))

    def clear(self) -> None:
        self.turns.clear()

    @property
    def is_empty(self) -> bool:
        return not self.turns

    def window(self) -> list[Turn]:
        """The last N turns — a sliding window keeps the prompt bounded."""
        return self.turns[-get_settings().memory_window :]

    def as_messages(self) -> list[BaseMessage]:
        return [
            HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content)
            for turn in self.window()
        ]

    def transcript(self) -> str:
        return "\n".join(
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}" for turn in self.window()
        )


def condense(question: str, memory: ConversationMemory) -> str:
    """Rewrites a follow-up into a standalone question.

    Falls back to the original question if there is no history, no LLM, or the
    rewrite fails — a degraded rewrite is always better than a failed turn.
    """
    if memory.is_empty:
        return question

    try:
        rewritten = complete(
            [
                CONDENSE_SYSTEM,
                HumanMessage(
                    content=f"Conversation so far:\n{memory.transcript()}\n\nFollow-up question: {question}"
                ),
            ]
        )
    except LLMNotConfiguredError:
        return question
    except Exception as error:  # noqa: BLE001 - never fail a turn on the rewrite
        log.warning("Question condensation failed (%s) — using the original question", error)
        return question

    if not rewritten or len(rewritten) > 400:
        return question

    if rewritten.lower() != question.lower():
        log.info("Condensed: %r → %r", question, rewritten)
    return rewritten
