"""The chain must never answer without evidence, and must always attach citations."""

from src.memory import ConversationMemory
from src.rag_chain import NO_ANSWER, RagChain, build_context


def _run(chain: RagChain, question: str, memory: ConversationMemory):
    response = None
    for _token, latest in chain.answer(question, memory):
        response = latest
    return response


def test_a_question_with_no_evidence_is_refused(retriever, monkeypatch):
    monkeypatch.setattr(retriever, "retrieve", lambda _query: [])
    response = _run(RagChain(retriever), "What is the capital of France?", ConversationMemory())

    assert response.answer == NO_ANSWER
    assert response.grounded is False
    assert response.citations == []


def test_an_answered_question_carries_its_sources(retriever):
    response = _run(
        RagChain(retriever),
        "What is the payment plan for Skyline Horizon Towers?",
        ConversationMemory(),
    )

    assert response.chunks
    assert response.citations
    assert all(citation["blocks"] for citation in response.citations)
    assert any("sht_" in citation["source"] for citation in response.citations)


def test_citations_merge_repeated_sources_and_keep_block_numbers(retriever):
    response = _run(RagChain(retriever), "amenities at Horizon Business Park", ConversationMemory())

    sources = [citation["source"] for citation in response.citations]
    assert len(sources) == len(set(sources)), "one document should appear as one citation"

    cited_blocks = sorted(block for citation in response.citations for block in citation["blocks"])
    assert cited_blocks == list(range(1, len(response.chunks) + 1))


def test_context_blocks_are_numbered_and_attributed(retriever):
    chunks = retriever.retrieve("possession guidelines for Urban Nest Riverside")
    context = build_context(chunks)

    for index in range(1, len(chunks) + 1):
        assert f"[{index}] source:" in context


def test_memory_records_both_sides_of_a_turn_and_clears():
    memory = ConversationMemory()
    memory.add("user", "What is the price?")
    memory.add("assistant", "It is X.", [{"source": "a.pdf", "blocks": [1]}])

    assert [turn.role for turn in memory.turns] == ["user", "assistant"]
    assert memory.turns[-1].citations[0]["source"] == "a.pdf"

    memory.clear()
    assert memory.is_empty
