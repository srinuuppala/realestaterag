"""Retrieval must surface the right document — and refuse when there is none."""

import pytest


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("What is the payment plan for Skyline Horizon Towers?", "sht_payment_plan.pdf"),
        ("cancellation and refund policy at Urban Nest", "urbannest_cancellation_refund_policy.docx"),
        ("amenities at Horizon Business Park", "hbp_amenities_guide.md"),
        ("floor plans for Meridian Lakeview Villas", "mlv_floor_plans.md"),
        ("possession timeline for Urban Nest Heights", "unh_possession_guidelines.docx"),
    ],
)
def test_retrieves_the_right_document(retriever, question, expected_source):
    sources = {chunk.source for chunk in retriever.retrieve(question)}
    assert expected_source in sources


def test_a_bare_rera_number_finds_its_project(retriever):
    """Embeddings are weak on identifiers; the exact-match path must carry these."""
    assert retriever.identifier_hit("P52100034899")

    chunks = retriever.retrieve("P52100034899")
    assert chunks
    assert all(chunk.project == "Meridian Lakeview Villas" for chunk in chunks)


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of France?",
        "Write me a Python function that reverses a string",
        "How do I bake sourdough bread?",
    ],
)
def test_gate_refuses_when_nothing_is_semantically_close(retriever, monkeypatch, question):
    """With no semantic match and no identifier, retrieval returns nothing at all.

    `_dense` is stubbed to empty because the production floor is calibrated for a
    sentence transformer, not for the bag-of-words stand-in used in tests.
    """
    monkeypatch.setattr(retriever, "_dense", lambda _query: [])
    assert retriever.retrieve(question) == []


def test_gate_still_opens_for_an_identifier_without_a_semantic_match(retriever, monkeypatch):
    monkeypatch.setattr(retriever, "_dense", lambda _query: [])
    assert retriever.retrieve("P52100034899")


def test_results_are_capped_deduplicated_and_ranked(retriever):
    chunks = retriever.retrieve("possession timeline for Urban Nest Heights")

    assert 0 < len(chunks) <= 5
    assert [chunk.rank for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert chunks == sorted(chunks, key=lambda chunk: chunk.score, reverse=True)

    keys = [(chunk.source, chunk.document.metadata["chunk_id"]) for chunk in chunks]
    assert len(keys) == len(set(keys)), "the same chunk was returned twice"
