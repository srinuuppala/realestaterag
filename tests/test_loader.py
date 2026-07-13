"""The corpus must load completely, and every chunk must know where it came from."""

from src.loader import SUPPORTED_EXTENSIONS, classify, discover, load_documents


def test_every_supported_file_is_discovered():
    files = discover()
    assert len(files) >= 90
    assert {path.suffix.lower() for path in files} <= SUPPORTED_EXTENSIONS
    assert {"pdf", "docx", "html", "md"} <= {path.suffix.lstrip(".") for path in files}


def test_all_four_formats_yield_text():
    by_type: dict[str, int] = {}
    for document in load_documents():
        by_type[document.metadata["file_type"]] = by_type.get(document.metadata["file_type"], 0) + 1
        assert len(document.page_content) > 40

    for file_type in ("pdf", "docx", "html", "md"):
        assert by_type.get(file_type, 0) > 0, f"no text extracted from any {file_type}"


def test_filenames_classify_into_project_builder_category():
    assert classify("mlv_payment_plan.pdf") == {
        "project": "Meridian Lakeview Villas",
        "builder": "Meridian Greens Realty",
        "category": "Payment Plan",
    }
    assert classify("urbannest_cancellation_refund_policy.docx")["category"] == "Cancellation & Refund Policy"
    assert classify("skyline_faq.html")["builder"] == "Skyline Horizon Developers"
    # Longest rule wins: `terms_of_use` must not be swallowed by the `home` rule.
    assert classify("propertybazaar_terms_of_use.html")["category"] == "Terms & Conditions"


def test_chunks_carry_metadata(chunks):
    assert len(chunks) > 100
    for chunk in chunks:
        assert chunk.metadata["source"]
        assert chunk.metadata["category"]
        assert "chunk_id" in chunk.metadata
