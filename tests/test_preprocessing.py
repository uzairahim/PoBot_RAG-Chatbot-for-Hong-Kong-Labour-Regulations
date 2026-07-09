"""Unit tests for text cleaning and chunking — pure functions, no models."""
from src.preprocessing import chunk_text, clean_text


def test_clean_text_dehyphenates_linebreaks():
    assert clean_text("employ-\nment ordinance") == "employment ordinance"


def test_clean_text_drops_page_number_lines():
    cleaned = clean_text("Real content here.\n12\nMore content.")
    assert "Real content here." in cleaned
    assert "More content." in cleaned
    assert "\n12\n" not in cleaned


def test_clean_text_collapses_whitespace():
    assert clean_text("a    b\t\tc") == "a b c"
    assert "\n\n\n" not in clean_text("x\n\n\n\n\ny")


def test_chunk_text_respects_max_size():
    text = " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # Normal (non-hard-split) chunks should not exceed chunk_size by much.
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_text_has_overlap():
    # With overlap > 0 and multiple chunks, at least one sentence must appear
    # in two different chunks (the carried-over tail).
    text = " ".join(f"Unique sentence {i} here." for i in range(80))
    chunks = chunk_text(text, chunk_size=200, overlap=60)
    assert len(chunks) >= 2
    shared = any(
        f"Unique sentence {i} here." in chunks[j] and f"Unique sentence {i} here." in chunks[j + 1]
        for i in range(80) for j in range(len(chunks) - 1)
    )
    assert shared, "expected overlapping sentences between consecutive chunks"


def test_chunk_text_hard_splits_overlong_sentence():
    long_sentence = "x" * 1000  # no sentence boundaries
    chunks = chunk_text(long_sentence, chunk_size=300, overlap=50)
    assert len(chunks) >= 4
    assert all(len(c) <= 300 for c in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("", chunk_size=300, overlap=50) == []
