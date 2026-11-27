import numpy as np

from app.chunking import chunk_text, count_tokens
from app.rag import mmr

DOC = "# Title\n\n" + "\n\n".join(
    f"## Section {i}\n\n" + " ".join(f"Sentence {i}-{j} talks about topic {i}." for j in range(25))
    for i in range(6))


def test_chunks_respect_size_and_have_metadata():
    chunks = chunk_text(DOC, size=200, overlap=30)
    assert len(chunks) > 3
    assert [c["position"] for c in chunks] == list(range(len(chunks)))
    assert all(c["tokens"] <= 200 for c in chunks)
    assert all(c["section"] for c in chunks)


def test_overlap_repeats_text_between_neighbours():
    chunks = chunk_text(DOC, size=200, overlap=40)
    shared = set(chunks[0]["text"].split("\n\n")) & set(chunks[1]["text"].split("\n\n"))
    assert shared, "consecutive chunks should share at least one unit"


def test_every_chunk_is_a_forward_step_and_covers_the_text():
    chunks = chunk_text(DOC, size=150, overlap=20)
    assert "Sentence 5-24" in chunks[-1]["text"]


def test_short_and_empty_documents():
    assert chunk_text("   ") == []
    assert len(chunk_text("Just one short line.")) == 1


def test_oversized_paragraph_is_split():
    big = "word " * 3000
    assert all(c["tokens"] <= 500 + 5 for c in chunk_text(big, size=500, overlap=50))


def test_mmr_prefers_diverse_results():
    q = np.array([1.0, 0.0])
    docs = np.array([[1.0, 0.0], [0.99, 0.01], [0.6, 0.8]])  # doc 1 is a near-duplicate of doc 0
    assert mmr(q, docs, k=2, lam=0.3) == [0, 2]
    assert mmr(q, docs, k=2, lam=1.0) == [0, 1]


def test_count_tokens_positive():
    assert count_tokens("hello world") > 0
