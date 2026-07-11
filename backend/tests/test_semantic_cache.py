"""Tests for app/semantic_cache.py

All tests require a live Redis instance and the bge-small-en-v1.5 model,
so they are marked @pytest.mark.integration.

Run: pytest tests/test_semantic_cache.py -m integration -s -v
"""

import sys

sys.path.insert(0, ".")

import pytest

from app.pipeline.shared.text_embedder import embed_text
from app.semantic_cache import (
    SIMILARITY_THRESHOLD,
    _dot,
    get_cached_answer,
    invalidate_cache,
    set_cached_answer,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

TEST_VIDEO_ID = "test-video-cache-unit"
_FAKE_ANSWER = {
    "answer": "The meeting covered Q3 budget and team allocation.",
    "citations": [
        {
            "chunk_id": "chunk_001",
            "speaker": "Alice",
            "start": 10.0,
            "end": 20.0,
            "text": "Q3 budget...",
        }
    ],
}


@pytest.fixture(autouse=True)
def clean_cache():
    """Wipe the test video's cache before and after every test."""
    invalidate_cache(TEST_VIDEO_ID)
    yield
    invalidate_cache(TEST_VIDEO_ID)


# ── Unit-level: _dot (no Redis, no model) ─────────────────────────────────────


def test_dot_identical_unit_vectors():
    v = [1.0, 0.0, 0.0]
    assert _dot(v, v) == pytest.approx(1.0)


def test_dot_orthogonal_vectors():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _dot(a, b) == pytest.approx(0.0)


# ── Integration tests ─────────────────────────────────────────────────────────

QUESTION_ORIGINAL = "What topics were discussed in the meeting?"


@pytest.mark.integration
def test_cache_hit_exact_same_question():
    """Exact same question after set → similarity ≈ 1.0 → cache hit."""
    set_cached_answer(TEST_VIDEO_ID, QUESTION_ORIGINAL, _FAKE_ANSWER)

    result = get_cached_answer(TEST_VIDEO_ID, QUESTION_ORIGINAL)

    assert result is not None, "Expected cache hit for exact same question"
    assert result["answer"] == _FAKE_ANSWER["answer"]
    assert result["citations"] == _FAKE_ANSWER["citations"]

    # Confirm similarity is effectively 1.0 (same vector dotted with itself)
    emb = embed_text(QUESTION_ORIGINAL)
    sim = _dot(emb, emb)
    print(f"\n[exact match] similarity = {sim:.6f}  (expected ~1.0)")
    assert sim >= 0.9999


@pytest.mark.integration
def test_cache_miss_unrelated_question():
    """Completely unrelated question → low similarity → None."""
    set_cached_answer(TEST_VIDEO_ID, QUESTION_ORIGINAL, _FAKE_ANSWER)

    unrelated = "What is the recipe for chocolate cake?"
    result = get_cached_answer(TEST_VIDEO_ID, unrelated)

    emb_orig = embed_text(QUESTION_ORIGINAL)
    emb_unrel = embed_text(unrelated)
    sim = _dot(emb_orig, emb_unrel)
    print(f"\n[unrelated] similarity = {sim:.6f}  (threshold = {SIMILARITY_THRESHOLD})")

    assert (
        result is None
    ), f"Expected cache MISS for unrelated question, but got a hit (sim={sim:.4f})"
    assert sim < SIMILARITY_THRESHOLD


@pytest.mark.integration
def test_cache_paraphrase_report_similarity():
    """Paraphrase of the cached question — report actual similarity without assuming pass/fail.

    bge-small-en-v1.5 (384-dim) may or may not reach 0.92 for short paraphrases.
    We report the real number and only assert it's above a loose lower bound (0.80),
    meaning the embedder does recognise the semantic relationship.
    """
    set_cached_answer(TEST_VIDEO_ID, QUESTION_ORIGINAL, _FAKE_ANSWER)

    paraphrase = "Which subjects came up during the meeting?"
    emb_orig = embed_text(QUESTION_ORIGINAL)
    emb_para = embed_text(paraphrase)
    sim = _dot(emb_orig, emb_para)

    result = get_cached_answer(TEST_VIDEO_ID, paraphrase)
    cache_hit = result is not None

    print("\n[paraphrase]")
    print(f"  original  : {QUESTION_ORIGINAL!r}")
    print(f"  paraphrase: {paraphrase!r}")
    print(f"  similarity: {sim:.6f}  (threshold = {SIMILARITY_THRESHOLD})")
    print(f"  cache hit : {cache_hit}")

    # The embedder must recognise some semantic similarity (loose bound).
    assert sim >= 0.80, f"Similarity {sim:.4f} unexpectedly low — embedder may not be loaded"
    # No assertion on >= 0.92; we just report whether it crossed the threshold.


@pytest.mark.integration
def test_cache_miss_empty_cache():
    """get on a video with no entries returns None without error."""
    result = get_cached_answer(TEST_VIDEO_ID, QUESTION_ORIGINAL)
    assert result is None


@pytest.mark.integration
def test_invalidate_clears_all_entries():
    """After invalidate, all entries for the video are gone."""
    questions = [
        "What topics were discussed?",
        "Who attended the meeting?",
        "What were the action items?",
    ]
    for q in questions:
        set_cached_answer(TEST_VIDEO_ID, q, _FAKE_ANSWER)

    # Confirm at least the exact question is in cache before invalidation.
    assert get_cached_answer(TEST_VIDEO_ID, questions[0]) is not None

    deleted = invalidate_cache(TEST_VIDEO_ID)
    print(f"\n[invalidate] deleted {deleted} entry keys + index key")

    # All entries must be gone after invalidation.
    for q in questions:
        result = get_cached_answer(TEST_VIDEO_ID, q)
        assert result is None, f"Expected None after invalidate for: {q!r}"


@pytest.mark.integration
def test_set_multiple_entries_returns_best_match():
    """Two entries cached — get returns the one with higher similarity."""
    q1 = "What topics were discussed in the meeting?"
    q2 = "What is the capital of France?"

    set_cached_answer(TEST_VIDEO_ID, q1, _FAKE_ANSWER)
    fake2 = {"answer": "Paris.", "citations": []}
    set_cached_answer(TEST_VIDEO_ID, q2, fake2)

    # Ask something close to q1
    result = get_cached_answer(TEST_VIDEO_ID, "What subjects came up in the meeting?")
    emb_q1 = embed_text(q1)
    emb_q2 = embed_text(q2)
    emb_ask = embed_text("What subjects came up in the meeting?")
    sim1 = _dot(emb_ask, emb_q1)
    sim2 = _dot(emb_ask, emb_q2)

    print(f"\n[best match] sim to q1={sim1:.4f}, sim to q2={sim2:.4f}")
    # If we get a hit, it should be q1's answer (higher similarity)
    if result is not None:
        assert (
            result["answer"] == _FAKE_ANSWER["answer"]
        ), "Cache returned wrong entry — should have matched q1"
