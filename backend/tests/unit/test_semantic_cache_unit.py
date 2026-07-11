"""Unit tests for app.semantic_cache — mock Redis + embed_text, no bge model."""

import json
from unittest.mock import patch

import pytest

from app.semantic_cache import (
    CACHE_TTL,
    SIMILARITY_THRESHOLD,
    get_cached_answer,
    invalidate_cache,
    set_cached_answer,
)

VIDEO_ID = "unit-cache-vid"
QUESTION = "What topics were discussed?"
ANSWER = {"answer": "Budget and roadmap.", "citations": [{"chunk_id": "c1", "text": "x"}]}

HIT_VECTOR = [1.0, 0.0, 0.0]
MISS_VECTOR = [0.0, 1.0, 0.0]


class FakeRedis:
    """Minimal Redis stand-in for semantic cache keys."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        assert ttl == CACHE_TTL
        self.kv[key] = value

    def sadd(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).add(member)

    def srem(self, key: str, member: str) -> None:
        self.sets.get(key, set()).discard(member)

    def expire(self, key: str, ttl: int) -> None:
        assert ttl == CACHE_TTL

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.kv:
                del self.kv[key]
                deleted += 1
            if key in self.sets:
                del self.sets[key]
                deleted += 1
        return deleted


@pytest.fixture
def fake_redis():
    store = FakeRedis()
    with patch("app.semantic_cache.redis_client", store):
        yield store


def test_get_cached_answer_miss_empty_index(fake_redis):
    with patch("app.semantic_cache.embed_text", return_value=HIT_VECTOR):
        assert get_cached_answer(VIDEO_ID, QUESTION) is None


def test_get_cached_answer_hit_high_similarity(fake_redis):
    with patch("app.semantic_cache.embed_text", return_value=HIT_VECTOR):
        set_cached_answer(VIDEO_ID, QUESTION, ANSWER)

        result = get_cached_answer(VIDEO_ID, QUESTION)

    assert result is not None
    assert result["answer"] == ANSWER["answer"]
    assert result["citations"] == ANSWER["citations"]


def test_get_cached_answer_miss_low_similarity(fake_redis):
    with patch("app.semantic_cache.embed_text", return_value=HIT_VECTOR):
        set_cached_answer(VIDEO_ID, QUESTION, ANSWER)

    with patch("app.semantic_cache.embed_text", return_value=MISS_VECTOR):
        assert get_cached_answer(VIDEO_ID, "Unrelated question about cake") is None


def test_get_cached_answer_removes_stale_index_entry(fake_redis):
    idx_key = f"sc:idx:{VIDEO_ID}"
    stale_key = f"sc:{VIDEO_ID}:stale-id"
    fake_redis.sets[idx_key] = {stale_key}

    with patch("app.semantic_cache.embed_text", return_value=HIT_VECTOR):
        assert get_cached_answer(VIDEO_ID, QUESTION) is None

    assert stale_key not in fake_redis.sets.get(idx_key, set())


def test_set_cached_answer_stores_entry_and_index(fake_redis):
    with patch("app.semantic_cache.embed_text", return_value=HIT_VECTOR):
        set_cached_answer(VIDEO_ID, QUESTION, ANSWER)

    idx_key = f"sc:idx:{VIDEO_ID}"
    assert len(fake_redis.sets[idx_key]) == 1
    entry_key = next(iter(fake_redis.sets[idx_key]))
    assert entry_key.startswith(f"sc:{VIDEO_ID}:")
    payload = json.loads(fake_redis.kv[entry_key])
    assert payload["question"] == QUESTION
    assert payload["embedding"] == HIT_VECTOR
    assert payload["answer"] == ANSWER["answer"]
    assert payload["citations"] == ANSWER["citations"]
    assert "created_at" in payload


def test_invalidate_cache_deletes_entries_and_index(fake_redis):
    with patch("app.semantic_cache.embed_text", return_value=HIT_VECTOR):
        set_cached_answer(VIDEO_ID, QUESTION, ANSWER)
        set_cached_answer(VIDEO_ID, "Second question?", ANSWER)

    idx_key = f"sc:idx:{VIDEO_ID}"
    assert len(fake_redis.sets[idx_key]) == 2

    deleted = invalidate_cache(VIDEO_ID)

    assert deleted == 2
    assert idx_key not in fake_redis.sets
    with patch("app.semantic_cache.embed_text", return_value=HIT_VECTOR):
        assert get_cached_answer(VIDEO_ID, QUESTION) is None


def test_invalidate_cache_empty_returns_zero(fake_redis):
    assert invalidate_cache(VIDEO_ID) == 0


def test_similarity_threshold_constant():
    assert SIMILARITY_THRESHOLD == 0.92
