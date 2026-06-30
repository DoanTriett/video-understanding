"""Semantic cache for /ask responses.

Design:
  - Backend: plain Redis 7 (redis:7-alpine, no Redis Stack, no RediSearch module).
  - Per-entry storage: `sc:{video_id}:{uuid}` → JSON blob containing
    {question, embedding, answer, citations, created_at}.
  - Per-video index: `sc:idx:{video_id}` (Redis SET) → set of active entry keys
    for that video. Enables O(entries) linear scan without SCAN on the whole keyspace.
  - Similarity: dot product of unit-normalized vectors (embed_text normalises by
    default), which equals cosine similarity.
  - Threshold: 0.92 (from project plan — do not change here).

Key namespacing (never overlaps with existing keys):
  progress:{video_id}          ← existing, owned by store.py
  sc:{video_id}:{entry_uuid}   ← this module, per-entry payload
  sc:idx:{video_id}            ← this module, per-video index SET

TTL choice — 7 days:
  Cache entries are valid as long as the video hasn't been re-processed.
  Re-processing triggers an explicit invalidate_cache() call (wired in Prompt 7),
  which handles the correctness concern.  A 7-day TTL auto-cleans entries for
  videos that are uploaded and never re-visited, preventing unbounded growth.
  The 7-day window is long enough that active users will always get cache hits.
"""

import json
import uuid
from datetime import datetime, timezone

from app.pipeline.shared.text_embedder import embed_text
from app.store import redis_client

SIMILARITY_THRESHOLD: float = 0.92
CACHE_TTL: int = 7 * 24 * 60 * 60  # 7 days in seconds


# ── Helpers ────────────────────────────────────────────────────────────────────


def _entry_key(video_id: str, entry_id: str) -> str:
    return f"sc:{video_id}:{entry_id}"


def _index_key(video_id: str) -> str:
    return f"sc:idx:{video_id}"


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors.

    Since embed_text() returns unit-normalised vectors, this equals cosine
    similarity without needing to divide by magnitudes.
    """
    return sum(x * y for x, y in zip(a, b))


# ── Public API ─────────────────────────────────────────────────────────────────


def get_cached_answer(video_id: str, question: str) -> dict | None:
    """Return a cached answer if a semantically similar question exists.

    Embeds *question*, then computes cosine similarity against every cached
    entry for *video_id*.  Returns the best-matching entry's
    ``{"answer": str, "citations": list}`` dict when similarity >= 0.92,
    or ``None`` if no match is found (caller runs full retrieval + LLM).

    Stale index members (entry expired from Redis but still in the SET) are
    silently removed during the scan.
    """
    q_emb: list[float] = embed_text(question)

    idx_key = _index_key(video_id)
    entry_keys: set[str] = redis_client.smembers(idx_key)

    if not entry_keys:
        return None

    best_sim: float = 0.0
    best_entry: dict | None = None

    for ek in entry_keys:
        raw = redis_client.get(ek)
        if raw is None:
            # Entry expired — remove from the index to keep it clean.
            redis_client.srem(idx_key, ek)
            continue
        entry: dict = json.loads(raw)
        sim = _dot(q_emb, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_entry = entry

    if best_sim >= SIMILARITY_THRESHOLD and best_entry is not None:
        return {"answer": best_entry["answer"], "citations": best_entry["citations"]}

    return None


def set_cached_answer(
    video_id: str,
    question: str,
    answer_response: dict,
) -> None:
    """Store a completed /ask response in the semantic cache.

    *answer_response* must be a dict with at least ``answer`` (str) and
    ``citations`` (list) keys — the same shape as AskResponse.
    Only call this after a successful, non-error LLM response.

    Each entry gets its own UUID key so multiple questions for the same
    video coexist independently.
    """
    q_emb: list[float] = embed_text(question)
    entry_id = str(uuid.uuid4())
    ek = _entry_key(video_id, entry_id)

    payload = json.dumps(
        {
            "question": question,
            "embedding": q_emb,
            "answer": answer_response["answer"],
            "citations": answer_response["citations"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Store entry with TTL.
    redis_client.setex(ek, CACHE_TTL, payload)

    # Register the entry key in the per-video index SET and refresh the SET's TTL.
    idx_key = _index_key(video_id)
    redis_client.sadd(idx_key, ek)
    redis_client.expire(idx_key, CACHE_TTL)


def invalidate_cache(video_id: str) -> int:
    """Delete all cached entries for *video_id*.

    Call this when a video is re-processed so subsequent /ask calls get fresh
    answers instead of stale cached ones.

    Returns the number of entry keys that were deleted.
    """
    idx_key = _index_key(video_id)
    entry_keys: set[str] = redis_client.smembers(idx_key)

    deleted = 0
    if entry_keys:
        deleted = redis_client.delete(*entry_keys)

    redis_client.delete(idx_key)
    return deleted
