import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.qa import get_db, router

# Minimal app — avoids importing videos.py → workers.tasks → heavy ML deps.
test_app = FastAPI()
test_app.include_router(router)


def _mock_db():
    yield MagicMock()


test_app.dependency_overrides[get_db] = _mock_db

client = TestClient(test_app)


def _video(status: str = "done") -> MagicMock:
    v = MagicMock()
    v.status = status
    return v


# ── test cases ────────────────────────────────────────────────────────────────


def test_video_not_found():
    with patch("app.db.crud.get_video", return_value=None):
        resp = client.post("/videos/nonexistent/ask", json={"question": "hello"})
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_video_not_done_returns_400():
    with patch("app.db.crud.get_video", return_value=_video("processing")):
        resp = client.post("/videos/vid123/ask", json={"question": "hello"})
    assert resp.status_code == 400
    assert "processing" in resp.json()["detail"].lower()


def test_empty_retrieve_skips_llm():
    with (
        patch("app.db.crud.get_video", return_value=_video("done")),
        patch("app.api.qa.get_cached_answer", return_value=None),
        patch("app.api.qa.set_cached_answer") as mock_set_cache,
        patch("app.api.qa.retrieve", return_value=[]),
        patch("app.api.qa.generate_answer") as mock_gen,
    ):
        resp = client.post("/videos/vid123/ask", json={"question": "anything"})

    assert resp.status_code == 200
    assert resp.headers.get("X-Cache") == "MISS"
    data = resp.json()
    assert data["citations"] == []
    assert "không tìm thấy" in data["answer"].lower()
    mock_gen.assert_not_called()
    mock_set_cache.assert_not_called()


def test_happy_path_returns_answer_and_citations():
    fake_chunks = [
        {
            "chunk_id": "vid123_chunk_0000",
            "speaker": "SPEAKER_00",
            "start": 10.0,
            "end": 25.0,
            "text": "We discussed the roadmap.",
            "chunk_type": "transcript",
            "score": 0.92,
        }
    ]
    with (
        patch("app.db.crud.get_video", return_value=_video("done")),
        patch("app.api.qa.get_cached_answer", return_value=None),
        patch("app.api.qa.set_cached_answer") as mock_set_cache,
        patch("app.api.qa.retrieve", return_value=fake_chunks),
        patch(
            "app.api.qa.build_context",
            return_value="[SPEAKER_00 @ 00:10] We discussed the roadmap.",
        ),
        patch(
            "app.api.qa.generate_answer",
            return_value="The team discussed the roadmap at [00:10].",
        ),
    ):
        resp = client.post(
            "/videos/vid123/ask",
            json={"question": "What was discussed?"},
        )

    assert resp.status_code == 200
    assert resp.headers.get("X-Cache") == "MISS"
    data = resp.json()
    assert data["answer"] == "The team discussed the roadmap at [00:10]."
    assert len(data["citations"]) == 1
    c = data["citations"][0]
    assert c["chunk_id"] == "vid123_chunk_0000"
    assert c["speaker"] == "SPEAKER_00"
    assert c["start"] == 10.0
    assert c["end"] == 25.0
    assert c["text"] == "We discussed the roadmap."
    mock_set_cache.assert_called_once()


def test_cache_hit_skips_retriever_and_llm():
    cached = {
        "answer": "Cached answer.",
        "citations": [
            {
                "chunk_id": "vid123_chunk_0000",
                "speaker": "SPEAKER_00",
                "start": 10.0,
                "end": 25.0,
                "text": "We discussed the roadmap.",
            }
        ],
    }
    with (
        patch("app.db.crud.get_video", return_value=_video("done")),
        patch("app.api.qa.get_cached_answer", return_value=cached),
        patch("app.api.qa.retrieve") as mock_retrieve,
        patch("app.api.qa.generate_answer") as mock_gen,
        patch("app.api.qa.set_cached_answer") as mock_set_cache,
    ):
        resp = client.post(
            "/videos/vid123/ask",
            json={"question": "What was discussed?"},
        )

    assert resp.status_code == 200
    assert resp.headers.get("X-Cache") == "HIT"
    assert resp.json()["answer"] == "Cached answer."
    mock_retrieve.assert_not_called()
    mock_gen.assert_not_called()
    mock_set_cache.assert_not_called()


def test_openai_down_returns_503():
    fake_chunks = [
        {
            "chunk_id": "vid123_chunk_0000",
            "speaker": None,
            "start": 5.0,
            "end": 15.0,
            "text": "Some content.",
            "chunk_type": "transcript",
            "score": 0.88,
        }
    ]
    with (
        patch("app.db.crud.get_video", return_value=_video("done")),
        patch("app.api.qa.get_cached_answer", return_value=None),
        patch("app.api.qa.set_cached_answer") as mock_set_cache,
        patch("app.api.qa.retrieve", return_value=fake_chunks),
        patch("app.api.qa.build_context", return_value="[00:05] Some content."),
        patch(
            "app.api.qa.generate_answer",
            side_effect=RuntimeError("OpenAI API request failed"),
        ),
    ):
        resp = client.post("/videos/vid123/ask", json={"question": "anything"})

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()
    mock_set_cache.assert_not_called()
