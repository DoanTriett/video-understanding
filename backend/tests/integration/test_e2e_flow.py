"""Integration test: upload → mock pipeline → Postgres + Qdrant → POST /ask."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.db import crud
from app.db.session import SessionLocal
from app.main import app
from app.pipeline.meeting.chunker import MeetingChunk
from app.pipeline.shared.indexer import index_chunks
from tests.integration.conftest import teardown_video

CHUNK_TEXT = "We discussed the Q3 budget roadmap."
MOCK_EMBED_VECTOR = [1.0] + [0.0] * 383  # 384-dim unit vector for mocked bge


def _mock_embed_texts(texts: list[str]) -> list[list[float]]:
    return [MOCK_EMBED_VECTOR for _ in texts]


def _mock_embed_text(_text: str) -> list[float]:
    return MOCK_EMBED_VECTOR


def _prebuilt_meeting_chunks(video_id: str) -> list[MeetingChunk]:
    return [
        MeetingChunk(
            chunk_id=f"{video_id}_chunk_0000",
            video_id=video_id,
            speaker="SPEAKER_00",
            text=CHUNK_TEXT,
            start=10.0,
            end=25.0,
            turn_index=0,
            chunk_type="speech",
        )
    ]


def _simulate_pipeline(video_id: str, _video_type: str) -> None:
    """Inject pre-built pipeline output — same steps as tasks.py tail, no ML."""
    db = SessionLocal()
    try:
        chunks = _prebuilt_meeting_chunks(video_id)
        crud.save_chunks(db, video_id, chunks)
        with patch("app.pipeline.shared.indexer.embed_texts", side_effect=_mock_embed_texts):
            index_chunks(video_id, chunks)
        crud.update_video_status(db, video_id, status="done", progress=100)
    finally:
        db.close()


def _run_upload_ask_flow(video_ids: list[str]) -> str:
    """Execute one full upload → pipeline mock → ask cycle. Returns video_id."""
    with (
        patch("app.api.videos.upload_bytes") as mock_upload,
        patch("app.api.videos.process_video") as mock_task,
        patch("app.api.qa.get_cached_answer", return_value=None),
        patch("app.api.qa.set_cached_answer"),
        patch("app.pipeline.shared.retriever.embed_text", side_effect=_mock_embed_text),
        patch(
            "app.api.qa.generate_answer",
            return_value="The team discussed the Q3 budget roadmap at [00:10].",
        ),
    ):
        mock_task.delay.side_effect = _simulate_pipeline

        with TestClient(app) as client:
            upload_resp = client.post(
                "/videos/upload",
                files={"file": ("meeting.mp4", b"fake-video-bytes", "video/mp4")},
                data={"video_type": "meeting"},
            )
            assert upload_resp.status_code == 200
            video_id = upload_resp.json()["video_id"]
            video_ids.append(video_id)

            mock_upload.assert_called_once()
            mock_task.delay.assert_called_once_with(video_id, "meeting")

            db = SessionLocal()
            try:
                video = crud.get_video(db, video_id)
                assert video is not None
                assert video.status == "done"
                chunks = crud.get_chunks(db, video_id)
                assert len(chunks) == 1
                assert chunks[0].text == CHUNK_TEXT
            finally:
                db.close()

            ask_resp = client.post(
                f"/videos/{video_id}/ask",
                json={"question": "What was discussed about the budget?"},
            )

    assert ask_resp.status_code == 200
    data = ask_resp.json()
    assert "Q3 budget" in data["answer"]
    assert len(data["citations"]) >= 1
    cited = data["citations"][0]
    assert cited["chunk_id"] == f"{video_id}_chunk_0000"
    assert cited["text"] == CHUNK_TEXT
    assert cited["speaker"] == "SPEAKER_00"
    assert cited["start"] == 10.0
    assert cited["end"] == 25.0
    return video_id


@pytest.mark.integration
def test_upload_pipeline_index_ask_e2e(clean_video_after_test):
    """Upload → mock Celery pipeline → real Postgres/Qdrant → /ask with citation."""
    _run_upload_ask_flow(clean_video_after_test)


@pytest.mark.integration
def test_e2e_teardown_allows_rerun(clean_video_after_test):
    """Run the flow twice with explicit teardown to confirm no ID conflicts."""
    video_ids: list[str] = []
    vid1 = _run_upload_ask_flow(video_ids)
    teardown_video(vid1)

    vid2 = _run_upload_ask_flow(video_ids)
    clean_video_after_test.extend(video_ids)
    assert vid1 != vid2
