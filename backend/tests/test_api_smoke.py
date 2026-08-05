"""Smoke tests cho FastAPI — mock process_video.delay, DB, storage, Redis progress.

Kiểm tra luồng upload → status mà không cần Postgres/MinIO/Redis/Celery thật.
"""

from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.videos import get_db
from app.main import app


@pytest.fixture
def client_and_mocks():
    # get_db dependency → MagicMock session (không kết nối Postgres).
    app.dependency_overrides[get_db] = lambda: iter([MagicMock()])

    with ExitStack() as stack:
        mocks = {
            "upload_bytes": stack.enter_context(patch("app.api.videos.upload_bytes")),
            "set_progress": stack.enter_context(patch("app.api.videos.set_progress")),
            "get_progress": stack.enter_context(patch("app.api.videos.get_progress")),
            "celery_app": stack.enter_context(patch("app.api.videos.celery_app")),
            "crud": stack.enter_context(patch("app.api.videos.crud")),
        }
        with TestClient(app) as client:
            yield client, mocks

    app.dependency_overrides.clear()


def test_upload_returns_pending_and_enqueues_job(client_and_mocks):
    client, mocks = client_and_mocks

    resp = client.post(
        "/videos/upload",
        files={"file": ("lecture.mp4", b"x" * 1024, "video/mp4")},
        data={"video_type": "lecture"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "pending"
    assert data["filename"] == "lecture.mp4"
    assert "video_id" in data

    # File được đẩy lên MinIO và job gửi cho Celery đúng video_type.
    mocks["upload_bytes"].assert_called_once()
    mocks["celery_app"].send_task.assert_called_once_with(
        "workers.tasks.process_video", args=[data["video_id"], "lecture"]
    )
    mocks["crud"].create_video.assert_called_once()


def test_upload_missing_video_type_returns_422(client_and_mocks):
    client, mocks = client_and_mocks

    resp = client.post(
        "/videos/upload",
        files={"file": ("lecture.mp4", b"x" * 1024, "video/mp4")},
    )

    assert resp.status_code == 422
    mocks["celery_app"].send_task.assert_not_called()


def test_upload_rejects_bad_extension(client_and_mocks):
    client, mocks = client_and_mocks

    resp = client.post(
        "/videos/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"video_type": "lecture"},
    )

    assert resp.status_code == 400
    mocks["celery_app"].send_task.assert_not_called()


def test_upload_cors_allows_production_vercel_origin(client_and_mocks):
    client, _ = client_and_mocks

    resp = client.options(
        "/videos/upload",
        headers={
            "Origin": "https://video-understanding.vercel.app",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == ("https://video-understanding.vercel.app")


def test_unhandled_error_still_returns_cors_headers(client_and_mocks):
    """Regression: plain 500 without ACAO is reported by browsers as a CORS failure."""
    client, mocks = client_and_mocks
    mocks["crud"].get_video.side_effect = RuntimeError("simulated db outage")

    resp = client.get(
        "/videos/abc-123/status",
        headers={"Origin": "https://video-understanding.vercel.app"},
    )

    assert resp.status_code == 500
    assert resp.headers["access-control-allow-origin"] == ("https://video-understanding.vercel.app")
    assert "detail" in resp.json()


def test_status_in_progress_reads_redis(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["get_progress"].return_value = {"stage": "transcribing", "pct": 25}
    mocks["crud"].get_video.return_value = SimpleNamespace(
        filename="lecture.mp4",
        created_at=datetime.utcnow(),
        video_type="lecture",
        status="processing",
        progress=25,
        error=None,
    )

    resp = client.get("/videos/abc-123/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    assert data["progress_percent"] == 25
    assert data["video_type"] == "lecture"


def test_status_done_falls_back_to_postgres(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["get_progress"].return_value = None  # không còn key Redis → đã xong
    mocks["crud"].get_video.return_value = SimpleNamespace(
        filename="lecture.mp4",
        created_at=datetime.utcnow(),
        video_type="lecture",
        status="done",
        progress=100,
        error=None,
    )

    resp = client.get("/videos/abc-123/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["progress_percent"] == 100


def test_status_not_found_returns_404(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["get_progress"].return_value = None
    mocks["crud"].get_video.return_value = None

    resp = client.get("/videos/missing/status")

    assert resp.status_code == 404
