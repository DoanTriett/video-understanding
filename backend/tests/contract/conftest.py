"""Shared fixtures for FastAPI contract tests — mock DB, storage, Redis, Celery."""

from contextlib import ExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.summary import get_db as summary_get_db
from app.api.videos import get_db as videos_get_db
from app.main import app


def _mock_db():
    yield MagicMock()


def _done_video(**overrides):
    defaults = {
        "filename": "lecture.mp4",
        "created_at": datetime.utcnow(),
        "video_type": "lecture",
        "status": "done",
        "progress": 100,
        "error": None,
        "object_key": "vid-123/source.mp4",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def client_and_mocks():
    """Full app TestClient with videos router dependencies mocked."""
    app.dependency_overrides[videos_get_db] = _mock_db
    app.dependency_overrides[summary_get_db] = _mock_db

    with ExitStack() as stack:
        mocks = {
            "upload_bytes": stack.enter_context(patch("app.api.videos.upload_bytes")),
            "download_to_path": stack.enter_context(patch("app.api.videos.download_to_path")),
            "presigned_url": stack.enter_context(
                patch(
                    "app.api.videos.presigned_url", return_value="https://minio.example/presigned"
                )
            ),
            "set_progress": stack.enter_context(patch("app.api.videos.set_progress")),
            "get_progress": stack.enter_context(patch("app.api.videos.get_progress")),
            "celery_app": stack.enter_context(patch("app.api.videos.celery_app")),
            "crud": stack.enter_context(patch("app.api.videos.crud")),
        }
        with TestClient(app) as client:
            yield client, mocks

    app.dependency_overrides.clear()


@pytest.fixture
def client_and_summary_mocks():
    """Full app TestClient with summary + videos DB mocks."""
    app.dependency_overrides[videos_get_db] = _mock_db
    app.dependency_overrides[summary_get_db] = _mock_db

    with ExitStack() as stack:
        mocks = {
            "generate_summary": stack.enter_context(patch("app.api.summary.generate_summary")),
            "crud": stack.enter_context(patch("app.api.summary.crud")),
        }
        with TestClient(app) as client:
            yield client, mocks

    app.dependency_overrides.clear()
