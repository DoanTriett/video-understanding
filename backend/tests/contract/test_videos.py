"""Contract tests for /videos/* endpoints (upload, status, transcript, chunks, url)."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from botocore.exceptions import ClientError

from tests.contract.conftest import _done_video


def test_upload_happy_path(client_and_mocks):
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

    mocks["upload_bytes"].assert_called_once()
    mocks["set_progress"].assert_called_once()
    mocks["celery_app"].send_task.assert_called_once_with(
        "workers.tasks.process_video", args=[data["video_id"], "lecture"]
    )
    mocks["crud"].create_video.assert_called_once()
    mocks["crud"].set_video_object_key.assert_called_once()


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
    assert "not allowed" in resp.json()["detail"].lower()
    mocks["upload_bytes"].assert_not_called()


def test_upload_rejects_oversized_file(client_and_mocks):
    client, mocks = client_and_mocks

    with patch("app.api.videos.settings.max_file_size_mb", 0.001):
        resp = client.post(
            "/videos/upload",
            files={"file": ("big.mp4", b"x" * 2048, "video/mp4")},
            data={"video_type": "meeting"},
        )

    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()
    mocks["upload_bytes"].assert_not_called()


def test_status_in_progress_reads_redis(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["get_progress"].return_value = {"stage": "transcribing", "pct": 25}
    mocks["crud"].get_video.return_value = _done_video(status="processing", progress=25)

    resp = client.get("/videos/abc-123/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "processing"
    assert data["progress_percent"] == 25
    assert data["video_type"] == "lecture"


def test_status_done_falls_back_to_postgres(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["get_progress"].return_value = None
    mocks["crud"].get_video.return_value = _done_video(status="done", progress=100)

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


def test_status_redis_progress_but_video_missing_returns_404(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["get_progress"].return_value = {"stage": "transcribing", "pct": 10}
    mocks["crud"].get_video.return_value = None

    resp = client.get("/videos/orphan/status")

    assert resp.status_code == 404


def test_transcript_happy_path(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = _done_video()

    transcript_payload = {"segments": [{"text": "hello", "start": 0.0, "end": 1.0}]}

    def _write_transcript(_object_key, local_path):
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(transcript_payload, f)

    mocks["download_to_path"].side_effect = _write_transcript

    resp = client.get("/videos/vid-123/transcript")

    assert resp.status_code == 200
    assert resp.json() == transcript_payload
    mocks["download_to_path"].assert_called_once()
    object_key, local_path = mocks["download_to_path"].call_args[0]
    assert object_key == "vid-123/transcript.json"
    assert local_path.endswith("transcript.json")


def test_transcript_video_not_found_returns_404(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = None

    resp = client.get("/videos/missing/transcript")

    assert resp.status_code == 404
    mocks["download_to_path"].assert_not_called()


def test_transcript_not_ready_returns_400(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = _done_video(status="processing")

    resp = client.get("/videos/vid-123/transcript")

    assert resp.status_code == 400
    assert "not ready" in resp.json()["detail"].lower()
    mocks["download_to_path"].assert_not_called()


def test_transcript_file_missing_in_storage_returns_404(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = _done_video()
    error = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "Not found"}},
        "GetObject",
    )
    mocks["download_to_path"].side_effect = error

    resp = client.get("/videos/vid-123/transcript")

    assert resp.status_code == 404
    assert "transcript" in resp.json()["detail"].lower()


def test_chunks_happy_path(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = _done_video()
    mocks["crud"].get_chunks.return_value = [
        SimpleNamespace(
            id="vid-123_chunk_0000",
            speaker="SPEAKER_00",
            text="Hello world",
            start=0.0,
            end=5.0,
            chunk_type="speech",
        )
    ]

    resp = client.get("/videos/vid-123/chunks")

    assert resp.status_code == 200
    data = resp.json()
    assert data["video_id"] == "vid-123"
    assert data["num_chunks"] == 1
    assert data["chunks"][0]["text"] == "Hello world"
    assert data["chunks"][0]["speaker"] == "SPEAKER_00"


def test_chunks_video_not_found_returns_404(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = None

    resp = client.get("/videos/missing/chunks")

    assert resp.status_code == 404


def test_chunks_not_ready_returns_400(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = _done_video(status="processing")

    resp = client.get("/videos/vid-123/chunks")

    assert resp.status_code == 400
    assert "not ready" in resp.json()["detail"].lower()


def test_video_url_happy_path(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = _done_video(object_key="vid-123/source.mp4")

    resp = client.get("/videos/vid-123/url")

    assert resp.status_code == 200
    assert resp.json() == {"url": "https://minio.example/presigned"}
    mocks["presigned_url"].assert_called_once_with("vid-123/source.mp4")


def test_video_url_not_found_returns_404(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = None

    resp = client.get("/videos/missing/url")

    assert resp.status_code == 404
    mocks["presigned_url"].assert_not_called()


def test_video_url_missing_object_key_returns_404(client_and_mocks):
    client, mocks = client_and_mocks
    mocks["crud"].get_video.return_value = _done_video(object_key=None)

    resp = client.get("/videos/vid-123/url")

    assert resp.status_code == 404
    mocks["presigned_url"].assert_not_called()
