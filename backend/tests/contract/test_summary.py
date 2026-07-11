"""Contract tests for GET /summary and POST /summary/regenerate."""

from datetime import datetime
from types import SimpleNamespace

from tests.contract.conftest import _done_video

_MEETING_CONTENT = {
    "agenda": ["Budget review"],
    "decisions": ["Approve Q3 budget"],
    "action_items": ["Alice to send report"],
    "participants": ["Speaker A"],
}


def _summary_row(**overrides):
    defaults = {
        "video_id": "vid-123",
        "video_type": "meeting",
        "created_at": datetime.utcnow(),
        "content": _MEETING_CONTENT,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_get_summary_returns_cached_row(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = _done_video(status="done", video_type="meeting")
    mocks["crud"].get_summary.return_value = _summary_row()

    resp = client.get("/videos/vid-123/summary")

    assert resp.status_code == 200
    data = resp.json()
    assert data["video_id"] == "vid-123"
    assert data["video_type"] == "meeting"
    assert data["content"]["agenda"] == ["Budget review"]
    mocks["generate_summary"].assert_not_called()


def test_get_summary_lazy_generates_when_missing(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    row = _summary_row()
    mocks["crud"].get_video.return_value = _done_video(status="done", video_type="meeting")
    mocks["crud"].get_summary.side_effect = [None, row]

    resp = client.get("/videos/vid-123/summary")

    assert resp.status_code == 200
    assert resp.json()["content"]["decisions"] == ["Approve Q3 budget"]
    mocks["generate_summary"].assert_called_once_with("vid-123")


def test_get_summary_video_not_found_returns_404(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = None

    resp = client.get("/videos/missing/summary")

    assert resp.status_code == 404
    mocks["generate_summary"].assert_not_called()


def test_get_summary_not_ready_returns_400(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = _done_video(status="processing")

    resp = client.get("/videos/vid-123/summary")

    assert resp.status_code == 400
    assert "not ready" in resp.json()["detail"].lower()
    mocks["generate_summary"].assert_not_called()


def test_get_summary_generation_unavailable_returns_503(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = _done_video(status="done", video_type="meeting")
    mocks["crud"].get_summary.return_value = None
    mocks["generate_summary"].side_effect = RuntimeError("Ollama down")

    resp = client.get("/videos/vid-123/summary")

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


def test_get_summary_invalid_video_type_returns_400(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = _done_video(status="done", video_type="meeting")
    mocks["crud"].get_summary.return_value = None
    mocks["generate_summary"].side_effect = ValueError("Unsupported video_type: unknown")

    resp = client.get("/videos/vid-123/summary")

    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


def test_regenerate_summary_happy_path(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = _done_video(status="done", video_type="meeting")
    mocks["crud"].get_summary.return_value = _summary_row()

    resp = client.post("/videos/vid-123/summary/regenerate")

    assert resp.status_code == 200
    assert resp.json()["video_type"] == "meeting"
    mocks["generate_summary"].assert_called_once_with("vid-123")


def test_regenerate_summary_video_not_found_returns_404(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = None

    resp = client.post("/videos/missing/summary/regenerate")

    assert resp.status_code == 404
    mocks["generate_summary"].assert_not_called()


def test_regenerate_summary_not_ready_returns_400(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = _done_video(status="processing")

    resp = client.post("/videos/vid-123/summary/regenerate")

    assert resp.status_code == 400
    mocks["generate_summary"].assert_not_called()


def test_regenerate_summary_generation_unavailable_returns_503(client_and_summary_mocks):
    client, mocks = client_and_summary_mocks
    mocks["crud"].get_video.return_value = _done_video(status="done", video_type="meeting")
    mocks["generate_summary"].side_effect = RuntimeError("Ollama down")

    resp = client.post("/videos/vid-123/summary/regenerate")

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()
