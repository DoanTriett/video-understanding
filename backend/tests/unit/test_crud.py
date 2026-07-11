"""Unit tests for app.db.crud — SQLAlchemy Session mocked, no real Postgres."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.db import crud
from app.db.models import Chunk, Summary, Video


def _video_query(db: MagicMock, first_return):
    db.query.return_value.filter.return_value.first.return_value = first_return


def _chunk_query(db: MagicMock, all_result: list):
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = all_result


def _summary_query(db: MagicMock, first_return):
    db.query.return_value.filter.return_value.first.return_value = first_return


def test_create_video_happy_path():
    db = MagicMock()

    video = crud.create_video(db, "vid-1", "meeting.mp4", "meeting")

    assert video.id == "vid-1"
    assert video.filename == "meeting.mp4"
    assert video.video_type == "meeting"
    assert video.status == "pending"
    assert video.progress == 0
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once_with(video)


def test_get_video_returns_first_match():
    db = MagicMock()
    expected = Video(id="vid-1", filename="a.mp4", video_type="meeting")
    _video_query(db, expected)

    assert crud.get_video(db, "vid-1") is expected


def test_get_video_not_found():
    db = MagicMock()
    _video_query(db, None)

    assert crud.get_video(db, "missing") is None


def test_update_video_status_updates_fields():
    db = MagicMock()
    video = SimpleNamespace(status="pending", progress=0, error=None)
    _video_query(db, video)

    crud.update_video_status(db, "vid-1", status="done", progress=100, error=None)

    assert video.status == "done"
    assert video.progress == 100
    db.commit.assert_called_once()


def test_update_video_status_raises_when_video_missing():
    db = MagicMock()
    _video_query(db, None)

    with pytest.raises(ValueError, match="not found"):
        crud.update_video_status(db, "missing", status="failed")


def test_set_video_object_key_when_video_exists():
    db = MagicMock()
    video = SimpleNamespace(object_key=None)
    _video_query(db, video)

    crud.set_video_object_key(db, "vid-1", "vid-1/source.mp4")

    assert video.object_key == "vid-1/source.mp4"
    db.commit.assert_called_once()


def test_set_video_object_key_noop_when_video_missing():
    db = MagicMock()
    _video_query(db, None)

    crud.set_video_object_key(db, "missing", "key")

    db.commit.assert_not_called()


def test_get_video_object_key():
    db = MagicMock()
    _video_query(db, SimpleNamespace(object_key="vid-1/source.mp4"))

    assert crud.get_video_object_key(db, "vid-1") == "vid-1/source.mp4"


def test_get_video_object_key_none_when_missing():
    db = MagicMock()
    _video_query(db, None)

    assert crud.get_video_object_key(db, "missing") is None


def test_save_chunks_empty_list_deletes_old_only():
    db = MagicMock()
    delete_mock = db.query.return_value.filter.return_value.delete

    crud.save_chunks(db, "vid-1", [])

    delete_mock.assert_called_once()
    db.add.assert_not_called()
    db.commit.assert_called_once()


def test_save_chunks_from_dicts():
    db = MagicMock()
    delete_mock = db.query.return_value.filter.return_value.delete
    chunks = [
        {
            "speaker": "SPEAKER_00",
            "text": "Hello",
            "start": 0.0,
            "end": 5.0,
            "chunk_type": "speech",
        }
    ]

    crud.save_chunks(db, "vid-1", chunks)

    delete_mock.assert_called_once()
    db.add.assert_called_once()
    added = db.add.call_args[0][0]
    assert isinstance(added, Chunk)
    assert added.id == "vid-1_chunk_0000"
    assert added.text == "Hello"
    db.commit.assert_called_once()


def test_save_chunks_unwraps_meeting_wrapper():
    db = MagicMock()
    inner = SimpleNamespace(
        speaker="SPEAKER_01",
        text="Wrapped turn",
        start=1.0,
        end=2.0,
        chunk_type="screen_share",
    )
    wrapper = SimpleNamespace(chunk=inner)

    crud.save_chunks(db, "vid-1", [wrapper])

    added = db.add.call_args[0][0]
    assert added.speaker == "SPEAKER_01"
    assert added.chunk_type == "screen_share"


def test_save_chunks_from_dataclass():
    db = MagicMock()

    @dataclass
    class FakeChunk:
        speaker: str
        text: str
        start: float
        end: float
        chunk_type: str

    crud.save_chunks(
        db,
        "vid-1",
        [FakeChunk("A", "dataclass text", 0.0, 1.0, "transcript")],
    )

    added = db.add.call_args[0][0]
    assert added.text == "dataclass text"


def test_get_chunks_returns_ordered_results():
    db = MagicMock()
    chunk_a = Chunk(id="a", video_id="vid-1", text="a", start=0.0, end=1.0)
    chunk_b = Chunk(id="b", video_id="vid-1", text="b", start=2.0, end=3.0)
    _chunk_query(db, [chunk_a, chunk_b])

    result = crud.get_chunks(db, "vid-1")

    assert result == [chunk_a, chunk_b]
    db.query.assert_called_with(Chunk)


def test_get_chunks_empty():
    db = MagicMock()
    _chunk_query(db, [])

    assert crud.get_chunks(db, "vid-1") == []


def test_upsert_summary_inserts_new_row():
    db = MagicMock()
    _summary_query(db, None)
    content = {"agenda": ["topic"]}

    summary = crud.upsert_summary(db, "vid-1", "meeting", content)

    assert summary.id == "vid-1_summary"
    assert summary.video_id == "vid-1"
    assert summary.content == content
    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once()


def test_upsert_summary_updates_existing_row():
    db = MagicMock()
    existing = Summary(
        id="vid-1_summary",
        video_id="vid-1",
        video_type="meeting",
        content={"agenda": ["old"]},
    )
    db.query.return_value.filter.return_value.first.return_value = existing
    new_content = {"agenda": ["updated"]}

    summary = crud.upsert_summary(db, "vid-1", "lecture", new_content)

    assert summary is existing
    assert existing.content == new_content
    assert existing.video_type == "lecture"
    db.add.assert_not_called()
    db.commit.assert_called_once()


def test_update_video_status_sets_error_message():
    db = MagicMock()
    video = SimpleNamespace(status="processing", progress=50, error=None)
    _video_query(db, video)

    crud.update_video_status(db, "vid-1", status="failed", error="pipeline exploded")

    assert video.status == "failed"
    assert video.error == "pipeline exploded"


def test_save_chunks_from_generic_object():
    db = MagicMock()
    generic = SimpleNamespace(
        speaker="SPEAKER_02",
        text="generic object text",
        start=3.0,
        end=4.0,
        chunk_type="speech",
    )

    crud.save_chunks(db, "vid-1", [generic])

    added = db.add.call_args[0][0]
    assert added.text == "generic object text"
    assert added.speaker == "SPEAKER_02"


def test_get_summary_found_and_not_found():
    db = MagicMock()
    row = Summary(id="vid-1_summary", video_id="vid-1", video_type="meeting", content={})
    db.query.return_value.filter.return_value.first.return_value = row

    assert crud.get_summary(db, "vid-1") is row

    db.query.return_value.filter.return_value.first.return_value = None
    assert crud.get_summary(db, "vid-1") is None
