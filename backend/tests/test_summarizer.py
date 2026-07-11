"""Tests for app/summarizer.py.

Split into two sections:
  - Unit tests (no real DB / LLM) — test JSON parsing and retry logic via mocks
  - Integration test (real DB + real Ollama) — marked with @pytest.mark.integration
    Run with: pytest tests/test_summarizer.py -m integration -s
"""

import sys

sys.path.insert(0, ".")

import json
from unittest.mock import MagicMock, patch

import pytest

from app.summarizer import (
    _LECTURE_KEYS,
    _MEETING_KEYS,
    _extract_json_string,
    _format_transcript,
    _parse_and_validate,
    generate_summary,
)

# ── Unit tests ─────────────────────────────────────────────────────────────────


class FakeChunk:
    def __init__(self, start, text, speaker=None):
        self.start = start
        self.text = text
        self.speaker = speaker


def test_format_transcript_with_speaker():
    chunks = [
        FakeChunk(start=65.0, text="Hello there.", speaker="Speaker A"),
        FakeChunk(start=125.0, text="Good morning.", speaker="Speaker B"),
    ]
    result = _format_transcript(chunks)
    assert "[Speaker A @ 01:05] Hello there." in result
    assert "[Speaker B @ 02:05] Good morning." in result


def test_format_transcript_no_speaker():
    chunks = [FakeChunk(start=30.0, text="No speaker here.")]
    result = _format_transcript(chunks)
    assert "[00:30] No speaker here." in result


def test_extract_json_string_plain():
    raw = '{"agenda": ["topic1"]}'
    assert _extract_json_string(raw) == '{"agenda": ["topic1"]}'


def test_extract_json_string_fenced():
    raw = '```json\n{"agenda": ["topic1"]}\n```'
    assert _extract_json_string(raw) == '{"agenda": ["topic1"]}'


def test_extract_json_string_fenced_no_lang():
    raw = '```\n{"key": "val"}\n```'
    assert _extract_json_string(raw) == '{"key": "val"}'


def test_extract_json_string_with_preamble():
    raw = 'Here is the summary:\n{"agenda": ["topic1"]}\nDone.'
    result = _extract_json_string(raw)
    parsed = json.loads(result)
    assert parsed["agenda"] == ["topic1"]


def test_parse_and_validate_ok():
    raw = '{"agenda": ["a"], "decisions": ["d"], "action_items": ["ai"], "participants": ["P1"]}'
    result = _parse_and_validate(raw, _MEETING_KEYS)
    assert set(result.keys()) == _MEETING_KEYS


def test_parse_and_validate_missing_key():
    raw = '{"agenda": ["a"], "decisions": ["d"], "action_items": ["ai"]}'  # missing participants
    with pytest.raises(ValueError, match="missing required keys"):
        _parse_and_validate(raw, _MEETING_KEYS)


def test_parse_and_validate_strips_extra_keys():
    raw = json.dumps(
        {
            "agenda": [],
            "decisions": [],
            "action_items": [],
            "participants": [],
            "extra_field": "should be dropped",
        }
    )
    result = _parse_and_validate(raw, _MEETING_KEYS)
    assert "extra_field" not in result
    assert set(result.keys()) == _MEETING_KEYS


def test_parse_and_validate_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_and_validate("not json at all", _MEETING_KEYS)


# ── Retry logic tests (mocked LLM + DB) ───────────────────────────────────────

_VALID_MEETING_JSON = json.dumps(
    {
        "agenda": ["Budget review"],
        "decisions": ["Approve Q3 budget"],
        "action_items": ["Alice to send report"],
        "participants": ["Speaker A", "Speaker B"],
    }
)

_VALID_LECTURE_JSON = json.dumps(
    {
        "topic_outline": ["Introduction to ML"],
        "key_concepts": ["gradient descent"],
        "examples": ["linear regression example"],
    }
)


def _make_fake_video(video_type: str):
    v = MagicMock()
    v.video_type = video_type
    return v


def _make_fake_chunks():
    return [FakeChunk(start=0.0, text="Some transcript text.", speaker="Speaker A")]


def test_retry_logic_success_on_second_attempt():
    """First call returns garbage JSON; second call returns valid JSON."""
    call_sequence = ["not valid json at all", _VALID_MEETING_JSON]

    with (
        patch("app.summarizer.crud.get_video", return_value=_make_fake_video("meeting")),
        patch("app.summarizer.crud.get_chunks", return_value=_make_fake_chunks()),
        patch("app.summarizer.crud.upsert_summary"),
        patch("app.summarizer.call_llm", side_effect=call_sequence),
        patch("app.summarizer.SessionLocal"),
    ):
        result = generate_summary("fake-video-id")

    assert set(result.keys()) == _MEETING_KEYS


def test_retry_logic_raises_after_two_failures():
    """Both attempts return garbage — RuntimeError must be raised."""
    with (
        patch("app.summarizer.crud.get_video", return_value=_make_fake_video("meeting")),
        patch("app.summarizer.crud.get_chunks", return_value=_make_fake_chunks()),
        patch("app.summarizer.crud.upsert_summary"),
        patch("app.summarizer.call_llm", return_value="definitely not json"),
        patch("app.summarizer.SessionLocal"),
    ):
        with pytest.raises(RuntimeError, match="invalid JSON after retry"):
            generate_summary("fake-video-id")


def test_retry_not_triggered_on_first_success():
    """Valid JSON on first try — call_llm called exactly once."""
    with (
        patch("app.summarizer.crud.get_video", return_value=_make_fake_video("lecture")),
        patch("app.summarizer.crud.get_chunks", return_value=_make_fake_chunks()),
        patch("app.summarizer.crud.upsert_summary"),
        patch("app.summarizer.call_llm", return_value=_VALID_LECTURE_JSON) as mock_llm,
        patch("app.summarizer.SessionLocal"),
    ):
        result = generate_summary("fake-video-id")

    mock_llm.assert_called_once()
    assert set(result.keys()) == _LECTURE_KEYS


def test_unsupported_video_type_raises():
    with (
        patch("app.summarizer.crud.get_video", return_value=_make_fake_video("unknown")),
        patch("app.summarizer.crud.get_chunks", return_value=_make_fake_chunks()),
        patch("app.summarizer.SessionLocal"),
    ):
        with pytest.raises(ValueError, match="Unsupported video_type"):
            generate_summary("fake-video-id")


# ── Integration tests (real DB + real Ollama) ──────────────────────────────────

MEETING_VIDEO_ID = "1320b859-0496-4bf0-8fe1-3bd564303c8c"  # meeting, 19 chunks, done
LECTURE_VIDEO_ID = "fd7d6717-2a3e-4266-828e-1f99c0869724"  # lecture, 8 chunks, done


@pytest.mark.integration
@pytest.mark.requires_ollama
def test_generate_summary_meeting_real():
    result = generate_summary(MEETING_VIDEO_ID)
    print("\n=== MEETING SUMMARY ===")
    print(json.dumps(result, indent=2, ensure_ascii=True))

    assert set(result.keys()) == _MEETING_KEYS
    assert isinstance(result["agenda"], list)
    assert isinstance(result["decisions"], list)
    assert isinstance(result["action_items"], list)
    assert isinstance(result["participants"], list)


@pytest.mark.integration
@pytest.mark.requires_ollama
def test_generate_summary_lecture_real():
    result = generate_summary(LECTURE_VIDEO_ID)
    print("\n=== LECTURE SUMMARY ===")
    print(json.dumps(result, indent=2, ensure_ascii=True))

    assert set(result.keys()) == _LECTURE_KEYS
    assert isinstance(result["topic_outline"], list)
    assert isinstance(result["key_concepts"], list)
    assert isinstance(result["examples"], list)


@pytest.mark.integration
@pytest.mark.requires_ollama
def test_summary_persisted_to_db():
    """Verify the record is actually in the summaries table after generation."""
    from app.db import crud
    from app.db.session import SessionLocal

    generate_summary(MEETING_VIDEO_ID)  # ensure it's there (idempotent)

    db = SessionLocal()
    try:
        row = crud.get_summary(db, MEETING_VIDEO_ID)
        assert row is not None, "Summary row not found in DB"
        assert row.video_id == MEETING_VIDEO_ID
        assert row.video_type == "meeting"
        assert set(row.content.keys()) == _MEETING_KEYS
        print(f"\nDB row id={row.id}, video_type={row.video_type}, created_at={row.created_at}")
    finally:
        db.close()


@pytest.mark.integration
@pytest.mark.requires_ollama
def test_upsert_idempotent():
    """Calling generate_summary twice should not create duplicate rows."""
    from app.db.models import Summary
    from app.db.session import SessionLocal

    generate_summary(MEETING_VIDEO_ID)
    generate_summary(MEETING_VIDEO_ID)

    db = SessionLocal()
    try:
        count = db.query(Summary).filter(Summary.video_id == MEETING_VIDEO_ID).count()
        assert count == 1, f"Expected 1 summary row, got {count}"
    finally:
        db.close()
