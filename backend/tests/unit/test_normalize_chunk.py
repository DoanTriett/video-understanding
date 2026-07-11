"""Unit tests cho _normalize_chunk — logic chuẩn hóa payload trước khi index."""

from types import SimpleNamespace

from app.pipeline.lecture.chunker import LectureChunk
from app.pipeline.meeting.chunker import MeetingChunk
from app.pipeline.shared.indexer import _normalize_chunk


def test_normalize_dict_with_text():
    chunk = {
        "text": "meeting speech",
        "start": 1.0,
        "end": 5.0,
        "speaker": "SPEAKER_00",
        "chunk_type": "speech",
    }

    result = _normalize_chunk(chunk, "vid1", 0)

    assert result == {
        "chunk_id": "vid1_chunk_0000",
        "text": "meeting speech",
        "start": 1.0,
        "end": 5.0,
        "speaker": "SPEAKER_00",
        "chunk_type": "speech",
        "slide_index": None,
        "visual": None,
    }


def test_normalize_dict_prefers_combined_text():
    chunk = {
        "combined_text": "[SLIDE] Title\n[SPEECH] hello",
        "text": "ignored fallback",
        "start": 0.0,
        "end": 10.0,
        "slide_index": 2,
        "clip_embedding": [0.5] * 512,
    }

    result = _normalize_chunk(chunk, "vidL", 3)

    assert result["chunk_id"] == "vidL_chunk_0003"
    assert result["text"] == "[SLIDE] Title\n[SPEECH] hello"
    assert result["slide_index"] == 2
    assert result["visual"] == [0.5] * 512
    assert result["chunk_type"] == "transcript"


def test_normalize_meeting_wrapper_unwraps_and_uses_visual_embedding():
    inner = MeetingChunk(
        chunk_id="ignored",
        video_id="vidM",
        speaker="SPEAKER_01",
        text="screen content",
        start=2.0,
        end=8.0,
        turn_index=0,
        chunk_type="screen_share",
    )
    wrapper = SimpleNamespace(chunk=inner, visual_embedding=[0.9] * 512)

    result = _normalize_chunk(wrapper, "vidM", 1)

    assert result["chunk_id"] == "vidM_chunk_0001"
    assert result["text"] == "screen content"
    assert result["speaker"] == "SPEAKER_01"
    assert result["chunk_type"] == "screen_share"
    assert result["visual"] == [0.9] * 512


def test_normalize_lecture_chunk_dataclass():
    chunk = LectureChunk(
        chunk_id="ignored",
        video_id="vidL",
        slide_index=0,
        start=0.0,
        end=10.0,
        transcript_text="spoken",
        ocr_text="slide title",
        combined_text="[SLIDE] slide title\n[SPEECH] spoken",
        frame_path="/tmp/frame.jpg",
        clip_embedding=[0.3] * 512,
    )

    result = _normalize_chunk(chunk, "vidL", 0)

    assert result["text"] == "[SLIDE] slide title\n[SPEECH] spoken"
    assert result["slide_index"] == 0
    assert result["visual"] == [0.3] * 512
    assert result["speaker"] is None
    assert result["chunk_type"] == "transcript"


def test_normalize_empty_text():
    chunk = {"text": "", "start": 0.0, "end": 1.0}

    result = _normalize_chunk(chunk, "vid1", 0)

    assert result["text"] == ""


def test_normalize_defaults_for_missing_fields():
    chunk = SimpleNamespace(text="minimal")

    result = _normalize_chunk(chunk, "vid1", 5)

    assert result == {
        "chunk_id": "vid1_chunk_0005",
        "text": "minimal",
        "start": 0.0,
        "end": 0.0,
        "speaker": None,
        "chunk_type": "transcript",
        "slide_index": None,
        "visual": None,
    }


def test_normalize_dataclass_clip_over_wrapper_visual():
    inner = SimpleNamespace(
        text="lecture",
        start=0.0,
        end=5.0,
        clip_embedding=[0.1] * 512,
    )
    wrapper = SimpleNamespace(chunk=inner, visual_embedding=[0.9] * 512)

    result = _normalize_chunk(wrapper, "vid1", 0)

    assert result["visual"] == [0.1] * 512
