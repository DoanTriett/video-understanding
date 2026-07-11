"""Unit tests for app.pipeline.shared.retriever — pure helpers + mocked Qdrant."""

from types import SimpleNamespace
from unittest.mock import patch

from app.pipeline.shared import retriever


def test_route_query_text_by_default():
    assert retriever.route_query("What was discussed in the meeting?") == "text"


def test_route_query_visual_for_slide_keyword():
    assert retriever.route_query("What is shown on the slide?") == "visual"


def test_route_query_visual_for_screen_keyword():
    assert retriever.route_query("Describe the screen content") == "visual"


def test_build_context_empty_list():
    assert retriever.build_context([]) == ""


def test_build_context_single_chunk_without_speaker():
    chunks = [{"start": 30.0, "text": "Hello world", "speaker": None}]

    context = retriever.build_context(chunks)

    assert context == "[00:30] Hello world"


def test_build_context_sorts_by_start_and_formats_speaker():
    chunks = [
        {"start": 125.0, "text": "Second", "speaker": "SPEAKER_01"},
        {"start": 10.0, "text": "First", "speaker": "SPEAKER_00"},
    ]

    context = retriever.build_context(chunks)

    assert context.splitlines() == [
        "[SPEAKER_00 @ 00:10] First",
        "[SPEAKER_01 @ 02:05] Second",
    ]


def test_retrieve_visual_route_skips_visual_collection(monkeypatch):
    hit = SimpleNamespace(
        score=0.91,
        payload={
            "chunk_id": "vid_chunk_0000",
            "speaker": "SPEAKER_00",
            "start": 5.0,
            "end": 15.0,
            "text": "Slide content",
            "chunk_type": "transcript",
        },
    )
    client = SimpleNamespace(search=lambda **kwargs: [hit])
    monkeypatch.setattr(retriever, "_get_qdrant_client", lambda: client)

    with patch("app.pipeline.shared.retriever.embed_text", return_value=[0.1] * 384):
        results = retriever.retrieve("vid-1", "What is on the slide?", top_k=3)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "vid_chunk_0000"
    assert results[0]["text"] == "Slide content"


def test_retrieve_deduplicates_by_chunk_id(monkeypatch):
    hits = [
        SimpleNamespace(
            score=0.5,
            payload={"chunk_id": "dup", "text": "low", "start": 0.0, "end": 1.0},
        ),
        SimpleNamespace(
            score=0.9,
            payload={"chunk_id": "dup", "text": "high", "start": 0.0, "end": 1.0},
        ),
    ]
    client = SimpleNamespace(search=lambda **kwargs: hits)
    monkeypatch.setattr(retriever, "_get_qdrant_client", lambda: client)

    with patch("app.pipeline.shared.retriever.embed_text", return_value=[0.1] * 384):
        results = retriever.retrieve("vid-1", "question", top_k=5)

    assert len(results) == 1
    assert results[0]["text"] == "high"
    assert results[0]["score"] == 0.9
