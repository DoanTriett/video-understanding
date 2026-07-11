"""Unit tests cho index_chunks — mock QdrantClient + embed_texts.

Kiểm tra:
- upsert được gọi với payload đúng format,
- chunk_id trong payload là index-based (khớp crud.save_chunks),
- visual point chỉ tạo khi chunk có CLIP embedding,
- hỗ trợ cả meeting wrapper lẫn lecture dict.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.pipeline.shared import indexer


def _make_client_mock():
    """Mock QdrantClient: chưa có collection nào (để ensure_collections không lỗi)."""
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    return client


def _points_for(client, collection_name):
    """Trả về list points của lần upsert vào collection chỉ định."""
    for call in client.upsert.call_args_list:
        if call.kwargs.get("collection_name") == collection_name:
            return call.kwargs["points"]
    return None


def test_lecture_dict_chunk_text_payload():
    client = _make_client_mock()
    chunks = [
        {
            "chunk_id": "ignored_id",
            "text": "slide one speech",
            "start": 0.0,
            "end": 10.0,
            "speaker": None,
            "chunk_type": "lecture_slide",
            "slide_index": 0,
            "clip_embedding": [0.2] * 512,
        }
    ]

    with (
        patch.object(indexer, "get_qdrant_client", return_value=client),
        patch.object(indexer, "embed_texts", return_value=[[0.1] * 384]) as mock_embed,
    ):
        indexer.index_chunks("vid1", chunks)

    mock_embed.assert_called_once_with(["slide one speech"])

    text_points = _points_for(client, "chunks_text")
    assert text_points is not None and len(text_points) == 1
    payload = text_points[0].payload
    assert payload == {
        "video_id": "vid1",
        "chunk_id": "vid1_chunk_0000",  # index-based, KHÔNG dùng "ignored_id"
        "start": 0.0,
        "end": 10.0,
        "text": "slide one speech",
        "chunk_type": "lecture_slide",
        "slide_index": 0,
    }
    assert text_points[0].vector == [0.1] * 384


def test_lecture_chunk_with_clip_creates_visual_point():
    client = _make_client_mock()
    chunks = [
        {
            "text": "x",
            "start": 0.0,
            "end": 5.0,
            "speaker": None,
            "chunk_type": "lecture_slide",
            "slide_index": 0,
            "clip_embedding": [0.3] * 512,
        }
    ]

    with (
        patch.object(indexer, "get_qdrant_client", return_value=client),
        patch.object(indexer, "embed_texts", return_value=[[0.1] * 384]),
    ):
        indexer.index_chunks("vid1", chunks)

    visual_points = _points_for(client, "chunks_visual")
    assert visual_points is not None and len(visual_points) == 1
    assert visual_points[0].vector == [0.3] * 512
    assert visual_points[0].payload["chunk_id"] == "vid1_chunk_0000"


def test_meeting_wrapper_chunk_unwrapped_and_indexed():
    client = _make_client_mock()
    # Giả lập MeetingChunkWithEmbedding(chunk=MeetingChunk(...), visual_embedding=...)
    inner = SimpleNamespace(
        text="meeting turn text",
        start=1.0,
        end=2.0,
        speaker="SPEAKER_00",
        chunk_type="screen_share",
    )
    wrapper = SimpleNamespace(chunk=inner, visual_embedding=[0.5] * 512)

    with (
        patch.object(indexer, "get_qdrant_client", return_value=client),
        patch.object(indexer, "embed_texts", return_value=[[0.1] * 384]) as mock_embed,
    ):
        indexer.index_chunks("vidM", [wrapper])

    mock_embed.assert_called_once_with(["meeting turn text"])

    text_points = _points_for(client, "chunks_text")
    payload = text_points[0].payload
    assert payload["chunk_id"] == "vidM_chunk_0000"
    assert payload["speaker"] == "SPEAKER_00"
    assert payload["chunk_type"] == "screen_share"
    assert "slide_index" not in payload

    # visual_embedding của wrapper → visual point.
    visual_points = _points_for(client, "chunks_visual")
    assert visual_points is not None and visual_points[0].vector == [0.5] * 512


def test_chunk_without_visual_skips_visual_collection():
    client = _make_client_mock()
    inner = SimpleNamespace(
        text="no visual",
        start=0.0,
        end=1.0,
        speaker="SPEAKER_01",
        chunk_type="transcript",
    )
    wrapper = SimpleNamespace(chunk=inner, visual_embedding=None)

    with (
        patch.object(indexer, "get_qdrant_client", return_value=client),
        patch.object(indexer, "embed_texts", return_value=[[0.1] * 384]),
    ):
        indexer.index_chunks("vidM", [wrapper])

    assert _points_for(client, "chunks_text") is not None
    assert _points_for(client, "chunks_visual") is None


def test_empty_chunks_noop():
    with patch.object(indexer, "get_qdrant_client") as mock_get_client:
        indexer.index_chunks("vid1", [])
    mock_get_client.assert_not_called()
