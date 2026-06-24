import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.pipeline.shared.text_embedder import embed_texts


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collections(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    for name, size in [("chunks_text", 384), ("chunks_visual", 512)]:
        if name not in existing:
            client.create_collection(
                name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE),
            )


def _normalize_chunk(chunk: Any, video_id: str, index: int) -> dict:
    """Chuẩn hóa chunk về 1 dict thống nhất cho việc index.

    Hỗ trợ 3 dạng input:
    - Meeting: ``MeetingChunkWithEmbedding(chunk=MeetingChunk(...), visual_embedding=...)``
    - Lecture: ``dict`` (speaker, text, start, end, chunk_type, + clip_embedding/slide_index)
    - Dataclass chunk bất kỳ có các thuộc tính tương ứng.

    ``chunk_id`` được sinh theo index để khớp với ``crud.save_chunks`` (Postgres),
    đảm bảo citation map đúng giữa Qdrant payload và chunk row trong DB.
    """
    # Meeting pipeline bọc chunk trong wrapper + giữ CLIP vector ở ngoài.
    visual = getattr(chunk, "visual_embedding", None)
    inner = getattr(chunk, "chunk", None)
    if inner is not None:
        chunk = inner

    if isinstance(chunk, dict):
        text = chunk.get("combined_text") or chunk.get("text", "")
        clip = chunk.get("clip_embedding")
        return {
            "chunk_id": f"{video_id}_chunk_{index:04d}",
            "text": text,
            "start": chunk.get("start", 0.0),
            "end": chunk.get("end", 0.0),
            "speaker": chunk.get("speaker"),
            "chunk_type": chunk.get("chunk_type", "transcript"),
            "slide_index": chunk.get("slide_index"),
            "visual": clip if clip else visual,
        }

    text = getattr(chunk, "combined_text", None) or getattr(chunk, "text", "")
    clip = getattr(chunk, "clip_embedding", None)
    return {
        "chunk_id": f"{video_id}_chunk_{index:04d}",
        "text": text,
        "start": getattr(chunk, "start", 0.0),
        "end": getattr(chunk, "end", 0.0),
        "speaker": getattr(chunk, "speaker", None),
        "chunk_type": getattr(chunk, "chunk_type", "transcript"),
        "slide_index": getattr(chunk, "slide_index", None),
        "visual": clip if clip else visual,
    }


def index_chunks(video_id: str, chunks: list) -> None:
    if not chunks:
        return

    client = get_qdrant_client()
    ensure_collections(client)

    normalized = [_normalize_chunk(c, video_id, i) for i, c in enumerate(chunks)]
    text_vectors = embed_texts([n["text"] for n in normalized])

    text_points = []
    visual_points = []

    for n, vector in zip(normalized, text_vectors):
        payload = {
            "video_id": video_id,
            "chunk_id": n["chunk_id"],
            "start": n["start"],
            "end": n["end"],
            "text": n["text"],
            "chunk_type": n["chunk_type"],
        }
        if n["speaker"]:
            payload["speaker"] = n["speaker"]
        if n["slide_index"] is not None:
            payload["slide_index"] = n["slide_index"]

        text_points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, n["chunk_id"])),
                vector=vector,
                payload=payload,
            )
        )

        if n["visual"]:
            visual_points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, n["chunk_id"] + "_visual")),
                    vector=n["visual"],
                    payload=payload,
                )
            )

    if text_points:
        client.upsert(collection_name="chunks_text", points=text_points)
    if visual_points:
        client.upsert(collection_name="chunks_visual", points=visual_points)
