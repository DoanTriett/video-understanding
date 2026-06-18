# backend/app/db/crud.py
from dataclasses import asdict, is_dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import Chunk, Video

# ─── Video operations ───


def create_video(db: Session, video_id: str, filename: str, video_type: str) -> Video:
    """Tạo row mới khi user upload."""
    video = Video(
        id=video_id,
        filename=filename,
        video_type=video_type,
        status="pending",
        progress=0,
        created_at=datetime.utcnow(),
    )
    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def get_video(db: Session, video_id: str) -> Video | None:
    """Đọc video record từ Postgres."""
    return db.query(Video).filter(Video.id == video_id).first()


def update_video_status(
    db: Session, video_id: str, status: str, progress: int = None, error: str = None
):
    """Cập nhật status và progress khi task chạy xong hoặc failed."""
    video = get_video(db, video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found in database")
    video.status = status
    if progress is not None:
        video.progress = progress
    if error is not None:
        video.error = error
    db.commit()


def set_video_object_key(db: Session, video_id: str, object_key: str):
    """Lưu MinIO object key sau khi upload file."""
    video = get_video(db, video_id)
    if video:
        video.object_key = object_key
        db.commit()


def get_video_object_key(db: Session, video_id: str) -> str | None:
    """Lấy MinIO object key để Celery download file."""
    video = get_video(db, video_id)
    return video.object_key if video else None


# ─── Chunk operations ───


def _chunk_to_dict(chunk) -> dict:
    """Normalize pipeline chunk objects/dicts before saving."""
    if isinstance(chunk, dict):
        return chunk

    # Meeting pipeline returns MeetingChunkWithEmbedding(chunk=MeetingChunk(...)).
    if hasattr(chunk, "chunk"):
        chunk = chunk.chunk

    if is_dataclass(chunk):
        return asdict(chunk)

    return {
        "speaker": getattr(chunk, "speaker", None),
        "text": getattr(chunk, "text", ""),
        "start": getattr(chunk, "start", 0.0),
        "end": getattr(chunk, "end", 0.0),
        "chunk_type": getattr(chunk, "chunk_type", "transcript"),
    }


def save_chunks(db: Session, video_id: str, chunks: list):
    """Lưu toàn bộ chunks vào Postgres sau khi pipeline xong.

    Mỗi chunk có thể là dict hoặc object từ meeting pipeline.
    """
    # Xóa chunks cũ nếu có (trường hợp re-process)
    db.query(Chunk).filter(Chunk.video_id == video_id).delete()

    for i, chunk in enumerate(chunks):
        chunk_data = _chunk_to_dict(chunk)
        db_chunk = Chunk(
            id=f"{video_id}_chunk_{i:04d}",
            video_id=video_id,
            speaker=chunk_data.get("speaker"),
            text=chunk_data.get("text", ""),
            start=chunk_data.get("start", 0.0),
            end=chunk_data.get("end", 0.0),
            chunk_type=chunk_data.get("chunk_type", "transcript"),
        )
        db.add(db_chunk)

    db.commit()


def get_chunks(db: Session, video_id: str) -> list[Chunk]:
    """Lấy tất cả chunks của 1 video, sắp xếp theo thời gian."""
    return db.query(Chunk).filter(Chunk.video_id == video_id).order_by(Chunk.start).all()
