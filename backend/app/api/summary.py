from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import SessionLocal
from app.summarizer import generate_summary

router = APIRouter(prefix="/videos", tags=["summary"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Response models ────────────────────────────────────────────────────────────


class MeetingContent(BaseModel):
    agenda: list[str]
    decisions: list[str]
    action_items: list[str]
    participants: list[str]


class LectureContent(BaseModel):
    topic_outline: list[str]
    key_concepts: list[str]
    examples: list[str]


class SummaryResponse(BaseModel):
    video_id: str
    video_type: str
    created_at: datetime
    content: MeetingContent | LectureContent


# ── Helpers ────────────────────────────────────────────────────────────────────


def _row_to_response(row) -> SummaryResponse:
    """Convert a Summary ORM row to the typed response model."""
    if row.video_type == "meeting":
        content = MeetingContent(**row.content)
    else:
        content = LectureContent(**row.content)
    return SummaryResponse(
        video_id=row.video_id,
        video_type=row.video_type,
        created_at=row.created_at,
        content=content,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.get("/{video_id}/summary", response_model=SummaryResponse)
def get_summary(video_id: str, db: Session = Depends(get_db)):
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Video is not ready for summarization. Current status: {video.status}",
        )

    row = crud.get_summary(db, video_id)
    if row:
        # Already generated — return from DB without calling LLM.
        return _row_to_response(row)

    # Lazy first-time generation.
    try:
        generate_summary(video_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Summary generation service unavailable"
        ) from exc

    row = crud.get_summary(db, video_id)
    if not row:
        raise HTTPException(status_code=500, detail="Summary was not persisted after generation")
    return _row_to_response(row)


@router.post("/{video_id}/summary/regenerate", response_model=SummaryResponse)
def regenerate_summary(video_id: str, db: Session = Depends(get_db)):
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Video is not ready for summarization. Current status: {video.status}",
        )

    try:
        generate_summary(video_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Summary generation service unavailable"
        ) from exc

    row = crud.get_summary(db, video_id)
    if not row:
        raise HTTPException(status_code=500, detail="Summary was not persisted after regeneration")
    return _row_to_response(row)
