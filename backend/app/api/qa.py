from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import SessionLocal
from app.llm import generate_answer
from app.pipeline.shared.retriever import build_context, retrieve

router = APIRouter(prefix="/videos", tags=["qa"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AskRequest(BaseModel):
    question: str
    top_k: int = 6


class Citation(BaseModel):
    chunk_id: str
    speaker: str | None
    start: float
    end: float
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


@router.post("/{video_id}/ask", response_model=AskResponse)
def ask_video(video_id: str, body: AskRequest, db: Session = Depends(get_db)):
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Video is still processing. Current status: {video.status}",
        )

    chunks = retrieve(video_id, body.question, body.top_k)
    if not chunks:
        return AskResponse(
            answer="Không tìm thấy thông tin liên quan trong video.",
            citations=[],
        )

    context = build_context(chunks)
    try:
        answer = generate_answer(body.question, context)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="Answer generation service unavailable"
        ) from exc

    citations = [
        Citation(
            chunk_id=c["chunk_id"],
            speaker=c.get("speaker"),
            start=c["start"],
            end=c["end"],
            text=c["text"],
        )
        for c in chunks
    ]

    return AskResponse(answer=answer, citations=citations)
