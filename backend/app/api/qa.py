import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException
from redis.exceptions import RedisError
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import SessionLocal
from app.limiter import LIMIT_ASK, limiter
from app.llm import generate_answer
from app.pipeline.shared.retriever import build_context, retrieve
from app.semantic_cache import get_cached_answer, set_cached_answer

logger = logging.getLogger("app.api.qa")

router = APIRouter(prefix="/videos", tags=["qa"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)


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
@limiter.limit(LIMIT_ASK)
def ask_video(
    request: Request,
    video_id: str,
    body: AskRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Video is still processing. Current status: {video.status}",
        )

    try:
        cached = get_cached_answer(video_id, body.question)
    except RedisError as exc:
        logger.warning("Semantic cache read failed (Redis down?), treating as cache miss: %s", exc)
        cached = None

    if cached is not None:
        response.headers["X-Cache"] = "HIT"
        return AskResponse(
            answer=cached["answer"],
            citations=[Citation(**c) for c in cached["citations"]],
        )

    response.headers["X-Cache"] = "MISS"

    try:
        chunks = retrieve(video_id, body.question, body.top_k)
    except ResponseHandlingException as exc:
        raise HTTPException(status_code=503, detail="Retrieval service unavailable") from exc
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

    result = AskResponse(answer=answer, citations=citations)
    try:
        set_cached_answer(
            video_id,
            body.question,
            {"answer": result.answer, "citations": [c.model_dump() for c in result.citations]},
        )
    except RedisError as exc:
        # Answer was already generated successfully — a cache-write failure must not
        # lose it. Log and still return the answer to the client.
        logger.warning("Semantic cache write failed (Redis down?), answer still returned: %s", exc)
    return result
