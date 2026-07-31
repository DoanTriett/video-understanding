import json
import os
import tempfile
import uuid

from botocore.exceptions import ClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.db import crud
from app.db.session import SessionLocal
from app.limiter import LIMIT_UPLOAD, limiter
from app.models.video import JobStatus, VideoStatusResponse, VideoType, VideoUploadResponse
from app.storage import download_to_path, presigned_url, upload_bytes
from app.store import get_progress, set_progress
from workers.celery_app import celery_app

router = APIRouter(prefix="/videos", tags=["videos"])
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
PROCESS_VIDEO_TASK = "workers.tasks.process_video"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/upload", response_model=VideoUploadResponse)
@limiter.limit(LIMIT_UPLOAD)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    video_type: VideoType = Form(...),
    db: Session = Depends(get_db),
):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not allowed.")

    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413, detail=f"File too large. Max: {settings.max_file_size_mb}MB"
        )

    video_id = str(uuid.uuid4())
    object_key = f"{video_id}/source{file_ext}"
    try:
        upload_bytes(contents, object_key)
    except BotoConnectionError as exc:
        raise HTTPException(status_code=503, detail="Storage service unavailable") from exc

    crud.create_video(
        db=db,
        video_id=video_id,
        filename=file.filename,
        video_type=video_type.value,
    )
    crud.set_video_object_key(db, video_id, object_key)

    set_progress(video_id, stage="queued", pct=0)
    celery_app.send_task(PROCESS_VIDEO_TASK, args=[video_id, video_type.value])

    return VideoUploadResponse(
        video_id=video_id,
        filename=file.filename,
        file_size_mb=round(file_size_mb, 2),
        status=JobStatus.PENDING,
        message="Video uploaded. Processing started.",
    )


@router.get("/{video_id}/status", response_model=VideoStatusResponse)
def get_video_status(video_id: str, db: Session = Depends(get_db)):
    progress = get_progress(video_id)
    if progress:
        video = crud.get_video(db, video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        return VideoStatusResponse(
            video_id=video_id,
            filename=video.filename,
            status=JobStatus.PROCESSING,
            progress_percent=progress["pct"],
            video_type=video.video_type,
            error_message=None,
            created_at=video.created_at,
        )

    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return VideoStatusResponse(
        video_id=video_id,
        filename=video.filename,
        status=JobStatus(video.status),
        progress_percent=video.progress,
        video_type=video.video_type,
        error_message=video.error,
        created_at=video.created_at,
    )


@router.get("/{video_id}/transcript")
def get_transcript(video_id: str, db: Session = Depends(get_db)):
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done":
        raise HTTPException(status_code=400, detail=f"Not ready. Status: {video.status}")

    object_key = f"{video_id}/transcript.json"
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = os.path.join(tmp_dir, "transcript.json")
        try:
            download_to_path(object_key, local_path)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                raise HTTPException(status_code=404, detail="Transcript file not found")
            raise

        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)


@router.get("/{video_id}/chunks")
def get_chunks(video_id: str, db: Session = Depends(get_db)):
    video = crud.get_video(db, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done":
        raise HTTPException(status_code=400, detail=f"Not ready: {video.status}")

    chunks = crud.get_chunks(db, video_id)
    return {
        "video_id": video_id,
        "num_chunks": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "speaker": c.speaker,
                "text": c.text,
                "start": c.start,
                "end": c.end,
                "chunk_type": c.chunk_type,
            }
            for c in chunks
        ],
    }


@router.get("/{video_id}/url")
def get_video_url(video_id: str, db: Session = Depends(get_db)):
    video = crud.get_video(db, video_id)
    if not video or not video.object_key:
        raise HTTPException(status_code=404, detail="Video not found")
    return {"url": presigned_url(video.object_key)}
