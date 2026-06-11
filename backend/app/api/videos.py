import os
import uuid
import aiofiles
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.video import VideoUploadResponse, VideoStatusResponse, JobStatus, VideoType
from app.config import settings
from app.store import set_job, get_job
from workers.tasks import process_video

router = APIRouter(prefix="/videos", tags=["videos"])
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    video_type: VideoType = VideoType.UNKNOWN
):
    # Validate extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Supported: {ALLOWED_EXTENSIONS}")

    # Validate size
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {settings.max_file_size_mb}MB")

    # Lưu file
    video_id = str(uuid.uuid4())
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{video_id}{file_ext}")
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    # Lưu job vào Redis (cả FastAPI lẫn Celery đều đọc được)
    set_job(video_id, {
        "video_id": video_id,
        "filename": file.filename,
        "file_path": file_path,
        "status": JobStatus.PENDING,
        "video_type": video_type,
        "progress_percent": 0,
        "error_message": None,
        "created_at": datetime.now(),
    })

    # Gửi job cho Celery
    process_video.delay(video_id)

    return VideoUploadResponse(
        video_id=video_id,
        filename=file.filename,
        file_size_mb=round(file_size_mb, 2),
        status=JobStatus.PENDING,
        message="Video uploaded. Processing started."
    )


@router.get("/{video_id}/status", response_model=VideoStatusResponse)
def get_video_status(video_id: str):
    job = get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoStatusResponse(**job)


@router.get("/{video_id}/transcript")
def get_transcript(video_id: str):
    job = get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    if job["status"] != JobStatus.DONE:
        raise HTTPException(status_code=400, detail=f"Not ready. Status: {job['status']}")
    transcript_path = job.get("transcript_path")
    if not transcript_path or not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail="Transcript file not found")
    with open(transcript_path, "r", encoding="utf-8") as f:
        import json
        return json.load(f)
    
@router.get("/{video_id}/chunks")
def get_chunks(video_id: str):
    """Xem chunks sau khi meeting pipeline chạy xong"""
    job = get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    if job["status"] != JobStatus.DONE:
        raise HTTPException(status_code=400, detail=f"Not ready: {job['status']}")
    
    chunks_path = job.get("chunks_path")
    if not chunks_path or not os.path.exists(chunks_path):
        raise HTTPException(status_code=404, detail="Chunks not found")
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        import json
        chunks = json.load(f)
    
    return {
        "video_id": video_id,
        "num_chunks": len(chunks),
        "chunks": chunks
    }