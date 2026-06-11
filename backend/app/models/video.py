from pydantic import BaseModel
from enum import Enum
from datetime import datetime

class VideoType(str, Enum):
    MEETING = "meeting"
    LECTURE = "lecture"
    UNKNOWN = "unknown"

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

class VideoUploadResponse(BaseModel):
    """Response khi upload video thành công"""
    video_id: str
    filename: str
    file_size_mb: float
    status: JobStatus
    message: str

class VideoStatusResponse(BaseModel):
    """Response khi hỏi status của một video"""
    video_id: str
    status: JobStatus
    video_type: VideoType
    progress_percent: int
    error_message: str | None = None
    created_at: datetime