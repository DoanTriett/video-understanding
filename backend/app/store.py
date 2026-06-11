import json
import redis
from app.config import settings

# Redis client dùng chung
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

JOB_TTL = 60 * 60 * 24  # giữ job data 24 giờ

def set_job(video_id: str, data: dict):
    """Lưu job vào Redis"""
    # Convert datetime sang string nếu có
    serializable = {}
    for k, v in data.items():
        if hasattr(v, 'isoformat'):  # datetime object
            serializable[k] = v.isoformat()
        else:
            serializable[k] = v
    redis_client.setex(f"job:{video_id}", JOB_TTL, json.dumps(serializable))

def get_job(video_id: str) -> dict | None:
    """Lấy job từ Redis"""
    raw = redis_client.get(f"job:{video_id}")
    if not raw:
        return None
    return json.loads(raw)

def update_job(video_id: str, **kwargs):
    """Update một vài fields của job"""
    job = get_job(video_id)
    if not job:
        raise KeyError(f"Job {video_id} not found")
    job.update(kwargs)
    redis_client.setex(f"job:{video_id}", JOB_TTL, json.dumps(job))