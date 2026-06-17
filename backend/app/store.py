import json

import redis

from app.config import settings

redis_client = redis.from_url(settings.redis_url, decode_responses=True)

PROGRESS_TTL = 60 * 60 * 24  # 24 giờ — tự xóa nếu worker chết không clean up được

# ─────────────────────────────────────────
# PHẦN 1: Redis — chỉ dùng cho live progress
# ─────────────────────────────────────────


def set_progress(video_id: str, stage: str, pct: int):
    """Celery task gọi cái này liên tục để update tiến độ."""
    redis_client.setex(
        f"progress:{video_id}", PROGRESS_TTL, json.dumps({"stage": stage, "pct": pct})
    )


def get_progress(video_id: str) -> dict | None:
    """Trả về None nếu không có key — nghĩa là task không đang chạy."""
    raw = redis_client.get(f"progress:{video_id}")
    if not raw:
        return None
    return json.loads(raw)


def delete_progress(video_id: str):
    """Gọi khi task done hoặc failed — dọn dẹp key Redis."""
    redis_client.delete(f"progress:{video_id}")
