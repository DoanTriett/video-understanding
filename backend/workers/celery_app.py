from celery import Celery
from app.config import settings

# Tạo Celery app
# broker = nơi nhận jobs (Redis)
# backend = nơi lưu kết quả (cũng Redis)
celery_app = Celery(
    "video_understanding",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks"]  # file chứa các tasks
)

celery_app.conf.update(
    task_serializer="json",     # Định dạng gửi tasks
    accept_content=["json"],    # Chỉ chấp nhận JSON
    result_serializer="json",   # Định dạng kết quả
    timezone="UTC",
    enable_utc=True,
    # Retry failed tasks
    task_acks_late=True,        # Ngăn mất job nếu worker crash giữa chừng
    task_reject_on_worker_lost=True,  # Quay lại job nếu worker crash
)