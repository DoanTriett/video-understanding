import os

from celery import Celery
from celery.signals import celeryd_init, setup_logging

# ── Set PROMETHEUS_MULTIPROC_DIR before tasks.py imports prometheus_client ────
# Must match the same directory set in main.py / OS env.
os.makedirs(
    os.environ.setdefault(
        "PROMETHEUS_MULTIPROC_DIR",
        os.path.join(os.path.dirname(__file__), "..", "prometheus_multiproc"),
    ),
    exist_ok=True,
)

from app.config import settings  # noqa: E402


@setup_logging.connect
def on_setup_logging(**kwargs) -> bool:
    """Override Celery's default logging with JSON formatter.

    Returning a truthy value prevents Celery from configuring its own handlers,
    so our JSON formatter takes full effect.
    """
    from app.observability.logging import configure_json_logging

    configure_json_logging()
    return True


@celeryd_init.connect
def on_celeryd_init(**kwargs) -> None:
    """Run once when the Celery worker daemon starts (before any tasks)."""
    # Clean stale prometheus multiproc files from previous worker runs.
    prom_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "")
    if prom_dir and os.path.isdir(prom_dir):
        for fname in os.listdir(prom_dir):
            if fname.endswith(".db"):
                try:
                    os.remove(os.path.join(prom_dir, fname))
                except OSError:
                    pass


# Hard kill sau 30 phút — đủ dư cho video dài thật, chặn decode-loop vô tận
TASK_HARD_TIMEOUT_SECONDS = 1800
# Soft signal trước hard kill 90s — cho task cơ hội cleanup / ghi status failed
TASK_SOFT_TIMEOUT_SECONDS = TASK_HARD_TIMEOUT_SECONDS - 90

# Tạo Celery app
# broker = nơi nhận jobs (Redis)
# backend = nơi lưu kết quả (cũng Redis)
celery_app = Celery(
    "video_understanding",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks"],  # file chứa các tasks
)

celery_app.conf.update(
    task_serializer="json",  # Định dạng gửi tasks
    accept_content=["json"],  # Chỉ chấp nhận JSON
    result_serializer="json",  # Định dạng kết quả
    timezone="UTC",
    enable_utc=True,
    # Retry failed tasks
    task_acks_late=True,  # Ngăn mất job nếu worker crash giữa chừng
    task_reject_on_worker_lost=True,  # Quay lại job nếu worker crash
    # Timeout lưới an toàn — ngăn task treo vô thời hạn (e.g. Whisper decode-loop)
    task_time_limit=TASK_HARD_TIMEOUT_SECONDS,
    task_soft_time_limit=TASK_SOFT_TIMEOUT_SECONDS,
)
