"""app/observability/metrics.py — Custom Prometheus metrics.

Both the API process and the Celery worker process import this module.
Cross-process aggregation is handled via prometheus_client multiprocess mode:
PROMETHEUS_MULTIPROC_DIR must be set (and the same dir) in both processes
before this module is imported.
"""

from prometheus_client import Counter, Gauge, Histogram

jobs_in_progress = Gauge(
    "jobs_in_progress",
    "Number of video processing jobs currently running",
    multiprocess_mode="livesum",  # sum across live processes; correct for --pool=solo
)

job_failures_total = Counter(
    "job_failures_total",
    "Total number of failed video processing jobs",
)

pipeline_stage_seconds = Histogram(
    "pipeline_stage_seconds",
    "Duration of each pipeline stage in seconds",
    labelnames=["stage"],
)
