"""app/observability/metrics.py — Custom Prometheus metrics.

Imported at app startup (main.py) so metrics are registered with the default
prometheus_client registry before the first /metrics scrape.

In Celery workers (separate process), these metrics update the worker-local
registry only. Cross-process aggregation requires PROMETHEUS_MULTIPROC_DIR,
which is out of scope for this phase.
"""

from prometheus_client import Counter, Gauge, Histogram

jobs_in_progress = Gauge(
    "jobs_in_progress",
    "Number of video processing jobs currently running",
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
