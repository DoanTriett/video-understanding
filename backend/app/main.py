import os
from contextlib import asynccontextmanager

# ruff: noqa: E402
# Set PROMETHEUS_MULTIPROC_DIR before any prometheus_client import.
# Both API and worker must point to the same directory.
# Override via env var; default is <backend_root>/prometheus_multiproc.
_PROM_DIR = os.path.abspath(
    os.environ.setdefault(
        "PROMETHEUS_MULTIPROC_DIR",
        os.path.join(os.path.dirname(__file__), "..", "prometheus_multiproc"),
    )
)
os.makedirs(_PROM_DIR, exist_ok=True)


def _clean_prometheus_multiproc_dir() -> None:
    """Delete stale .db files from previous runs to avoid metric accumulation."""
    for fname in os.listdir(_PROM_DIR):
        if fname.endswith(".db"):
            try:
                os.remove(os.path.join(_PROM_DIR, fname))
            except OSError:
                pass


@asynccontextmanager
async def lifespan(app):
    _clean_prometheus_multiproc_dir()
    yield


# Now safe to import prometheus-aware modules.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, multiprocess
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.qa import router as qa_router
from app.api.summary import router as summary_router
from app.api.videos import router as videos_router
from app.config import settings
from app.limiter import limiter
from app.middleware import RequestLoggingMiddleware
from app.observability import metrics as _metrics  # noqa: F401 - registers custom metrics
from app.observability.logging import configure_json_logging

os.environ["SB_DISABLE_K2"] = "1"  # disable speechbrain k2

configure_json_logging()

cors_allow_origins = [
    origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
]

app = FastAPI(
    title="Video Understanding API",
    description="Meeting & Lecture video Q&A system",
    version="0.1.0",
    lifespan=lifespan,
)

# Attach slowapi limiter to app state and register the 429 handler.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS for configured frontend origins.
# Must be added after SlowAPIMiddleware so CORS headers are present on 429 responses too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging is innermost middleware, so it sees final status codes.
app.add_middleware(RequestLoggingMiddleware)

app.include_router(videos_router)
app.include_router(qa_router)
app.include_router(summary_router)

Instrumentator().instrument(app)  # track HTTP metrics; /metrics served below


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint() -> Response:
    """Aggregate metrics from all live processes via multiprocess collector."""
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    return {"message": "Video Understanding API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
