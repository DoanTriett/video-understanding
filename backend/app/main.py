import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.qa import router as qa_router
from app.api.summary import router as summary_router
from app.api.videos import router as videos_router
from app.limiter import limiter
from app.middleware import RequestLoggingMiddleware
from app.observability import metrics as _metrics  # noqa: F401 — registers custom metrics
from app.observability.logging import configure_json_logging

os.environ["SB_DISABLE_K2"] = "1"  # disable speechbrain k2

configure_json_logging()

app = FastAPI(
    title="Video Understanding API",
    description="Meeting & Lecture video Q&A system",
    version="0.1.0",
)

# Attach slowapi limiter to app state and register the 429 handler.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS: Cho phép frontend (localhost:3000) gọi API này (localhost:8000).
# Must be added after SlowAPIMiddleware so CORS headers are present on 429 responses too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging — innermost middleware, sees final status codes.
app.add_middleware(RequestLoggingMiddleware)

app.include_router(videos_router)
app.include_router(qa_router)
app.include_router(summary_router)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def root():
    return {"message": "Video Understanding API is running"}


@app.get("/health")
def health_check():  # endpoint để frontend kiểm tra backend còn sống hay không
    return {"status": "ok"}
