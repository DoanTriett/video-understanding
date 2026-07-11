"""app/middleware.py — Structured request logging middleware.

Logs every request with:
    request_id  short uuid for correlating logs within one request
    method      HTTP verb
    path        URL path
    video_id    path parameter if present, else "-"
    status      HTTP response status code
    latency_ms  total time from first byte received to response sent

Uses stdlib logging — no structlog/JSON format yet (deferred to Phase 5.2).
Configure log level via LOG_LEVEL env var or Python logging config.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("app.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000
        # video_id is populated by FastAPI path param matching.
        video_id = request.path_params.get("video_id", "-")

        logger.info(
            "request_id=%s method=%s path=%s video_id=%s status=%d latency_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            video_id,
            response.status_code,
            latency_ms,
        )
        return response
