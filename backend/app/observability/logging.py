"""app/observability/logging.py — Structured JSON logging via stdlib.

Replaces the default basicConfig plain-text formatter with a JSON formatter
so logs are easy to ship to Loki or any log aggregator later.

Each log line is a single JSON object:
    {"timestamp": "...", "level": "INFO", "logger": "app.requests", "message": "..."}

Usage:
    from app.observability.logging import configure_json_logging
    configure_json_logging()
"""

import json
import logging


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Replace all root-logger handlers with a single JSON stdout handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
