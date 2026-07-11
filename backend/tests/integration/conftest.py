"""Fixtures and helpers for integration tests (real Postgres, Redis, Qdrant)."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sqlalchemy import text

from app.config import settings
from app.db.models import Chunk, Summary, Video
from app.db.session import SessionLocal
from app.semantic_cache import invalidate_cache
from app.store import delete_progress, redis_client


def _postgres_ok() -> bool:
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        db.close()


def _redis_ok() -> bool:
    try:
        return bool(redis_client.ping())
    except Exception:
        return False


def _qdrant_ok() -> bool:
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def require_integration_services():
    """Skip integration tests when docker-compose services are not reachable."""
    missing = []
    if not _postgres_ok():
        missing.append("Postgres")
    if not _redis_ok():
        missing.append("Redis")
    if not _qdrant_ok():
        missing.append("Qdrant")
    if missing:
        pytest.skip(f"Integration services unavailable: {', '.join(missing)}")


def delete_qdrant_points_for_video(video_id: str) -> None:
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    video_filter = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])
    for collection in ("chunks_text", "chunks_visual"):
        try:
            client.delete(collection_name=collection, points_selector=video_filter)
        except Exception:
            # Collection may not exist yet on a fresh Qdrant instance.
            pass


def delete_postgres_video_data(video_id: str) -> None:
    db = SessionLocal()
    try:
        db.query(Chunk).filter(Chunk.video_id == video_id).delete()
        db.query(Summary).filter(Summary.video_id == video_id).delete()
        db.query(Video).filter(Video.id == video_id).delete()
        db.commit()
    finally:
        db.close()


def teardown_video(video_id: str) -> None:
    """Remove test artifacts from Postgres, Qdrant, Redis semantic cache."""
    delete_postgres_video_data(video_id)
    delete_qdrant_points_for_video(video_id)
    delete_progress(video_id)
    invalidate_cache(video_id)


@pytest.fixture
def clean_video_after_test(require_integration_services):
    """Track video IDs created during a test and wipe them afterward."""
    created: list[str] = []
    yield created
    for video_id in created:
        teardown_video(video_id)
