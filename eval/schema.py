"""eval/schema.py — Pydantic models for evaluation dataset annotation files.

One AnnotationFile per video, stored at eval/datasets/{video_id}.json.

QAPairs can be written manually by a human annotator, or auto-exported from the
Redis semantic cache by eval/export_from_cache.py.

Field alignment with backend:
  GroundTruthTimestamp.start / .end  →  Chunk.start / Chunk.end  (Float in Postgres)
  AnnotationFile.video_type          →  Video.video_type  ("meeting" | "lecture")
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class GroundTruthTimestamp(BaseModel):
    """A time range in the video that supports the expected answer.

    start/end are seconds (float), matching Chunk.start and Chunk.end in the
    Postgres chunks table.
    """

    start: float
    end: float


class QAPair(BaseModel):
    """One question/answer pair with supporting time-range evidence."""

    question: str
    expected_answer: str
    ground_truth_timestamps: list[GroundTruthTimestamp] = []
    notes: Optional[str] = None
    verified: bool = False


class AnnotationFile(BaseModel):
    """Top-level container for one video's annotation file."""

    # extra="ignore" lets the loader tolerate extra fields (e.g. _comment) without
    # raising an error, while still validating all declared fields strictly.
    model_config = ConfigDict(extra="ignore")

    video_id: str
    video_type: Optional[Literal["meeting", "lecture"]] = None
    qa_pairs: list[QAPair]
