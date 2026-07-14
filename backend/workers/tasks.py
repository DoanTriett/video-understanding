import json
import os
import tempfile
import time

from app.db import crud  # ← Postgres operations
from app.db.session import SessionLocal  # ← Postgres session
from app.models.video import VideoType
from app.observability.metrics import (
    job_failures_total,
    jobs_in_progress,
    pipeline_stage_seconds,
)
from app.pipeline.lecture.pipeline import run_lecture_pipeline
from app.pipeline.meeting.pipeline import run_meeting_pipeline
from app.pipeline.shared.audio_extractor import extract_audio
from app.pipeline.shared.indexer import index_chunks
from app.pipeline.shared.transcriber import transcribe
from app.storage import download_to_path, upload_file
from app.store import delete_progress, set_progress  # ← chỉ dùng progress
from workers.celery_app import celery_app


@celery_app.task(bind=True, max_retries=3)
def process_video(self, video_id: str, video_type: str):
    # Mở DB session — Celery worker là process riêng, cần session riêng
    db = SessionLocal()
    jobs_in_progress.inc()

    try:
        # Mỗi task download source từ MinIO về temp dir, cuối task upload artifacts lên MinIO.
        with tempfile.TemporaryDirectory() as work_dir:
            video_path = os.path.join(work_dir, "source.mp4")
            object_key = crud.get_video_object_key(db, video_id)  # Lấy object_key từ Postgres
            if not object_key:
                raise ValueError(f"Video {video_id} has no MinIO object key")
            download_to_path(object_key, video_path)

            # ── Bước 1: Extract audio ──────────────────
            set_progress(video_id, stage="extracting_audio", pct=10)
            print(f"[{video_id}] Extracting audio...")
            _t = time.perf_counter()
            audio_path = extract_audio(video_path, work_dir)
            pipeline_stage_seconds.labels(stage="extract_audio").observe(time.perf_counter() - _t)

            # ── Bước 2: Transcribe ─────────────────────
            set_progress(video_id, stage="transcribing", pct=25)
            print(f"[{video_id}] Transcribing...")
            _t = time.perf_counter()
            result = transcribe(audio_path)
            pipeline_stage_seconds.labels(stage="transcribe").observe(time.perf_counter() - _t)

            transcript_path = os.path.join(work_dir, "transcript.json")
            transcript_data = {
                "language": result.language,
                "duration": result.duration,
                "segments": [
                    {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                        "words": [
                            {"word": w.word, "start": w.start, "end": w.end} for w in seg.words
                        ],
                    }
                    for seg in result.segments
                ],
            }
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)

            # ── Bước 3: Pipeline theo video type ───────
            set_progress(video_id, stage="running_pipeline", pct=40)
            _t = time.perf_counter()

            if video_type == VideoType.MEETING:
                print(f"[{video_id}] Running meeting pipeline...")

                def update_progress(pct):
                    set_progress(video_id, stage="meeting_pipeline", pct=40 + int(pct * 0.55))

                chunks = run_meeting_pipeline(
                    video_id=video_id,
                    video_path=video_path,
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    work_dir=work_dir,
                    update_progress_fn=update_progress,
                )
                pipeline_stage_seconds.labels(stage="meeting_pipeline").observe(
                    time.perf_counter() - _t
                )
            elif video_type == VideoType.LECTURE:
                print(f"[{video_id}] Running lecture pipeline...")

                def update_progress(pct):
                    set_progress(video_id, stage="lecture_pipeline", pct=40 + int(pct * 0.55))

                chunks = run_lecture_pipeline(
                    video_id=video_id,
                    video_path=video_path,
                    transcript_path=transcript_path,
                    work_dir=work_dir,
                    update_progress_fn=update_progress,
                )
                pipeline_stage_seconds.labels(stage="lecture_pipeline").observe(
                    time.perf_counter() - _t
                )
            else:
                raise ValueError(f"Unsupported video_type: {video_type}")

            # ── Bước 4: Lưu + index chunks (dùng chung cho mọi pipeline) ──
            # ← Lưu chunks vào Postgres (thay vì chỉ lưu path file)
            set_progress(video_id, stage="saving_to_db", pct=95)
            _t = time.perf_counter()
            crud.save_chunks(db, video_id, chunks)
            pipeline_stage_seconds.labels(stage="save_chunks").observe(time.perf_counter() - _t)

            # ← Index chunks vào Qdrant để phục vụ retrieval/QA
            set_progress(video_id, stage="indexing_qdrant", pct=97)
            _t = time.perf_counter()
            index_chunks(video_id, chunks)
            pipeline_stage_seconds.labels(stage="index_qdrant").observe(time.perf_counter() - _t)

            # upload transcript json back
            upload_file(transcript_path, f"{video_id}/transcript.json")

        # ── Done: ghi Postgres, xóa Redis ──────────
        set_progress(video_id, stage="done", pct=100)
        crud.update_video_status(db, video_id, status="done", progress=100)
        delete_progress(video_id)  # ← xóa Redis key, không cần nữa
        print(f"[{video_id}] All done!")

    except ImportError as exc:
        if "k2_fsa" in str(exc) or "LazyModule" in str(exc):
            print(f"[{video_id}] Known import warning (k2_fsa), retrying...")
            db.close()
            raise self.retry(exc=exc, countdown=60)
        # ImportError khác → failed thật
        job_failures_total.inc()
        crud.update_video_status(db, video_id, status="failed", error=str(exc))
        delete_progress(video_id)
        db.close()
        raise

    except Exception as exc:
        job_failures_total.inc()
        crud.update_video_status(db, video_id, status="failed", error=str(exc))
        delete_progress(video_id)
        db.close()
        raise self.retry(exc=exc, countdown=60)

    finally:
        jobs_in_progress.dec()
        db.close()
