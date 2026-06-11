from workers.celery_app import celery_app
from app.store import get_job, update_job
from app.models.video import JobStatus, VideoType
from app.pipeline.shared.audio_extractor import extract_audio
from app.pipeline.shared.transcriber import transcribe
from app.pipeline.meeting.pipeline import run_meeting_pipeline
from app.config import settings
import os, json

@celery_app.task(bind=True, max_retries=3)
def process_video(self, video_id: str):
    try:
        job = get_job(video_id)
        if not job:
            raise ValueError(f"Job {video_id} not found")

        work_dir = os.path.join(settings.upload_dir, video_id)
        os.makedirs(work_dir, exist_ok=True)

        # ── Bước 1: Extract audio ──────────────────
        update_job(video_id, status=JobStatus.PROCESSING, progress_percent=10)
        print(f"[{video_id}] Extracting audio...")
        audio_path = extract_audio(job["file_path"], work_dir)
        update_job(video_id, progress_percent=25)

        # ── Bước 2: Transcribe ─────────────────────
        print(f"[{video_id}] Transcribing...")
        update_job(video_id, progress_percent=30)
        result = transcribe(audio_path)
        transcript_path = os.path.join(work_dir, "transcript.json")
        transcript_data = {
            "language": result.language,
            "duration": result.duration,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                    "words": [{"word": w.word, "start": w.start, "end": w.end}
                              for w in seg.words]
                }
                for seg in result.segments
            ]
        }
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        update_job(video_id, progress_percent=40, transcript_path=transcript_path)

        # ── Bước 3: Meeting Pipeline ────────────────
        video_type = job.get("video_type", VideoType.UNKNOWN)
        if video_type in (VideoType.MEETING, VideoType.UNKNOWN):
            print(f"[{video_id}] Running meeting pipeline...")

            def update_progress(pct):
                update_job(video_id, progress_percent=40 + int(pct * 0.55))

            chunks = run_meeting_pipeline(
                video_id=video_id,
                video_path=job["file_path"],
                audio_path=audio_path,
                transcript_path=transcript_path,
                work_dir=work_dir,
                update_progress_fn=update_progress
            )
            update_job(video_id,
                chunks_path=os.path.join(work_dir, "chunks.json"),
                num_chunks=len(chunks)
            )

        # ── Xóa error_message cũ khi thành công ──
        update_job(video_id,
            status=JobStatus.DONE,
            progress_percent=100,
            error_message=None      # ← quan trọng
        )
        print(f"[{video_id}] All done!")

    except ImportError as exc:
        # k2_fsa và speechbrain lazy import lỗi trên Windows
        # Không set FAILED vì retry lần 2 sẽ tự chạy được
        if "k2_fsa" in str(exc) or "LazyModule" in str(exc):
            print(f"[{video_id}] Known import warning (k2_fsa), retrying...")
            raise self.retry(exc=exc, countdown=60)
        # ImportError khác → xử lý như lỗi thật
        if get_job(video_id):
            update_job(video_id, status=JobStatus.FAILED, error_message=str(exc))
        raise

    except Exception as exc:
        if get_job(video_id):
            update_job(video_id, status=JobStatus.FAILED, error_message=str(exc))
        raise self.retry(exc=exc, countdown=60)