import json
import os

from app.pipeline.lecture.chunker import build_lecture_chunks
from app.pipeline.lecture.ocr import extract_slide_texts
from app.pipeline.lecture.slide_detector import detect_slide_changes
from app.pipeline.shared.visual_embedder import embed_image


def run_lecture_pipeline(
    video_id: str,
    video_path: str,
    transcript_path: str,
    work_dir: str,
    update_progress_fn=None,  # callback để update Redis
) -> list[dict]:
    """Chạy toàn bộ lecture pipeline.

    Nhận video + transcript đã có (output của bước transcribe), trả về list[dict]
    với cùng format chunks như meeting pipeline (speaker, text, start, end, chunk_type)
    để tasks.py xử lý đồng nhất. Các key phụ (chunk_id, slide_index, clip_embedding)
    được thêm để indexer index visual + slide metadata.

    Hàm này KHÔNG gọi save_chunks/index_chunks — đó là việc của tasks.py.
    """

    def progress(pct, msg):
        print(f"[{video_id}] {pct}% — {msg}")
        if update_progress_fn:
            update_progress_fn(pct)

    # Load transcript từ file JSON (output của bước transcribe).
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    transcript_segments = transcript_data["segments"]

    # ── Bước 1: Detect slide changes ─────────────────────
    progress(45, "Detecting slide changes...")
    slide_dir = os.path.join(work_dir, "slide_frames")
    os.makedirs(slide_dir, exist_ok=True)
    slide_changes = detect_slide_changes(video_path, slide_dir)
    print(f"  Found {len(slide_changes)} slide changes")

    # ── Bước 2: OCR slide frames ─────────────────────────
    progress(65, "Running OCR on slide frames...")
    ocr_results = extract_slide_texts(slide_changes)

    # ── Bước 3: CLIP embed slide frames ──────────────────
    progress(80, "Embedding slide frames with CLIP...")
    clip_embeddings = [embed_image(slide.frame_path) for slide in slide_changes]

    # ── Bước 4: Build lecture chunks ─────────────────────
    progress(90, "Building lecture chunks...")
    chunks = build_lecture_chunks(
        video_id=video_id,
        transcript_segments=transcript_segments,
        slide_changes=slide_changes,
        ocr_results=ocr_results,
        clip_embeddings=clip_embeddings,
    )
    print(f"  Created {len(chunks)} lecture chunks")

    result_chunks = [
        {
            "chunk_id": c.chunk_id,
            "speaker": None,
            "text": c.combined_text,
            "start": c.start,
            "end": c.end,
            "chunk_type": "lecture_slide",
            "slide_index": c.slide_index,
            "clip_embedding": c.clip_embedding,
        }
        for c in chunks
    ]

    progress(95, f"Lecture pipeline done. {len(result_chunks)} chunks ready.")
    return result_chunks
