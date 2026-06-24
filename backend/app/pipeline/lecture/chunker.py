from dataclasses import dataclass


@dataclass
class LectureChunk:
    chunk_id: str
    video_id: str
    slide_index: int
    start: float
    end: float
    transcript_text: str  # text từ Whisper trong khoảng thời gian này
    ocr_text: str  # text OCR từ slide image
    combined_text: str  # transcript + ocr, dùng để embed
    frame_path: str | None
    clip_embedding: list[float] | None


def build_lecture_chunks(
    video_id: str,
    transcript_segments: list[dict],  # từ Whisper merge với diarization
    slide_changes: list,  # từ slide_detector
    ocr_results: list[dict],  # từ ocr.py
    clip_embeddings: list[list[float]] | None = None,
) -> list[LectureChunk]:
    chunks = []

    for i, slide in enumerate(slide_changes):
        slide_start = slide.timestamp
        slide_end = slide_changes[i + 1].timestamp if i + 1 < len(slide_changes) else float("inf")

        # Lấy transcript segments nằm trong khoảng [slide_start, slide_end)
        slide_segments = [
            s for s in transcript_segments if s["end"] > slide_start and s["start"] < slide_end
        ]
        transcript_text = " ".join(s["text"] for s in slide_segments).strip()
        ocr_text = ocr_results[i]["ocr_text"] if i < len(ocr_results) else ""

        # Combined text: OCR đứng trước (heading/title trên slide) + transcript
        combined_text = f"[SLIDE] {ocr_text}\n[SPEECH] {transcript_text}".strip()

        chunk = LectureChunk(
            chunk_id=f"{video_id}_slide_{i:04d}",
            video_id=video_id,
            slide_index=i,
            start=slide_start,
            end=min(slide_end, slide_segments[-1]["end"]) if slide_segments else slide_start + 60,
            transcript_text=transcript_text,
            ocr_text=ocr_text,
            combined_text=combined_text,
            frame_path=slide.frame_path,
            clip_embedding=clip_embeddings[i] if clip_embeddings else None,
        )
        chunks.append(chunk)

    return chunks
