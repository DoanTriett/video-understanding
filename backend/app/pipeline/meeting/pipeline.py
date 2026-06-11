from dataclasses import dataclass
from typing import List
from app.pipeline.meeting.diarizer import diarize
from app.pipeline.meeting.merger import merge_transcript_with_diarization
from app.pipeline.meeting.chunker import chunk_meeting, MeetingChunk
from app.pipeline.meeting.screen_detector import detect_screen_share
from app.pipeline.shared.visual_embedder import embed_image
import json, os

@dataclass
class MeetingChunkWithEmbedding:
    chunk: MeetingChunk
    visual_embedding: List[float] = None  # chỉ có nếu là screen share chunk

def run_meeting_pipeline(
    video_id: str,
    video_path: str,
    audio_path: str,
    transcript_path: str,
    work_dir: str,
    update_progress_fn=None  # callback để update Redis
) -> List[MeetingChunkWithEmbedding]:
    """
    Chạy toàn bộ meeting pipeline.
    Nhận video + transcript từ Tuần 1, trả về enriched chunks.
    """
    
    def progress(pct, msg):
        print(f"[{video_id}] {pct}% — {msg}")
        if update_progress_fn:
            update_progress_fn(pct)
    
    # Load transcript từ file JSON (output của Tuần 1)
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    whisper_segments = transcript_data["segments"]
    
    # ── Bước 1: Diarization ──────────────────────────────
    progress(40, "Running speaker diarization...")
    diarization_segments = diarize(audio_path)
    print(f"  Found {len(set(s.speaker for s in diarization_segments))} speakers")
    
    # ── Bước 2: Merge transcript + diarization ───────────
    progress(55, "Merging transcript with diarization...")
    speaker_turns = merge_transcript_with_diarization(
        whisper_segments, diarization_segments
    )
    print(f"  Created {len(speaker_turns)} speaker turns")
    
    # ── Bước 3: Chunk ────────────────────────────────────
    progress(65, "Chunking by speaker turns...")
    chunks = chunk_meeting(speaker_turns, video_id)
    print(f"  Created {len(chunks)} chunks")
    
    # ── Bước 4: Screen share detection ──────────────────
    progress(75, "Detecting screen share segments...")
    screen_dir = os.path.join(work_dir, "screen_frames")
    screen_segments = detect_screen_share(video_path, screen_dir)
    print(f"  Found {len(screen_segments)} screen share segments")
    
    # ── Bước 5: Embed screen share frames ────────────────
    progress(85, "Embedding screen share frames...")
    result_chunks = []
    
    for chunk in chunks:
        visual_emb = None
        
        # Kiểm tra chunk này có overlap với screen share không
        for screen_seg in screen_segments:
            overlap = (
                chunk.start < screen_seg.end and
                chunk.end > screen_seg.start
            )
            if overlap:
                visual_emb = embed_image(screen_seg.frame_path)
                chunk.chunk_type = "screen_share"
                break
        
        result_chunks.append(MeetingChunkWithEmbedding(
            chunk=chunk,
            visual_embedding=visual_emb
        ))
    
    # Lưu chunks ra file để debug
    chunks_path = os.path.join(work_dir, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump([{
            "chunk_id": c.chunk.chunk_id,
            "speaker": c.chunk.speaker,
            "text": c.chunk.text,
            "start": c.chunk.start,
            "end": c.chunk.end,
            "turn_index": c.chunk.turn_index,
            "chunk_type": c.chunk.chunk_type,
            "has_visual": c.visual_embedding is not None
        } for c in result_chunks], f, ensure_ascii=False, indent=2)
    
    progress(95, f"Meeting pipeline done. {len(result_chunks)} chunks ready.")
    return result_chunks