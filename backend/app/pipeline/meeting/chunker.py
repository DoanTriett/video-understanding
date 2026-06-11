from dataclasses import dataclass, field
from typing import List
from app.pipeline.meeting.merger import SpeakerTurn

@dataclass
class MeetingChunk:
    """
    Đơn vị sẽ được embed và lưu vào vector DB.
    
    Mỗi chunk là một đoạn hội thoại có nghĩa:
    - Không quá ngắn (< 5s: thiếu context)
    - Không quá dài (> 45s: quá nhiều thông tin, khó retrieve)
    """
    chunk_id: str
    video_id: str
    speaker: str
    text: str
    start: float
    end: float
    turn_index: int          # vị trí trong cuộc họp
    chunk_type: str = "speech"   # "speech" hoặc "screen_share"

MAX_CHUNK_DURATION = 45.0   # giây
MIN_TURN_DURATION = 5.0     # giây — turn ngắn hơn sẽ merge với turn kế tiếp


def chunk_meeting(
    turns: List[SpeakerTurn],
    video_id: str
) -> List[MeetingChunk]:
    """
    Chuyển speaker turns thành chunks phù hợp để embed.
    
    Logic:
    1. Merge các turns quá ngắn với turn liền kề
    2. Split các turns quá dài thành nhiều chunks
    """
    
    # Bước 1: Merge short turns
    merged_turns = _merge_short_turns(turns)
    
    # Bước 2: Split long turns + tạo chunks
    chunks = []
    chunk_index = 0
    
    for turn_idx, turn in enumerate(merged_turns):
        duration = turn.end - turn.start
        
        if duration <= MAX_CHUNK_DURATION:
            # Turn đủ ngắn → 1 chunk
            chunks.append(MeetingChunk(
                chunk_id=f"{video_id}_chunk_{chunk_index:04d}",
                video_id=video_id,
                speaker=turn.speaker,
                text=turn.text,
                start=turn.start,
                end=turn.end,
                turn_index=turn_idx
            ))
            chunk_index += 1
        else:
            # Turn quá dài → split theo câu/thời gian
            sub_chunks = _split_long_turn(turn, video_id, chunk_index, turn_idx)
            chunks.extend(sub_chunks)
            chunk_index += len(sub_chunks)
    
    return chunks


def _merge_short_turns(turns: List[SpeakerTurn]) -> List[SpeakerTurn]:
    """
    Merge các turns quá ngắn.
    
    Ví dụ: "Yes." (1s) + "I agree with that." (3s) → merge thành 1 turn.
    Chỉ merge nếu cùng speaker hoặc turn quá ngắn để đứng một mình.
    """
    if not turns:
        return turns
    
    merged = [turns[0]]
    
    for turn in turns[1:]:
        prev = merged[-1]
        prev_duration = prev.end - prev.start
        
        should_merge = (
            prev_duration < MIN_TURN_DURATION or   # turn trước quá ngắn
            (turn.speaker == prev.speaker and       # cùng speaker
             turn.start - prev.end < 2.0)           # và nói liên tiếp (gap < 2s)
        )
        
        if should_merge:
            # Merge vào turn trước
            merged[-1] = SpeakerTurn(
                speaker=prev.speaker,  # giữ speaker của turn trước
                text=prev.text + " " + turn.text,
                start=prev.start,
                end=turn.end,
                words=prev.words + turn.words
            )
        else:
            merged.append(turn)
    
    return merged


def _split_long_turn(
    turn: SpeakerTurn,
    video_id: str,
    start_index: int,
    turn_idx: int
) -> List[MeetingChunk]:
    """
    Split turn dài thành nhiều chunks ~30s mỗi chunk.
    Split tại boundary giữa các words để không cắt giữa câu.
    """
    chunks = []
    chunk_index = start_index
    
    if not turn.words:
        # Không có word timestamps → split đều theo thời gian
        duration = turn.end - turn.start
        n_chunks = int(duration / MAX_CHUNK_DURATION) + 1
        chunk_dur = duration / n_chunks
        
        for i in range(n_chunks):
            chunk_start = turn.start + i * chunk_dur
            chunk_end = min(turn.start + (i + 1) * chunk_dur, turn.end)
            chunks.append(MeetingChunk(
                chunk_id=f"{video_id}_chunk_{chunk_index:04d}",
                video_id=video_id,
                speaker=turn.speaker,
                text=turn.text,  # text đầy đủ, không split được nếu không có words
                start=chunk_start,
                end=chunk_end,
                turn_index=turn_idx
            ))
            chunk_index += 1
        return chunks
    
    # Có word timestamps → split tại word boundaries
    current_words = []
    current_start = turn.words[0].start
    
    for word in turn.words:
        current_words.append(word)
        current_duration = word.end - current_start
        
        if current_duration >= MAX_CHUNK_DURATION:
            text = " ".join(w.word for w in current_words).strip()
            chunks.append(MeetingChunk(
                chunk_id=f"{video_id}_chunk_{chunk_index:04d}",
                video_id=video_id,
                speaker=turn.speaker,
                text=text,
                start=current_start,
                end=word.end,
                turn_index=turn_idx
            ))
            chunk_index += 1
            current_words = []
            current_start = word.end
    
    # Words còn lại chưa đủ MAX_CHUNK_DURATION
    if current_words:
        text = " ".join(w.word for w in current_words).strip()
        chunks.append(MeetingChunk(
            chunk_id=f"{video_id}_chunk_{chunk_index:04d}",
            video_id=video_id,
            speaker=turn.speaker,
            text=text,
            start=current_start,
            end=current_words[-1].end,
            turn_index=turn_idx
        ))
    
    return chunks