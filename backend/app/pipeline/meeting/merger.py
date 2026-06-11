from dataclasses import dataclass, field
from typing import List
from app.pipeline.meeting.diarizer import DiarizationSegment

@dataclass
class WordWithSpeaker:
    word: str
    start: float
    end: float
    speaker: str

@dataclass
class SpeakerTurn:
    """
    Một lượt nói liên tục của một speaker.
    Đây là unit cơ bản để chunk sau này.
    """
    speaker: str
    text: str
    start: float
    end: float
    words: List[WordWithSpeaker] = field(default_factory=list)


def find_speaker_at_time(
    t: float,
    diarization: List[DiarizationSegment]
) -> str:
    """
    Tìm speaker đang nói tại thời điểm t.
    
    Dùng overlap logic: speaker nào có segment chứa thời điểm t
    thì là speaker đó. Nếu không tìm thấy → "UNKNOWN".
    """
    for seg in diarization:
        if seg.start <= t <= seg.end:
            return seg.speaker
    return "UNKNOWN"


def merge_transcript_with_diarization(
    whisper_segments: List[dict],
    diarization: List[DiarizationSegment]
) -> List[SpeakerTurn]:
    """
    Gộp Whisper words với Diarization segments.
    
    Thuật toán:
    1. Lấy từng word từ Whisper (có timestamp)
    2. Tìm speaker tại midpoint của word đó
    3. Gán speaker cho word
    4. Group các words liên tiếp cùng speaker → 1 SpeakerTurn
    """
    
    # Bước 1: Flatten tất cả words từ whisper segments
    all_words = []
    for seg in whisper_segments:
        for word in seg.get("words", []):
            all_words.append({
                "word": word["word"].strip(),
                "start": word["start"],
                "end": word["end"]
            })
    
    if not all_words:
        # Fallback: không có word-level timestamps
        # Dùng segment midpoint thay thế
        return _merge_by_segment(whisper_segments, diarization)
    
    # Bước 2: Gán speaker cho từng word
    words_with_speaker = []
    for w in all_words:
        # Dùng midpoint của word để tìm speaker
        # Vì đôi khi word nằm ở boundary 2 speaker turns
        midpoint = (w["start"] + w["end"]) / 2
        speaker = find_speaker_at_time(midpoint, diarization)
        words_with_speaker.append(WordWithSpeaker(
            word=w["word"],
            start=w["start"],
            end=w["end"],
            speaker=speaker
        ))
    
    # Bước 3: Group words liên tiếp cùng speaker
    turns = []
    if not words_with_speaker:
        return turns
    
    current_speaker = words_with_speaker[0].speaker
    current_words = [words_with_speaker[0]]
    
    for word in words_with_speaker[1:]:
        if word.speaker == current_speaker:
            # Cùng speaker → tiếp tục group
            current_words.append(word)
        else:
            # Đổi speaker → kết thúc turn hiện tại
            turns.append(_words_to_turn(current_speaker, current_words))
            current_speaker = word.speaker
            current_words = [word]
    
    # Đừng quên turn cuối cùng
    if current_words:
        turns.append(_words_to_turn(current_speaker, current_words))
    
    return turns


def _words_to_turn(speaker: str, words: List[WordWithSpeaker]) -> SpeakerTurn:
    """Convert list of words thành SpeakerTurn"""
    text = " ".join(w.word for w in words).strip()
    return SpeakerTurn(
        speaker=speaker,
        text=text,
        start=words[0].start,
        end=words[-1].end,
        words=words
    )


def _merge_by_segment(whisper_segments, diarization) -> List[SpeakerTurn]:
    """Fallback khi không có word-level timestamps"""
    turns = []
    for seg in whisper_segments:
        midpoint = (seg["start"] + seg["end"]) / 2
        speaker = find_speaker_at_time(midpoint, diarization)
        turns.append(SpeakerTurn(
            speaker=speaker,
            text=seg["text"].strip(),
            start=seg["start"],
            end=seg["end"]
        ))
    return turns