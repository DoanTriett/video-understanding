from dataclasses import dataclass
from typing import List

import torch
from faster_whisper import WhisperModel


@dataclass
class WordTimestamp:
    word: str
    start: float  # giây
    end: float  # giây


@dataclass
class TranscriptSegment:
    """
    Một đoạn transcript liên tục.
    Whisper tự chia thành segments (thường ~30s mỗi đoạn).
    """

    text: str
    start: float
    end: float
    words: List[WordTimestamp]  # timestamp từng từ


@dataclass
class TranscriptResult:
    segments: List[TranscriptSegment]
    language: str
    duration: float


# Load model một lần, dùng lại nhiều lần
# "base" model: nhỏ, nhanh, đủ tốt cho demo
# "large-v3": tốt nhất nhưng cần ~10GB RAM
# Bắt đầu với "base", sau đổi sang "large-v3" khi deploy
_model = None


def get_model():
    global _model
    if _model is None:
        print("Loading Whisper model... (lần đầu sẽ download, ~150MB)")
        # Auto-detect device: use CUDA if available, otherwise CPU
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"  # Faster on GPU
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            compute_type = "float32"  # CPU doesn't support float16 well
            print("CUDA not available. Using CPU.")

        _model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    return _model


def transcribe(audio_path: str) -> TranscriptResult:
    """
    Transcribe audio file, trả về text với word-level timestamps.

    word_timestamps=True là quan trọng nhất —
    cho phép sau này biết chính xác từng từ xuất hiện lúc mấy giây.
    """
    model = get_model()

    segments_raw, info = model.transcribe(
        audio_path,
        word_timestamps=True,  # cần cho temporal citations
        vad_filter=True,  # bỏ qua đoạn silence (VAD = Voice Activity Detection)
        vad_parameters={
            "min_silence_duration_ms": 500  # silence > 500ms thì skip
        },
    )

    segments = []
    for seg in segments_raw:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    WordTimestamp(word=w.word.strip(), start=round(w.start, 2), end=round(w.end, 2))
                )

        segments.append(
            TranscriptSegment(
                text=seg.text.strip(), start=round(seg.start, 2), end=round(seg.end, 2), words=words
            )
        )

    return TranscriptResult(segments=segments, language=info.language, duration=info.duration)
