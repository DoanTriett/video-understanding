from dataclasses import dataclass
from typing import List

import torch
from pyannote.audio import Pipeline

from app.config import settings


@dataclass
class DiarizationSegment:
    """
    Một đoạn audio của một speaker cụ thể.
    speaker: "SPEAKER_00", "SPEAKER_01"...
    start/end: tính bằng giây
    """

    speaker: str
    start: float
    end: float


# Load model một lần, tái sử dụng
# Lần đầu chạy sẽ download model ~1GB về máy
_pipeline = None


def get_diarization_pipeline():
    global _pipeline
    if _pipeline is None:
        print("Loading pyannote diarization model (first run downloads ~1GB)...")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=settings.huggingface_token
        )
        # Use GPU if available, otherwise fallback to CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Diarization model loaded on: {device}")
        _pipeline.to(device)

    return _pipeline


def diarize(audio_path: str, num_speakers: int = None) -> List[DiarizationSegment]:
    """
    Chạy speaker diarization trên file audio.

    num_speakers: nếu biết trước số người, truyền vào để tăng accuracy.
    Nếu không biết, để None — model tự detect.

    Returns: list các đoạn audio, mỗi đoạn gắn với 1 speaker.
    """
    pipeline = get_diarization_pipeline()

    # Truyền num_speakers nếu có
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    # Chạy diarization
    # Output là pyannote Annotation object
    diarization = pipeline(audio_path, **kwargs)

    output = pipeline(audio_path)

    diarization = output.speaker_diarization
    segments = []

    for turn, speaker in diarization:
        segments.append(
            DiarizationSegment(
                speaker=speaker,  # "SPEAKER_00", "SPEAKER_01"...
                start=round(turn.start, 3),
                end=round(turn.end, 3),
            )
        )

    # Convert sang dataclass đơn giản hơn
    """for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(DiarizationSegment(
            speaker=speaker,          # "SPEAKER_00", "SPEAKER_01"...
            start=round(turn.start, 3),
            end=round(turn.end, 3)
        ))"""

    # Sort theo thời gian cho chắc
    segments.sort(key=lambda x: x.start)
    return segments
