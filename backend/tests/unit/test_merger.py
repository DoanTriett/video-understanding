"""Unit tests cho merge_transcript_with_diarization và find_speaker_at_time."""

from app.pipeline.meeting.diarizer import DiarizationSegment
from app.pipeline.meeting.merger import (
    find_speaker_at_time,
    merge_transcript_with_diarization,
)


def _diarization(*segments: tuple[str, float, float]) -> list[DiarizationSegment]:
    return [DiarizationSegment(speaker=s, start=start, end=end) for s, start, end in segments]


def _whisper_with_words(
    segments: list[tuple[str, float, float, list[tuple[str, float, float]]]],
) -> list[dict]:
    return [
        {
            "text": text,
            "start": start,
            "end": end,
            "words": [{"word": w, "start": ws, "end": we} for w, ws, we in words],
        }
        for text, start, end, words in segments
    ]


def test_find_speaker_at_time_normal():
    diar = _diarization(("SPEAKER_00", 0.0, 10.0), ("SPEAKER_01", 10.0, 20.0))
    assert find_speaker_at_time(5.0, diar) == "SPEAKER_00"
    assert find_speaker_at_time(15.0, diar) == "SPEAKER_01"


def test_find_speaker_at_time_inclusive_boundaries():
    diar = _diarization(("SPEAKER_00", 0.0, 10.0))
    assert find_speaker_at_time(0.0, diar) == "SPEAKER_00"
    assert find_speaker_at_time(10.0, diar) == "SPEAKER_00"


def test_find_speaker_at_time_unknown_when_no_match():
    diar = _diarization(("SPEAKER_00", 5.0, 10.0))
    assert find_speaker_at_time(2.0, diar) == "UNKNOWN"


def test_find_speaker_at_time_empty_diarization():
    assert find_speaker_at_time(1.0, []) == "UNKNOWN"


def test_find_speaker_at_time_first_match_on_overlap():
    diar = _diarization(
        ("SPEAKER_00", 0.0, 10.0),
        ("SPEAKER_01", 5.0, 15.0),
    )
    assert find_speaker_at_time(7.0, diar) == "SPEAKER_00"


def test_merge_normal_two_speakers():
    whisper = _whisper_with_words(
        [
            (
                "hello world goodbye",
                0.0,
                6.0,
                [
                    ("hello", 0.0, 1.0),
                    ("world", 1.0, 2.0),
                    ("goodbye", 4.0, 5.0),
                ],
            ),
        ]
    )
    diar = _diarization(
        ("SPEAKER_00", 0.0, 2.5),
        ("SPEAKER_01", 2.5, 6.0),
    )

    turns = merge_transcript_with_diarization(whisper, diar)

    assert len(turns) == 2
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].text == "hello world"
    assert turns[0].start == 0.0
    assert turns[0].end == 2.0
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[1].text == "goodbye"
    assert turns[1].start == 4.0
    assert turns[1].end == 5.0


def test_merge_single_speaker_single_word():
    whisper = _whisper_with_words(
        [
            ("yes", 1.0, 2.0, [("yes", 1.0, 2.0)]),
        ]
    )
    diar = _diarization(("SPEAKER_00", 0.0, 5.0))

    turns = merge_transcript_with_diarization(whisper, diar)

    assert len(turns) == 1
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].text == "yes"
    assert turns[0].start == 1.0
    assert turns[0].end == 2.0


def test_merge_empty_whisper_segments():
    diar = _diarization(("SPEAKER_00", 0.0, 10.0))
    assert merge_transcript_with_diarization([], diar) == []


def test_merge_empty_diarization_assigns_unknown():
    whisper = _whisper_with_words(
        [
            ("hi", 0.0, 1.0, [("hi", 0.0, 1.0)]),
        ]
    )
    turns = merge_transcript_with_diarization(whisper, [])

    assert len(turns) == 1
    assert turns[0].speaker == "UNKNOWN"


def test_merge_fallback_without_word_timestamps():
    whisper = [
        {"text": "segment one", "start": 0.0, "end": 5.0},
        {"text": "segment two", "start": 5.0, "end": 10.0},
    ]
    diar = _diarization(
        ("SPEAKER_00", 0.0, 5.0),
        ("SPEAKER_01", 5.0, 10.0),
    )

    turns = merge_transcript_with_diarization(whisper, diar)

    assert len(turns) == 2
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].text == "segment one"
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[1].text == "segment two"


def test_merge_strips_word_whitespace():
    whisper = _whisper_with_words(
        [
            (" hello ", 0.0, 1.0, [(" hello ", 0.0, 1.0)]),
        ]
    )
    diar = _diarization(("SPEAKER_00", 0.0, 2.0))

    turns = merge_transcript_with_diarization(whisper, diar)

    assert turns[0].text == "hello"
