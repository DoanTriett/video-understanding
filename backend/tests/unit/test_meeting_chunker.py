"""Unit tests cho chunk_meeting."""

from app.pipeline.meeting.chunker import (
    MAX_CHUNK_DURATION,
    MIN_TURN_DURATION,
    MeetingChunk,
    chunk_meeting,
)
from app.pipeline.meeting.merger import SpeakerTurn, WordWithSpeaker


def _turn(
    speaker: str,
    text: str,
    start: float,
    end: float,
    words: list[WordWithSpeaker] | None = None,
) -> SpeakerTurn:
    return SpeakerTurn(speaker=speaker, text=text, start=start, end=end, words=words or [])


def test_chunk_meeting_normal_multiple_turns():
    turns = [
        _turn("SPEAKER_00", "First topic here.", 0.0, 12.0),
        _turn("SPEAKER_01", "Second speaker reply.", 13.0, 25.0),
        _turn("SPEAKER_00", "Closing remark.", 26.0, 35.0),
    ]

    chunks = chunk_meeting(turns, "vid1")

    assert len(chunks) == 3
    assert all(isinstance(c, MeetingChunk) for c in chunks)
    assert chunks[0].chunk_id == "vid1_chunk_0000"
    assert chunks[0].speaker == "SPEAKER_00"
    assert chunks[0].text == "First topic here."
    assert chunks[1].chunk_id == "vid1_chunk_0001"
    assert chunks[2].turn_index == 2


def test_chunk_meeting_empty_turns():
    assert chunk_meeting([], "vid1") == []


def test_chunk_meeting_single_turn():
    turns = [_turn("SPEAKER_00", "Only turn.", 0.0, 10.0)]

    chunks = chunk_meeting(turns, "vidX")

    assert len(chunks) == 1
    assert chunks[0].text == "Only turn."
    assert chunks[0].start == 0.0
    assert chunks[0].end == 10.0


def test_chunk_meeting_merges_short_turn():
    short = _turn("SPEAKER_00", "Yes.", 0.0, 1.0)
    long_enough = _turn("SPEAKER_01", "I agree with the proposal.", 2.0, 10.0)
    assert short.end - short.start < MIN_TURN_DURATION

    chunks = chunk_meeting([short, long_enough], "vid1")

    assert len(chunks) == 1
    assert "Yes." in chunks[0].text
    assert "I agree" in chunks[0].text
    assert chunks[0].start == 0.0
    assert chunks[0].end == 10.0


def test_chunk_meeting_merges_same_speaker_with_small_gap():
    turns = [
        _turn("SPEAKER_00", "Part one.", 0.0, 8.0),
        _turn("SPEAKER_00", "Part two.", 9.0, 18.0),
    ]

    chunks = chunk_meeting(turns, "vid1")

    assert len(chunks) == 1
    assert chunks[0].speaker == "SPEAKER_00"
    assert chunks[0].text == "Part one. Part two."


def test_chunk_meeting_splits_long_turn_without_words():
    duration = MAX_CHUNK_DURATION * 2 + 10.0
    turns = [_turn("SPEAKER_00", "Very long monologue.", 0.0, duration)]

    chunks = chunk_meeting(turns, "vid1")

    assert len(chunks) >= 2
    assert chunks[0].start == 0.0
    assert chunks[-1].end == duration
    assert all(c.speaker == "SPEAKER_00" for c in chunks)


def test_chunk_meeting_splits_long_turn_at_word_boundaries():
    words = []
    t = 0.0
    while t < 100.0:
        words.append(WordWithSpeaker(word="word", start=t, end=t + 10.0, speaker="SPEAKER_00"))
        t += 10.0
    turn = _turn("SPEAKER_00", "word " * 10, 0.0, 100.0, words=words)

    chunks = chunk_meeting([turn], "vid1")

    assert len(chunks) >= 2
    assert all(c.end - c.start <= MAX_CHUNK_DURATION + 10.0 for c in chunks)
    assert chunks[0].text.count("word") >= 1
