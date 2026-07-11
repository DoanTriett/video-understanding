"""Unit tests cho build_lecture_chunks."""

from types import SimpleNamespace

from app.pipeline.lecture.chunker import LectureChunk, build_lecture_chunks


def _slide(timestamp: float, index: int) -> SimpleNamespace:
    return SimpleNamespace(timestamp=timestamp, frame_path=f"/tmp/slide_{index:04d}.jpg")


def _segments() -> list[dict]:
    return [
        {"text": "intro one", "start": 0.0, "end": 8.0},
        {"text": "topic two", "start": 9.0, "end": 14.0},
        {"text": "detail two", "start": 15.0, "end": 22.0},
        {"text": "conclusion three", "start": 26.0, "end": 30.0},
    ]


def test_build_lecture_chunks_normal():
    slides = [_slide(0.0, 0), _slide(10.0, 1), _slide(25.0, 2)]
    ocr = [{"ocr_text": "Title A"}, {"ocr_text": "Title B"}, {"ocr_text": "Title C"}]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr)

    assert len(chunks) == 3
    assert all(isinstance(c, LectureChunk) for c in chunks)
    # Segment "topic two" (9–14) overlap biên slide 0 [0,10) nên cũng nằm trong slide 0.
    assert chunks[0].transcript_text == "intro one topic two"
    assert chunks[1].transcript_text == "topic two detail two"
    assert chunks[2].transcript_text == "conclusion three"


def test_build_lecture_chunks_empty_slides():
    assert build_lecture_chunks("vid1", _segments(), [], []) == []


def test_build_lecture_chunks_single_slide():
    slides = [_slide(0.0, 0)]
    ocr = [{"ocr_text": "Only slide"}]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr)

    assert len(chunks) == 1
    assert chunks[0].slide_index == 0
    assert chunks[0].end == 30.0


def test_build_lecture_chunks_segment_overlap_at_boundary():
    slides = [_slide(0.0, 0), _slide(10.0, 1)]
    segments = [{"text": "spanning", "start": 8.0, "end": 12.0}]
    ocr = [{"ocr_text": "A"}, {"ocr_text": "B"}]

    chunks = build_lecture_chunks("vid1", segments, slides, ocr)

    assert chunks[0].transcript_text == "spanning"
    assert chunks[0].end == 10.0
    assert chunks[1].transcript_text == "spanning"
    assert chunks[1].end == 12.0


def test_build_lecture_chunks_missing_ocr_results():
    slides = [_slide(0.0, 0), _slide(10.0, 1)]
    ocr = [{"ocr_text": "Only first slide OCR"}]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr)

    assert chunks[0].ocr_text == "Only first slide OCR"
    assert chunks[1].ocr_text == ""
    assert "[SLIDE]" in chunks[1].combined_text


def test_build_lecture_chunks_empty_transcript():
    slides = [_slide(0.0, 0)]
    ocr = [{"ocr_text": "Heading"}]

    chunks = build_lecture_chunks("vid1", [], slides, ocr)

    assert chunks[0].transcript_text == ""
    assert chunks[0].combined_text == "[SLIDE] Heading\n[SPEECH]"
    assert chunks[0].end == 60.0


def test_build_lecture_chunks_clip_embeddings_optional():
    slides = [_slide(0.0, 0), _slide(10.0, 1)]
    ocr = [{"ocr_text": "a"}, {"ocr_text": "b"}]
    embeddings = [[0.1] * 512, [0.2] * 512]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr, clip_embeddings=embeddings)

    assert chunks[0].clip_embedding == embeddings[0]
    assert chunks[1].clip_embedding == embeddings[1]

    chunks_no_emb = build_lecture_chunks("vid1", _segments(), slides, ocr)
    assert chunks_no_emb[0].clip_embedding is None
