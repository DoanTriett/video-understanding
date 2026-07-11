"""Unit tests cho build_lecture_chunks — kiểm tra chunk boundaries khớp slide timestamps.

Không cần GPU/cv2/OCR: dùng SimpleNamespace giả lập SlideChange (chunker chỉ đọc
.timestamp và .frame_path).
"""

from types import SimpleNamespace

from app.pipeline.lecture.chunker import build_lecture_chunks


def _slide(timestamp, index):
    return SimpleNamespace(timestamp=timestamp, frame_path=f"/tmp/slide_{index:04d}.jpg")


def _segments():
    return [
        {"text": "intro one", "start": 0.0, "end": 8.0},
        {"text": "topic two", "start": 9.0, "end": 14.0},
        {"text": "detail two", "start": 15.0, "end": 22.0},
        {"text": "conclusion three", "start": 26.0, "end": 30.0},
    ]


def test_chunk_starts_align_with_slide_timestamps():
    slides = [_slide(0.0, 0), _slide(10.0, 1), _slide(25.0, 2)]
    ocr = [{"ocr_text": "Title A"}, {"ocr_text": "Title B"}, {"ocr_text": "Title C"}]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr)

    assert len(chunks) == len(slides)
    for i, slide in enumerate(slides):
        assert chunks[i].start == slide.timestamp
        assert chunks[i].slide_index == i
        assert chunks[i].chunk_id == f"vid1_slide_{i:04d}"


def test_chunk_end_clamped_to_next_slide_boundary():
    slides = [_slide(0.0, 0), _slide(10.0, 1), _slide(25.0, 2)]
    ocr = [{"ocr_text": ""}, {"ocr_text": ""}, {"ocr_text": ""}]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr)

    # Slide 0 [0,10): segment "topic two" (9-14) tràn qua biên → end clamp về 10.
    assert chunks[0].end == 10.0
    # Slide 1 [10,25): kết thúc tại segment cuối (detail two end=22).
    assert chunks[1].end == 22.0
    # Slide cuối [25,inf): kết thúc tại segment cuối thực tế (30).
    assert chunks[2].end == 30.0
    # Mỗi chunk không bao giờ vượt quá biên slide kế tiếp.
    for i in range(len(slides) - 1):
        assert chunks[i].end <= slides[i + 1].timestamp


def test_combined_text_merges_ocr_and_transcript():
    slides = [_slide(0.0, 0), _slide(10.0, 1)]
    ocr = [{"ocr_text": "Heading One"}, {"ocr_text": "Heading Two"}]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr)

    assert "Heading One" in chunks[0].combined_text
    assert "intro one" in chunks[0].combined_text
    assert chunks[0].combined_text.startswith("[SLIDE]")
    assert "[SPEECH]" in chunks[0].combined_text


def test_clip_embeddings_attached_per_slide():
    slides = [_slide(0.0, 0), _slide(10.0, 1)]
    ocr = [{"ocr_text": "a"}, {"ocr_text": "b"}]
    embeddings = [[0.1] * 512, [0.2] * 512]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr, clip_embeddings=embeddings)

    assert chunks[0].clip_embedding == embeddings[0]
    assert chunks[1].clip_embedding == embeddings[1]


def test_slide_without_transcript_uses_fallback_duration():
    # Slide tại t=100 không có segment nào overlap → end = start + 60.
    slides = [_slide(100.0, 0)]
    ocr = [{"ocr_text": "lonely slide"}]

    chunks = build_lecture_chunks("vid1", _segments(), slides, ocr)

    assert chunks[0].start == 100.0
    assert chunks[0].end == 160.0
    assert chunks[0].transcript_text == ""
