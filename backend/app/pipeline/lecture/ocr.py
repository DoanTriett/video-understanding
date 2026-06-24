import re
from functools import lru_cache

from paddleocr import PaddleOCR  # noqa: E402


@lru_cache(maxsize=1)
def _get_ocr_engine() -> PaddleOCR:
    """Lazy-load one shared PaddleOCR engine for all slide images.

    PaddleOCR is CPU-based only. We don't set CUDA_VISIBLE_DEVICES globally
    because that would hide GPU from other components (CLIP embedder).
    Instead, PaddleOCR will just use CPU internally.
    """
    return PaddleOCR(
        lang="en",
        device="cpu",
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


def extract_text_from_slide(image_path: str) -> str:
    """OCR one slide image and return cleaned top-to-bottom text."""
    ocr = _get_ocr_engine()
    result = ocr.predict(image_path)

    if not result or not result[0]:
        return ""

    ocr_result = result[0]
    texts = ocr_result.get("rec_texts", [])
    scores = ocr_result.get("rec_scores", [])
    polys = ocr_result.get("rec_polys", [])

    lines = []
    for text, confidence, poly in zip(texts, scores, polys):
        if confidence < 0.5:
            continue
        top_y = min(point[1] for point in poly)
        lines.append((top_y, text))

    lines.sort(key=lambda x: x[0])
    text = "\n".join(t for _, t in lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_slide_texts(slide_changes: list) -> list[dict]:
    """Run OCR on slide frames."""
    results = []
    for slide in slide_changes:
        text = extract_text_from_slide(slide.frame_path)
        results.append(
            {
                "timestamp": slide.timestamp,
                "frame_path": slide.frame_path,
                "ocr_text": text,
                "has_content": len(text.strip()) > 20,
            }
        )
    return results
