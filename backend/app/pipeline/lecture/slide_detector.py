from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


@dataclass
class SlideChange:
    timestamp: float
    frame_index: int
    frame_path: str


def detect_slide_changes(
    video_path: str,
    output_dir: str,
    sample_fps: float = 1.0,
    ssim_threshold: float = 0.85,
    min_gap_seconds: float = 3.0,
    min_slide_score: float = 0.45,
) -> list[SlideChange]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps / sample_fps), 1)

    changes = []
    prev_gray = None
    last_change_ts = -min_gap_seconds
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))

            timestamp = frame_idx / fps
            slide_score = _compute_projected_slide_score(frame)

            if prev_gray is not None:
                score, _ = ssim(prev_gray, gray, full=True)
                if (
                    slide_score >= min_slide_score
                    and score < ssim_threshold
                    and (timestamp - last_change_ts) >= min_gap_seconds
                ):
                    frame_path = f"{output_dir}/slide_{len(changes):04d}_{timestamp:.1f}s.jpg"
                    cv2.imwrite(frame_path, frame)
                    changes.append(
                        SlideChange(
                            timestamp=timestamp,
                            frame_index=frame_idx,
                            frame_path=frame_path,
                        )
                    )
                    last_change_ts = timestamp
            elif slide_score >= min_slide_score:
                frame_path = f"{output_dir}/slide_0000_0.0s.jpg"
                cv2.imwrite(frame_path, frame)
                changes.append(SlideChange(timestamp=0.0, frame_index=0, frame_path=frame_path))
                last_change_ts = 0.0

            prev_gray = gray
        frame_idx += 1

    cap.release()
    return changes


def _compute_projected_slide_score(frame: np.ndarray) -> float:
    """Estimate whether a frame contains one large projected slide area."""
    if _looks_like_gallery_layout(frame):
        return 0.0

    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    foreground = (hsv[:, :, 2] > 35).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / float(width * height)
        if area_ratio < 0.30:
            continue

        aspect_ratio = w / float(h)
        if not 1.15 <= aspect_ratio <= 2.4:
            continue

        crop = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        white_ratio = float(np.mean((crop_hsv[:, :, 2] > 190) & (crop_hsv[:, :, 1] < 70)))
        low_saturation = float(1.0 - np.mean(crop_hsv[:, :, 1]) / 255.0)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))
        text_like_edges = 1.0 if 0.01 <= edge_density <= 0.18 else 0.0

        score = min(
            1.0,
            area_ratio * 0.45 + white_ratio * 0.20 + low_saturation * 0.20 + text_like_edges * 0.15,
        )
        best_score = max(best_score, score)

    return best_score


def _looks_like_gallery_layout(frame: np.ndarray) -> bool:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    foreground = (hsv[:, :, 2] > 35).astype(np.uint8) * 255

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tile_count = 0

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / float(width * height)
        aspect_ratio = w / float(h)
        if 0.035 <= area_ratio <= 0.09 and 1.3 <= aspect_ratio <= 2.2:
            tile_count += 1

    return tile_count >= 6
