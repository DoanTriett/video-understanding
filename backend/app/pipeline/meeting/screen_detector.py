import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class ScreenShareSegment:
    """
    Một đoạn video có screen share đang xảy ra.
    frame_path: đường dẫn đến representative frame đã lưu
    """
    start: float
    end: float
    frame_path: str
    confidence: float

def detect_screen_share(
    video_path: str,
    output_dir: str,
    sample_fps: float = 1.0        # sample 1 frame mỗi giây
) -> List[ScreenShareSegment]:
    """
    Detect các đoạn có screen share trong video meeting.
    
    Cách hoạt động:
    - Sample frames theo fps định sẵn
    - Tính "screen share score" cho mỗi frame
    - Các đoạn có score cao liên tiếp → screen share segment
    
    Screen share score dựa trên:
    1. Màu sắc: screen share thường có nhiều màu trắng/đơn sắc hơn
    2. Edge density: slide/screen có nhiều cạnh thẳng (text, border)
    3. Saturation thấp: UI thường ít màu sặc sỡ hơn video thật
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Sample interval tính bằng số frame
    sample_interval = int(fps / sample_fps)
    
    frame_scores = []  # [(timestamp, score, frame)]
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            timestamp = frame_idx / fps
            score = _compute_screen_share_score(frame)
            frame_scores.append((timestamp, score, frame.copy()))
        
        frame_idx += 1  
    
    cap.release()
    
    # Ngưỡng để coi là screen share
    THRESHOLD = 0.25
    
    # ── DEBUG: in score stats để xem threshold có hợp lý không ──
    if frame_scores:
        scores = [s for _, s, _ in frame_scores]
        print(f"  [Screen Detection Debug]")
        print(f"  Total frames sampled: {len(scores)}")
        print(f"  Score range: min={min(scores):.3f}, max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}")
        print(f"  Current threshold: {THRESHOLD}")
        #print(f"  Frames above threshold: {sum(1 for s in scores if s >= THRESHOLD)}")
        print(f"  Frames above threshold {THRESHOLD}: {sum(1 for s in scores if s >= THRESHOLD)}")
        
        # In phân bố score theo bucket
        buckets = [0]*10
        for s in scores:
            buckets[min(int(s*10), 9)] += 1
        print(f"  Score distribution (0.0-1.0):")
        for i, count in enumerate(buckets):
            bar = '█' * count
            print(f"    {i/10:.1f}-{(i+1)/10:.1f}: {bar} ({count})")
    
    #THRESHOLD = 0.6
    segments = _group_into_segments(
        frame_scores,
        THRESHOLD, output_dir)
    
    return segments

    """# Group các frames liên tiếp có score cao
    segments = _group_into_segments(
        frame_scores, THRESHOLD, output_dir
    )
    
    return segments"""


def _compute_screen_share_score(frame: np.ndarray) -> float:
    """
    Tính điểm "khả năng là screen share" cho một frame.
    Trả về float 0.0-1.0, càng cao càng có khả năng là screen share.
    """
    # Convert sang các color spaces cần dùng
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Feature 1: Tỷ lệ pixel màu trắng (slide thường có nền trắng)
    # Pixel "gần trắng": Value cao (>200) và Saturation thấp (<30)
    white_mask = (hsv[:,:,2] > 200) & (hsv[:,:,1] < 30)
    white_ratio = np.sum(white_mask) / white_mask.size
    
    # Feature 2: Edge density (slide/screen có nhiều edge rõ nét)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Feature 3: Saturation thấp = ít màu sặc sỡ = UI/slide
    mean_saturation = np.mean(hsv[:,:,1]) / 255.0
    low_saturation_score = 1.0 - mean_saturation
    
    """# Feature 4: Uniformity cao = nhiều vùng màu đồng nhất (nền slide)
    # Tính bằng std của gray: thấp = đồng nhất, cao = phức tạp
    # Nghịch đảo: đồng nhất cao → score cao
    local_std = cv2.GaussianBlur(
        (gray.astype(float) - cv2.GaussianBlur(gray, (15,15), 0))**2,
        (15,15), 0
    )
    complexity = np.mean(np.sqrt(local_std)) / 255.0
    uniformity_score = 1.0 - min(complexity * 3, 1.0)"""
    
    # Feature mới — detect vùng màu đồng nhất lớn (nền slide/screen)
    # Chia frame thành grid 4x4, đếm số cell có variance thấp
    h, w = gray.shape
    cell_h, cell_w = h // 4, w // 4
    uniform_cells = 0
    for i in range(4):
        for j in range(4):
            cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            if cell.std() < 20:  # variance thấp = màu đồng nhất
                uniform_cells += 1
    uniformity_score = uniform_cells / 16.0
    
    # Feature mới — detect text density (slide có nhiều text)
    # Dùng threshold để tìm pixel tối trên nền sáng = text
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    text_density = np.sum(binary > 0) / binary.size
    # Text density vừa phải (0.05-0.3) = có slide text
    text_score = 1.0 if 0.05 < text_density < 0.3 else 0.0
    
    # Weighted combination
    score = (
        white_ratio          * 0.25 +   
        edge_density         * 0.20 +
        low_saturation_score * 0.15 +
        uniformity_score     * 0.25 +  # tăng weight cho feature mới
        text_score           * 0.15    # thêm text detection
    )
    
    return float(min(score, 1.0))


def _group_into_segments(
    frame_scores: list,
    threshold: float,
    output_dir: str
) -> List[ScreenShareSegment]:
    """Group frames liên tiếp có score >= threshold thành segments"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    segments = []
    in_segment = False
    seg_start = None
    seg_frames = []
    seg_index = 0
    
    MIN_SEGMENT_DURATION = 3.0  # bỏ qua segment ngắn hơn 3s
    
    for i, (timestamp, score, frame) in enumerate(frame_scores):
        if score >= threshold:
            if not in_segment:
                in_segment = True
                seg_start = timestamp
                seg_frames = []
            seg_frames.append((timestamp, score, frame))
        else:
            if in_segment:
                # Kết thúc segment
                seg_end = frame_scores[i-1][0]
                duration = seg_end - seg_start
                
                if duration >= MIN_SEGMENT_DURATION:
                    # Lưu frame đại diện (frame có score cao nhất)
                    best_frame = max(seg_frames, key=lambda x: x[1])
                    frame_path = os.path.join(
                        output_dir, f"screen_{seg_index:04d}.jpg"
                    )
                    cv2.imwrite(frame_path, best_frame[2])
                    
                    avg_confidence = np.mean([s for _, s, _ in seg_frames])
                    segments.append(ScreenShareSegment(
                        start=seg_start,
                        end=seg_end,
                        frame_path=frame_path,
                        confidence=float(avg_confidence)
                    ))
                    seg_index += 1
                
                in_segment = False
                seg_frames = []
    
    # Đừng quên segment cuối
    if in_segment and seg_frames:
        seg_end = seg_frames[-1][0]
        if seg_end - seg_start >= MIN_SEGMENT_DURATION:
            best_frame = max(seg_frames, key=lambda x: x[1])
            frame_path = os.path.join(output_dir, f"screen_{seg_index:04d}.jpg")
            cv2.imwrite(frame_path, best_frame[2])
            segments.append(ScreenShareSegment(
                start=seg_start,
                end=seg_end,
                frame_path=frame_path,
                confidence=float(np.mean([s for _, s, _ in seg_frames]))
            ))
    
    return segments