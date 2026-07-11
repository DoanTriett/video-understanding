# test_ocr_manual.py — script chạy tay để kiểm tra quality
from app.pipeline.lecture.ocr import extract_text_from_slide

"""detect_slide_changes(
    video_path="uploads/test_videos/meeting-with-slide.mp4",
    output_dir="uploads/test_videos")"""

# Chạy thử trên 2-3 slide frames đã extract từ bước slide detection
test_frames = [
    "uploads/test_videos/slide_0000_0.0s.jpg",
    "uploads/test_videos/slide_0001_9.0s.jpg",
    "uploads/test_videos/slide_0002_12.0s.jpg",
    "uploads/test_videos/slide_0003_59.0s.jpg",
    "uploads/test_videos/slide_0004_85.0s.jpg",
    "uploads/test_videos/slide_0005_89.0s.jpg",
    "uploads/test_videos/slide_0006_94.0s.jpg",
    "uploads/test_videos/slide_0007_111.0s.jpg",
    "uploads/test_videos/slide_0008_118.0s.jpg",
    "uploads/test_videos/slide_0009_121.0s.jpg",
    "uploads/test_videos/slide_0010_151.0s.jpg",
    "uploads/test_videos/slide_0011_195.0s.jpg",
    "uploads/test_videos/slide_0012_223.0s.jpg",
]

for frame_path in test_frames:
    text = extract_text_from_slide(frame_path)
    print(f"=== {frame_path} ===")
    print(text)
    print()
