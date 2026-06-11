import subprocess
import os
from pathlib import Path

def extract_audio(video_path: str, output_dir: str) -> str:
    """
    Dùng ffmpeg để extract audio từ video.
    
    Tại sao cần extract riêng?
    Whisper nhận audio file, không nhận video file trực tiếp.
    Extract ra WAV 16kHz mono — đây là format tối ưu cho Whisper.
    
    Returns: đường dẫn đến file audio
    """
    video_path = Path(video_path)
    audio_filename = video_path.stem + ".wav"
    audio_path = os.path.join(output_dir, audio_filename)
    
    # ffmpeg command breakdown:
    # -i input_file        : input
    # -vn                  : no video (chỉ lấy audio)
    # -acodec pcm_s16le    : audio codec WAV 16-bit
    # -ar 16000            : sample rate 16kHz (Whisper requirement)
    # -ac 1                : mono channel
    # -y                   : overwrite nếu file đã tồn tại
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_path
    ]
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    return audio_path