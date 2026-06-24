from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Load một lần, dùng lại
_model = None
_processor = None
_device = None


def _get_device():
    """Determine device once at startup."""
    global _device
    if _device is None:
        # Check if CUDA is available and properly initialized
        if torch.cuda.is_available():
            try:
                # Try to allocate a small tensor to verify CUDA works
                torch.zeros(1).cuda()
                _device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            except RuntimeError as e:
                print(f"CUDA available but error during init: {e}. Falling back to CPU.")
                _device = torch.device("cpu")
        else:
            _device = torch.device("cpu")
            print("CUDA not available. Using CPU.")
    return _device


def get_clip_model():
    global _model, _processor
    if _model is None:
        device = _get_device()
        print("Loading CLIP model (downloads ~600MB first run)...")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = _model.to(device)
        _model.eval()
        print(f"CLIP model loaded on {device}")
    return _model, _processor


def embed_image(image_path: str) -> List[float]:
    """
    Tạo vector embedding cho một ảnh.

    CLIP encode ảnh thành vector 512 chiều.
    Các ảnh "giống nhau về nội dung" sẽ có vector gần nhau.
    Ví dụ: slide về "budget" sẽ gần với query "budget presentation".

    Returns: list 512 floats
    """
    model, processor = get_clip_model()
    device = _get_device()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.get_image_features(**inputs)
        # Extract tensor from BaseModelOutputWithPooling
        features = output.pooler_output
        # Normalize để cosine similarity hoạt động đúng
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()


def embed_text_with_clip(text: str) -> List[float]:
    """
    Embed text bằng CLIP — dùng khi query về visual content.

    CLIP được train để text và image embed vào cùng space,
    nên "a slide about budget" sẽ gần với ảnh slide thật.
    """
    model, processor = get_clip_model()
    device = _get_device()

    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.get_text_features(**inputs)
        # Extract tensor from BaseModelOutputWithPooling
        features = output.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().tolist()
