from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from typing import List
import numpy as np

# Load một lần, dùng lại
_model = None
_processor = None

def get_clip_model():
    global _model, _processor
    if _model is None:
        print("Loading CLIP model (downloads ~600MB first run)...")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = _model.to("cuda")
        _model.eval()
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
    
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        features = model.get_image_features(**inputs)
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
    
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features[0].tolist()