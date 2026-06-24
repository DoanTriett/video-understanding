from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Auto-detect device: use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Loading SentenceTransformer on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Loading SentenceTransformer on CPU")
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)


def embed_text(text: str) -> list[float]:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
