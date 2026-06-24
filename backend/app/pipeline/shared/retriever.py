from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.config import settings
from app.pipeline.shared.text_embedder import embed_text

_VISUAL_KEYWORDS = {
    # Vietnamese
    "slide",
    "hình",
    "màn hình",
    "biểu đồ",
    "công thức",
    # English
    "screen",
    "show",
    "diagram",
    "code",
    "chart",
    "figure",
    "image",
    "graph",
}


def route_query(question: str) -> str:
    """Return "visual" if the question targets on-screen content, else "text"."""
    lowered = question.lower()
    if any(kw in lowered for kw in _VISUAL_KEYWORDS):
        return "visual"
    return "text"


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def retrieve(video_id: str, question: str, top_k: int = 6) -> list[dict]:
    """Dense retrieval from Qdrant for a single video.

    Always searches ``chunks_text``.  When ``route_query`` returns "visual",
    also searches ``chunks_visual`` and merges the results.

    TODO: ``chunks_visual`` stores 512-dim CLIP image vectors.  Searching it
    with a 384-dim text vector will fail with a dimension mismatch.  Visual
    search is skipped until a CLIP text encoder (producing 512-dim vectors) is
    wired in.  At that point, replace ``_text_vector_for_visual`` below with
    ``clip_encode_text(question)``.
    """
    route = route_query(question)
    vector = embed_text(question)
    video_filter = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])

    client = _get_qdrant_client()

    raw: list[tuple[str, float, dict]] = []  # (chunk_id, score, payload)

    # Always search the text collection.
    text_hits = client.search(
        collection_name="chunks_text",
        query_vector=vector,
        query_filter=video_filter,
        limit=top_k,
        with_payload=True,
    )
    for hit in text_hits:
        payload = hit.payload or {}
        raw.append((payload.get("chunk_id", ""), hit.score, payload))

    # Visual search: skipped until CLIP text encoding is available (see TODO above).
    if route == "visual":
        pass  # TODO: clip_vector = clip_encode_text(question); search chunks_visual here

    # Dedupe by chunk_id, keeping the highest score per chunk.
    seen: dict[str, tuple[float, dict]] = {}
    for chunk_id, score, payload in raw:
        if chunk_id not in seen or score > seen[chunk_id][0]:
            seen[chunk_id] = (score, payload)

    results = sorted(seen.values(), key=lambda x: x[0], reverse=True)[:top_k]

    return [
        {
            "chunk_id": payload.get("chunk_id", ""),
            "speaker": payload.get("speaker"),
            "start": payload.get("start", 0.0),
            "end": payload.get("end", 0.0),
            "text": payload.get("text", ""),
            "chunk_type": payload.get("chunk_type", "transcript"),
            "score": score,
        }
        for score, payload in results
    ]


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM.

    Sorts by ``start`` (ascending) so the LLM reads events in timeline order,
    not relevance order.  Each line is ``[speaker @ mm:ss] text`` or
    ``[mm:ss] text`` when the chunk has no speaker.
    """
    ordered = sorted(chunks, key=lambda c: c.get("start", 0.0))
    lines: list[str] = []
    for chunk in ordered:
        start_sec = int(chunk.get("start", 0.0))
        mm, ss = divmod(start_sec, 60)
        timestamp = f"{mm:02d}:{ss:02d}"
        speaker = chunk.get("speaker")
        prefix = f"[{speaker} @ {timestamp}]" if speaker else f"[{timestamp}]"
        lines.append(f"{prefix} {chunk.get('text', '')}")
    return "\n".join(lines)
