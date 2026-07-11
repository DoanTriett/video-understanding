import ollama

from app.config import settings

_QA_SYSTEM_PROMPT = (
    "Chỉ trả lời dựa trên context được cung cấp. "
    "Nếu context không chứa câu trả lời, nói rõ là không tìm thấy thông tin, "
    "không tự suy đoán hay bịa.\n"
    "Khi trả lời, luôn trích timestamp dạng [mm:ss] lấy từ context cho mỗi claim quan trọng."
)


# qwen2.5:7b generates a full /ask answer in a few seconds on the RTX 4050 dev
# box; 60s gives ample headroom for a cold model load or a slow context while
# still failing fast instead of hanging the request indefinitely.
OLLAMA_TIMEOUT_SECONDS = 60


def call_llm(system: str, user: str) -> str:
    """Generic single-turn LLM call via Ollama. Raises RuntimeError if unreachable or timed out."""
    client = ollama.Client(host=settings.ollama_host, timeout=OLLAMA_TIMEOUT_SECONDS)
    try:
        response = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama server not reachable or timed out after {OLLAMA_TIMEOUT_SECONDS}s "
            f"at {settings.ollama_host}. Run `ollama serve`."
        ) from exc
    return response.message.content  # type: ignore[return-value]


def generate_answer(question: str, context: str) -> str:
    user_message = f"Context:\n{context}\n\nCâu hỏi: {question}"
    return call_llm(_QA_SYSTEM_PROMPT, user_message)
