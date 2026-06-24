import ollama

from app.config import settings

_SYSTEM_PROMPT = (
    "Chỉ trả lời dựa trên context được cung cấp. "
    "Nếu context không chứa câu trả lời, nói rõ là không tìm thấy thông tin, "
    "không tự suy đoán hay bịa.\n"
    "Khi trả lời, luôn trích timestamp dạng [mm:ss] lấy từ context cho mỗi claim quan trọng."
)


def generate_answer(question: str, context: str) -> str:
    client = ollama.Client(host=settings.ollama_host)
    user_message = f"Context:\n{context}\n\nCâu hỏi: {question}"

    try:
        response = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama server not reachable at {settings.ollama_host}. Run `ollama serve`."
        ) from exc

    return response.message.content  # type: ignore[return-value]
