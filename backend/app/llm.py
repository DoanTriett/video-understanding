from openai import OpenAI

from .config import settings

_QA_SYSTEM_PROMPT = (
    "Chỉ trả lời dựa trên context được cung cấp. "
    "Nếu context không chứa câu trả lời, nói rõ là không tìm thấy thông tin, "
    "không tự suy đoán hay bịa.\n"
    "Khi trả lời, luôn trích timestamp dạng [mm:ss] lấy từ context cho mỗi claim quan trọng."
)


def call_llm(system: str, user: str) -> str:
    """Generic single-turn LLM call via the OpenAI API."""
    client = OpenAI(api_key=settings.openai_api_key)
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:
        raise RuntimeError("OpenAI API request failed") from exc
    return response.choices[0].message.content or ""


def generate_answer(question: str, context: str) -> str:
    user_message = f"Context:\n{context}\n\nCâu hỏi: {question}"
    return call_llm(_QA_SYSTEM_PROMPT, user_message)
