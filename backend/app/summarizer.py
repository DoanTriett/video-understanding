"""Auto-summary generation for meeting and lecture videos.

Uses full transcript from Postgres (not Qdrant retrieval), so this is a
full-document summary, not a RAG query.

Transcripts are truncated at _MAX_TRANSCRIPT_CHARS before being sent to the
model to keep summary generation bounded.
"""

import json
import re

from openai import OpenAI
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import SessionLocal

from .config import settings

_MAX_TRANSCRIPT_CHARS = 30_000

_MEETING_KEYS: set[str] = {"agenda", "decisions", "action_items", "participants"}
_LECTURE_KEYS: set[str] = {"topic_outline", "key_concepts", "examples"}

# Hallucination-guard: ground the model in the transcript, no external knowledge.
_SYSTEM_PROMPT = (
    "You are a precise summarizer. "
    "Use ONLY information explicitly present in the transcript provided. "
    "Do not infer, add, or fabricate any information not stated in the transcript. "
    "Respond with ONLY a valid JSON object - no markdown fences, no explanation text."
)

_MEETING_USER_TEMPLATE = """\
Transcript:
{transcript}

Summarize this meeting transcript into a JSON object with exactly these 4 keys:
- "agenda": list of strings - topics discussed, in order of appearance
- "decisions": list of strings - explicit decisions made during the meeting
- "action_items": list of strings - tasks assigned or agreed upon
- "participants": list of strings - speaker labels exactly as they appear in the transcript \
(e.g. "Speaker A", "SPEAKER_00"); do NOT guess or invent real names

Return ONLY the JSON object. Start with '{{' and end with '}}'."""

_LECTURE_USER_TEMPLATE = """\
Transcript:
{transcript}

Summarize this lecture transcript into a JSON object with exactly these 3 keys:
- "topic_outline": list of strings - main topics covered, in order of appearance
- "key_concepts": list of strings - important concepts, terms, or definitions introduced
- "examples": list of strings - examples or case studies explicitly mentioned

Return ONLY the JSON object. Start with '{{' and end with '}}'."""

# Appended on retry to be even more explicit about format.
_RETRY_SUFFIX = (
    "\n\nCRITICAL: Output ONLY valid JSON. "
    "No markdown fences (no ```), no preamble, no trailing text. "
    "Your entire response must start with '{' and end with '}'."
)


def _format_transcript(chunks: list) -> str:
    """Convert ORM Chunk objects to chronological transcript text.

    Format mirrors build_context() in retriever.py:
      [Speaker @ mm:ss] text   (when speaker label is present)
      [mm:ss] text             (when no speaker)
    chunks are already sorted by start from crud.get_chunks().
    """
    lines: list[str] = []
    for chunk in chunks:
        start_sec = int(chunk.start)
        mm, ss = divmod(start_sec, 60)
        timestamp = f"{mm:02d}:{ss:02d}"
        prefix = f"[{chunk.speaker} @ {timestamp}]" if chunk.speaker else f"[{timestamp}]"
        lines.append(f"{prefix} {chunk.text}")
    return "\n".join(lines)


def _extract_json_string(text: str) -> str:
    """Best-effort extraction of a JSON object from raw LLM output.

    Handles the common case where the model wraps its answer in markdown fences
    despite being told not to.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fenced:
        return fenced.group(1).strip()
    # Fallback: find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_and_validate(raw: str, expected_keys: set[str]) -> dict:
    """Parse JSON from raw LLM text and validate required keys are present.

    Raises json.JSONDecodeError or ValueError on failure; callers handle retry.
    """
    cleaned = _extract_json_string(raw)
    data = json.loads(cleaned)  # raises json.JSONDecodeError if not valid JSON
    missing = expected_keys - data.keys()
    if missing:
        raise ValueError(f"LLM response missing required keys: {missing}")
    # Return only the expected keys; discard any extra fields the model invented.
    return {k: data[k] for k in expected_keys}


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


def generate_summary(video_id: str) -> dict:
    """Generate and persist an auto-summary for *video_id*.

    Reads all chunks from Postgres, builds a full-document prompt, calls the
    configured LLM once (with one retry on JSON parse failure), and upserts the
    result into the ``summaries`` table.

    Returns:
        The summary content dict (keys depend on video_type).

    Raises:
        ValueError: video not found, no chunks, or unsupported video_type.
        RuntimeError: LLM unavailable, or JSON parse failed after retry.
    """
    db: Session = SessionLocal()
    try:
        video = crud.get_video(db, video_id)
        if not video:
            raise ValueError(f"Video '{video_id}' not found in database")

        video_type = video.video_type
        chunks = crud.get_chunks(db, video_id)
        if not chunks:
            raise ValueError(
                f"No chunks found for video '{video_id}'. "
                "Run the processing pipeline before requesting a summary."
            )

        transcript = _format_transcript(chunks)

        if len(transcript) > _MAX_TRANSCRIPT_CHARS:
            transcript = transcript[:_MAX_TRANSCRIPT_CHARS]
            transcript += "\n\n[Transcript truncated - content above this line only]"

        if video_type == "meeting":
            user_prompt = _MEETING_USER_TEMPLATE.format(transcript=transcript)
            expected_keys = _MEETING_KEYS
        elif video_type == "lecture":
            user_prompt = _LECTURE_USER_TEMPLATE.format(transcript=transcript)
            expected_keys = _LECTURE_KEYS
        else:
            raise ValueError(
                f"Unsupported video_type '{video_type}'. " "Expected 'meeting' or 'lecture'."
            )

        raw = call_llm(_SYSTEM_PROMPT, user_prompt)
        try:
            content = _parse_and_validate(raw, expected_keys)
        except (json.JSONDecodeError, ValueError):
            raw = call_llm(_SYSTEM_PROMPT, user_prompt + _RETRY_SUFFIX)
            try:
                content = _parse_and_validate(raw, expected_keys)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"LLM returned invalid JSON after retry for video '{video_id}'. "
                    f"Raw response (first 500 chars): {raw[:500]!r}"
                ) from exc

        crud.upsert_summary(db, video_id=video_id, video_type=video_type, content=content)
        return content
    finally:
        db.close()
