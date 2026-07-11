"""eval/metrics/_judge_utils.py — Shared LLM-judge helpers.

Extracted from faithfulness.py so hallucination.py can reuse without
duplicating the Ollama call + JSON-parse-retry pattern.

All LLM calls go through call_llm() from backend/app/llm.py — no new client.
JSON-parse-retry pattern is the same as backend/app/summarizer.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# ── backend on sys.path (idempotent) ─────────────────────────────────────────
_BACKEND = Path(__file__).parent.parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.llm import call_llm  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

RETRY_SUFFIX = (
    "\n\nCRITICAL: Output ONLY valid JSON. "
    "No markdown fences, no preamble, no trailing text. "
    "Your entire response must start with '{' and end with '}'."
)

_EXPECTED_KEYS = {"supported", "reasoning"}


# ── helpers ───────────────────────────────────────────────────────────────────


def extract_json_string(text: str) -> str:
    """Strip markdown fences and return the innermost JSON object string."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_supported_response(raw: str) -> dict:
    """Parse ``{"supported": bool, "reasoning": str}`` from raw LLM output.

    Raises json.JSONDecodeError or ValueError on failure — callers handle retry.
    """
    cleaned = extract_json_string(raw)
    data = json.loads(cleaned)
    missing = _EXPECTED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Judge response missing keys: {missing}")
    return {
        "supported": bool(data["supported"]),
        "reasoning": str(data["reasoning"]),
    }


def call_supported_judge(system: str, user_message: str) -> dict:
    """Call the LLM judge once (with one retry) and return parsed result.

    Uses call_llm() — same Ollama client as generate_answer().
    Retry pattern mirrors summarizer.generate_summary().

    Returns:
        {"supported": bool, "reasoning": str}

    Raises:
        RuntimeError: Ollama unreachable or JSON parse failed after retry.
    """
    raw = call_llm(system, user_message)
    try:
        return parse_supported_response(raw)
    except (json.JSONDecodeError, ValueError):
        raw = call_llm(system, user_message + RETRY_SUFFIX)
        try:
            return parse_supported_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"Judge returned invalid JSON after retry. "
                f"Raw (first 500 chars): {raw[:500]!r}"
            ) from exc


def safe_print(text: str) -> None:
    """Write text to stdout, replacing unencodable characters (Windows cp1252)."""
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.buffer.write((text + "\n").encode(enc, errors="replace"))
    sys.stdout.buffer.flush()
