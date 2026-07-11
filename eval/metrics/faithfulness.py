"""eval/metrics/faithfulness.py — LLM-as-judge faithfulness metric (answer-level).

Measures whether a generated answer is supported by the retrieved context.
Uses the same Ollama model as generate_answer() but via a separate judge call.

Shared LLM call + JSON-parse-retry logic lives in _judge_utils.py.

Usage (from repo root):
    python eval/metrics/faithfulness.py --video-id <id> --question "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

# ── eval imports ───────────────────────────────────────────────────────────────
_EVAL_ROOT = Path(__file__).parent.parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from eval.metrics._judge_utils import call_supported_judge, safe_print  # noqa: E402

# ── backend imports (via _judge_utils sys.path setup) ─────────────────────────
from app.pipeline.shared.retriever import build_context, retrieve  # noqa: E402


# ── judge prompt ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict faithfulness judge. "
    "Your only job is to decide whether the given ANSWER is fully supported by "
    "the CONTEXT provided — meaning every factual claim in the answer can be "
    "traced to a specific part of the context. "
    "Do not use any outside knowledge. "
    "Respond with ONLY a valid JSON object — no markdown fences, no extra text."
)

_JUDGE_USER_TEMPLATE = """\
CONTEXT:
{context}

ANSWER:
{answer}

Is every claim in the ANSWER supported by the CONTEXT above?

Return ONLY this JSON object:
{{"supported": <true or false>, "reasoning": "<one sentence explanation>"}}

Start your response with '{{' and end with '}}'."""


# ── public API ────────────────────────────────────────────────────────────────


class FaithfulnessResult(TypedDict):
    supported: bool
    reasoning: str


def judge_faithfulness(answer: str, context: str) -> FaithfulnessResult:
    """Return {"supported": bool, "reasoning": str} for the given answer/context."""
    user_message = _JUDGE_USER_TEMPLATE.format(context=context, answer=answer)
    return call_supported_judge(_JUDGE_SYSTEM, user_message)  # type: ignore[return-value]


# ── CLI demo ──────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM faithfulness judge on a single question/answer pair.",
    )
    parser.add_argument("--video-id", required=True, metavar="ID")
    parser.add_argument("--question", required=True, metavar="Q")
    parser.add_argument("--top-k", type=int, default=6, metavar="N")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    safe_print(f"\n[faithfulness] video_id={args.video_id}")
    safe_print(f"[faithfulness] question : {args.question}\n")

    chunks = retrieve(args.video_id, args.question, args.top_k)
    if not chunks:
        safe_print("[faithfulness] No chunks retrieved -- cannot evaluate.")
        return

    context = build_context(chunks)
    safe_print(f"[faithfulness] Context ({len(chunks)} chunks):\n{context}\n")

    from app.llm import generate_answer  # noqa: PLC0415

    answer = generate_answer(args.question, context)
    safe_print(f"[faithfulness] Generated answer:\n{answer}\n")

    result = judge_faithfulness(answer, context)
    safe_print("[faithfulness] Judge result:")
    safe_print(json.dumps(result, indent=2, ensure_ascii=False))

    safe_print("\n[faithfulness] Sanity test -- clearly wrong answer:")
    fake = "The meeting was held on Mars and the CEO announced a pet hamster policy."
    sanity = judge_faithfulness(fake, context)
    safe_print(json.dumps(sanity, indent=2, ensure_ascii=False))
    if not sanity["supported"]:
        safe_print("[faithfulness] PASS: judge correctly flagged unsupported answer.")
    else:
        safe_print(
            "[faithfulness] WARN: judge did NOT flag unsupported answer (check prompt)."
        )


if __name__ == "__main__":
    main()
