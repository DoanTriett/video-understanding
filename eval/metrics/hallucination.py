"""eval/metrics/hallucination.py — Claim-level hallucination rate.

Algorithm:
  1. Split the answer into sentences (regex, no heavy NLP).
  2. Judge each sentence independently: is this claim supported by the context?
     Uses the same call_supported_judge() helper as faithfulness.py.
  3. hallucination_rate = unsupported_sentences / total_sentences.

A sentence is skipped (not judged) if it is empty or a pure timestamp/filler
(e.g. "[02:34]") — those carry no factual claim.

Shared LLM call + JSON-parse-retry logic: eval/metrics/_judge_utils.py.

Usage (from repo root):
    python eval/metrics/hallucination.py --video-id <id> --question "..."
"""

from __future__ import annotations

import argparse
import re
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


# ── claim-level judge prompt (different from answer-level faithfulness) ───────

_CLAIM_JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. "
    "Given a CONTEXT and a single CLAIM, decide whether the claim is "
    "directly supported by the context. "
    "A claim is supported only if the context explicitly contains the "
    "information needed to verify it — do not use outside knowledge. "
    "Respond with ONLY a valid JSON object — no markdown fences, no extra text."
)

_CLAIM_JUDGE_USER_TEMPLATE = """\
CONTEXT:
{context}

CLAIM:
{claim}

Is this claim directly supported by the context above?

Return ONLY this JSON object:
{{"supported": <true or false>, "reasoning": "<one sentence>"}}

Start your response with '{{' and end with '}}'."""


# ── sentence splitter ─────────────────────────────────────────────────────────

# Split on sentence-ending punctuation (.!?) followed by whitespace or end-of-string.
# Vietnamese text ends sentences the same way; this regex handles both languages.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A sentence is a "filler" (no factual claim) if it consists only of:
#   - whitespace
#   - timestamp tokens like [02:34] or [Speaker @ 02:34]
#   - very short fragments (< 8 chars after stripping)
_FILLER = re.compile(r"^\s*(\[[^\]]*\]\s*)*\s*$")


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using punctuation regex.

    Returns only sentences that carry at least one factual claim
    (non-empty, not pure timestamp/filler, >= 8 chars).
    """
    raw = _SENT_SPLIT.split(text.strip())
    return [
        s.strip()
        for s in raw
        if s.strip() and not _FILLER.match(s) and len(s.strip()) >= 8
    ]


# ── public API ────────────────────────────────────────────────────────────────


class ClaimResult(TypedDict):
    claim: str
    supported: bool
    reasoning: str


class HallucinationResult(TypedDict):
    hallucination_rate: float
    total_claims: int
    unsupported_count: int
    claims: list[ClaimResult]


def compute_hallucination_rate(answer: str, context: str) -> HallucinationResult:
    """Judge each sentence in *answer* and return claim-level hallucination rate.

    Returns:
        {
            "hallucination_rate": float,   # unsupported / total  (0.0 if no claims)
            "total_claims": int,
            "unsupported_count": int,
            "claims": [{"claim": str, "supported": bool, "reasoning": str}, ...]
        }
    """
    sentences = split_sentences(answer)

    if not sentences:
        return HallucinationResult(
            hallucination_rate=0.0,
            total_claims=0,
            unsupported_count=0,
            claims=[],
        )

    claim_results: list[ClaimResult] = []
    for sentence in sentences:
        user_msg = _CLAIM_JUDGE_USER_TEMPLATE.format(context=context, claim=sentence)
        verdict = call_supported_judge(_CLAIM_JUDGE_SYSTEM, user_msg)
        claim_results.append(
            ClaimResult(
                claim=sentence,
                supported=verdict["supported"],
                reasoning=verdict["reasoning"],
            )
        )

    unsupported = sum(1 for c in claim_results if not c["supported"])
    rate = unsupported / len(claim_results)

    return HallucinationResult(
        hallucination_rate=rate,
        total_claims=len(claim_results),
        unsupported_count=unsupported,
        claims=claim_results,
    )


# ── CLI demo ──────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute claim-level hallucination rate for a QA pair.",
    )
    parser.add_argument("--video-id", required=True, metavar="ID")
    parser.add_argument("--question", required=True, metavar="Q")
    parser.add_argument("--top-k", type=int, default=6, metavar="N")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    safe_print(f"\n[hallucination] video_id={args.video_id}")
    safe_print(f"[hallucination] question : {args.question}\n")

    chunks = retrieve(args.video_id, args.question, args.top_k)
    if not chunks:
        safe_print("[hallucination] No chunks retrieved -- cannot evaluate.")
        return

    context = build_context(chunks)
    safe_print(f"[hallucination] Context ({len(chunks)} chunks):\n{context}\n")

    from app.llm import generate_answer  # noqa: PLC0415

    answer = generate_answer(args.question, context)
    safe_print(f"[hallucination] Generated answer:\n{answer}\n")

    # ── claim-level breakdown ──────────────────────────────────────────────────
    result = compute_hallucination_rate(answer, context)

    safe_print("[hallucination] Claim-level breakdown:")
    safe_print(f"  {'#':<3} {'sup':>4}  claim")
    safe_print("  " + "-" * 72)
    for i, c in enumerate(result["claims"], 1):
        tag = "YES" if c["supported"] else "NO "
        claim_display = c["claim"][:65] + ".." if len(c["claim"]) > 67 else c["claim"]
        safe_print(f"  {i:<3} {tag:>4}  {claim_display}")
        safe_print(f"       reason: {c['reasoning']}")

    safe_print(
        f"\n[hallucination] hallucination_rate = "
        f"{result['unsupported_count']}/{result['total_claims']} = "
        f"{result['hallucination_rate']:.2%}"
    )

    # ── aggregate over dataset (all auto-exported pairs for this video) ────────
    safe_print(
        "\n[hallucination] Aggregate over dataset: see run_eval.py (TODO Prompt 6)."
    )


if __name__ == "__main__":
    main()
