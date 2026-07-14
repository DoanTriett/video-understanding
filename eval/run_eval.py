"""eval/run_eval.py — Full evaluation harness for the Video Understanding QA pipeline.

Design decision — in-process calls, NOT HTTP:
  retrieve(), build_context(), and generate_answer() have no FastAPI lifecycle
  dependencies and were already used directly in eval/metrics/*.py (Prompts 3-5).
  Calling them in-process avoids requiring a running uvicorn server, eliminates
  Redis semantic-cache hits (which would return stale answers instead of fresh
  pipeline output), and skips the DB video-status guard (eval dataset already
  identifies valid video_ids).

Pipeline per QA pair:
  1. retrieve(video_id, question, top_k)          → chunks (Qdrant)
  2. build_context(chunks)                         → context string
  3. generate_answer(question, context)            → answer (Ollama)
  4. compute_timestamp_overlap(chunks, gt_ranges)  → float  [only verified pairs]
  5. judge_faithfulness(answer, context)           → {supported, reasoning}
  6. compute_hallucination_rate(answer, context)   → {rate, claims, ...}

Only QA pairs with verified=True are used for retrieval overlap.
All QA pairs (verified or not) are used for faithfulness / hallucination.

Outputs:
  --output/report.json   machine-readable full results
  --output/report.md     human-readable summary tables

Usage:
    python eval/run_eval.py --dataset eval/datasets [--output eval/reports] [--top-k 6]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Disable rate limiting for this batch run. This must happen before any
# `app.*` import so app.config.settings.testing / app.limiter pick it up.
# Only affects Settings.testing — does NOT change production defaults
# (10/min ask, 5/min upload) used by the running API server.
os.environ.setdefault("TESTING", "true")

# ── sys.path setup ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_BACKEND = _REPO_ROOT / "backend"
for _p in [str(_REPO_ROOT), str(_BACKEND)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── backend imports ───────────────────────────────────────────────────────────
from app.llm import generate_answer  # noqa: E402
from app.pipeline.shared.retriever import build_context, retrieve  # noqa: E402

# ── eval imports ──────────────────────────────────────────────────────────────
from eval.loader import load_annotations  # noqa: E402
from eval.metrics._judge_utils import safe_print  # noqa: E402
from eval.metrics.faithfulness import judge_faithfulness  # noqa: E402
from eval.metrics.hallucination import compute_hallucination_rate  # noqa: E402
from eval.metrics.retrieval import compute_timestamp_overlap  # noqa: E402
from eval.schema import AnnotationFile  # noqa: E402


# ── MLflow helpers ────────────────────────────────────────────────────────────


def _get_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _mlflow_tracking_uri() -> str:
    """Always write to repo-root mlflow.db regardless of cwd."""
    db_path = (_REPO_ROOT / "mlflow.db").resolve().as_posix()
    return f"sqlite:///{db_path}"


def _log_mlflow(report: dict, top_k: int) -> None:
    """Log overall metrics and params for this eval run to MLflow."""
    import mlflow

    mlflow.set_tracking_uri(_mlflow_tracking_uri())
    o = report["overall"]
    with mlflow.start_run():
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("n_pairs", o["n_pairs"])
        mlflow.log_param("n_verified", o["n_verified"])
        git_sha = _get_git_sha()
        if git_sha:
            mlflow.log_param("git_sha", git_sha)
        if o["faithfulness_rate"] is not None:
            mlflow.log_metric("faithfulness_rate", o["faithfulness_rate"])
        if o["avg_hallucination_rate"] is not None:
            mlflow.log_metric("hallucination_rate", o["avg_hallucination_rate"])
        if o["avg_overlap"] is not None:
            mlflow.log_metric("retrieval_overlap", o["avg_overlap"])


# ── per-pair evaluation ───────────────────────────────────────────────────────


def _eval_pair(
    video_id: str,
    question: str,
    gt_ranges: list,
    top_k: int,
    verified: bool,
) -> dict:
    """Run the full pipeline for one QA pair and return raw metric dict."""
    chunks = retrieve(video_id, question, top_k)
    if not chunks:
        return {
            "question": question,
            "answer": None,
            "chunks_retrieved": 0,
            "overlap": None,
            "faithfulness_supported": None,
            "faithfulness_reasoning": None,
            "hallucination_rate": None,
            "hallucination_total_claims": None,
            "hallucination_unsupported": None,
            "verified": verified,
            "error": "no_chunks_retrieved",
        }

    context = build_context(chunks)
    answer = generate_answer(question, context)

    # Retrieval overlap — only meaningful for verified pairs.
    if verified and gt_ranges:
        overlap = compute_timestamp_overlap(chunks, gt_ranges)
    else:
        overlap = None

    # Faithfulness (answer-level).
    faith = judge_faithfulness(answer, context)

    # Hallucination (claim-level).
    hall = compute_hallucination_rate(answer, context)

    return {
        "question": question,
        "answer": answer,
        "chunks_retrieved": len(chunks),
        "overlap": overlap,
        "faithfulness_supported": faith["supported"],
        "faithfulness_reasoning": faith["reasoning"],
        "hallucination_rate": hall["hallucination_rate"],
        "hallucination_total_claims": hall["total_claims"],
        "hallucination_unsupported": hall["unsupported_count"],
        "verified": verified,
        "error": None,
    }


# ── report rendering ──────────────────────────────────────────────────────────


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2%}" if pct else f"{v:.3f}"


def _build_report(
    pair_results: list[dict],
    annotations: list[AnnotationFile],
    top_k: int,
    run_ts: str,
) -> dict:
    """Aggregate pair-level results into a structured report dict."""
    by_video: dict[str, dict] = {}
    for ann in annotations:
        by_video[ann.video_id] = {
            "video_id": ann.video_id,
            "video_type": ann.video_type or "unknown",
            "pairs": [],
        }

    for pr in pair_results:
        vid = pr.get("video_id", "")
        if vid in by_video:
            by_video[vid]["pairs"].append(pr)

    # Per-video aggregates.
    videos_agg = []
    for vid, entry in by_video.items():
        pairs = entry["pairs"]
        videos_agg.append(
            {
                "video_id": vid,
                "video_type": entry["video_type"],
                "n_pairs": len(pairs),
                "n_verified": sum(1 for p in pairs if p["verified"]),
                "avg_overlap": _mean([p["overlap"] for p in pairs]),
                "faithfulness_rate": _mean(
                    [
                        1.0 if p["faithfulness_supported"] else 0.0
                        for p in pairs
                        if p["faithfulness_supported"] is not None
                    ]
                ),
                "avg_hallucination_rate": _mean(
                    [
                        p["hallucination_rate"]
                        for p in pairs
                        if p["hallucination_rate"] is not None
                    ]
                ),
            }
        )

    # By video_type aggregate.
    by_type: dict[str, list[dict]] = defaultdict(list)
    for v in videos_agg:
        by_type[v["video_type"]].append(v)

    type_agg = {
        vtype: {
            "n_videos": len(vlist),
            "n_verified": sum(v["n_verified"] for v in vlist),
            "n_pairs": sum(v["n_pairs"] for v in vlist),
            "avg_overlap": _mean([v["avg_overlap"] for v in vlist]),
            "faithfulness_rate": _mean([v["faithfulness_rate"] for v in vlist]),
            "avg_hallucination_rate": _mean(
                [v["avg_hallucination_rate"] for v in vlist]
            ),
        }
        for vtype, vlist in by_type.items()
    }

    overall_pairs = pair_results
    overall = {
        "n_pairs": len(overall_pairs),
        "n_verified": sum(1 for p in overall_pairs if p["verified"]),
        "n_faith_hall": sum(
            1 for p in overall_pairs if p["faithfulness_supported"] is not None
        ),
        "avg_overlap": _mean([p["overlap"] for p in overall_pairs]),
        "faithfulness_rate": _mean(
            [
                1.0 if p["faithfulness_supported"] else 0.0
                for p in overall_pairs
                if p["faithfulness_supported"] is not None
            ]
        ),
        "avg_hallucination_rate": _mean(
            [
                p["hallucination_rate"]
                for p in overall_pairs
                if p["hallucination_rate"] is not None
            ]
        ),
    }

    return {
        "run_at": run_ts,
        "top_k": top_k,
        "overall": overall,
        "by_video_type": type_agg,
        "videos": videos_agg,
        "pairs": pair_results,
    }


def _render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Eval Report")
    lines.append(f"\n**Run:** `{report['run_at']}`  |  **top_k:** {report['top_k']}\n")

    # Overall
    o = report["overall"]
    n_v = o["n_verified"]
    n_fh = o.get("n_faith_hall", o["n_pairs"])
    overlap_cell = (
        "N/A — no verified QA pairs"
        if n_v == 0
        else f"{_fmt(o['avg_overlap'])} (n={n_v} verified pairs)"
    )
    lines.append("## Overall")
    lines.append("| metric | value |")
    lines.append("|--------|-------|")
    lines.append(f"| QA pairs evaluated (total) | {o['n_pairs']} |")
    lines.append(f"| — of which verified (used for overlap) | {n_v} |")
    lines.append(
        f"| — used for faithfulness / hallucination | {n_fh} (all pairs, incl. unverified) |"
    )
    lines.append(f"| Avg retrieval overlap (IoU) | {overlap_cell} |")
    lines.append(
        f"| Faithfulness rate (answer-level) | "
        f"{_fmt(o['faithfulness_rate'], pct=True)} (n={n_fh} pairs) |"
    )
    lines.append(
        f"| Avg hallucination rate (claim-level) | "
        f"{_fmt(o['avg_hallucination_rate'], pct=True)} (n={n_fh} pairs) |"
    )

    # By video_type
    lines.append("\n## By Video Type")
    lines.append(
        "| video_type | n_videos | overlap (n=verified) | faithful (n=total) | halluc (n=total) |"
    )
    lines.append(
        "|------------|----------|----------------------|--------------------|------------------|"
    )
    for vtype, agg in sorted(report["by_video_type"].items()):
        nv = agg.get("n_verified", 0)
        nt = agg.get("n_pairs", 0)
        ov = "N/A (0 verified)" if nv == 0 else f"{_fmt(agg['avg_overlap'])} (n={nv})"
        lines.append(
            f"| {vtype} | {agg['n_videos']} "
            f"| {ov} "
            f"| {_fmt(agg['faithfulness_rate'], pct=True)} (n={nt}) "
            f"| {_fmt(agg['avg_hallucination_rate'], pct=True)} (n={nt}) |"
        )

    # Per-video
    lines.append("\n## Per Video")
    lines.append("| video_id | type | pairs | verified | overlap | faithful | halluc |")
    lines.append("|----------|------|-------|----------|---------|----------|--------|")
    for v in report["videos"]:
        lines.append(
            f"| {v['video_id'][:16]}… | {v['video_type']} | {v['n_pairs']} "
            f"| {v['n_verified']} "
            f"| {_fmt(v['avg_overlap'])} "
            f"| {_fmt(v['faithfulness_rate'], pct=True)} "
            f"| {_fmt(v['avg_hallucination_rate'], pct=True)} |"
        )

    lines.append(
        "\n---\n"
        "*overlap scored on **verified** pairs only — unverified ground-truth timestamps "
        "come from retriever citations (circular), so are excluded from retrieval metrics.*\n"
        "*faithfulness and hallucination use **all** pairs (no ground-truth timestamp needed).*"
    )
    return "\n".join(lines) + "\n"


# ── main eval loop ────────────────────────────────────────────────────────────


def run_eval(dataset_dir: Path, output_dir: Path, top_k: int) -> None:
    annotations = load_annotations(dataset_dir)
    # Skip the _example.json (fake video_id all-zeros)
    annotations = [
        a for a in annotations if a.video_id != "00000000-0000-0000-0000-000000000000"
    ]

    if not annotations:
        safe_print(f"[run_eval] No annotation files found in {dataset_dir}")
        return

    safe_print(f"[run_eval] {len(annotations)} video(s), top_k={top_k}")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()
    pair_results: list[dict] = []

    for ann in annotations:
        video_id = ann.video_id
        vtype = ann.video_type or "unknown"
        safe_print(
            f"\n[run_eval] video={video_id[:16]}... type={vtype}  pairs={len(ann.qa_pairs)}"
        )

        unverified = sum(1 for p in ann.qa_pairs if not p.verified)
        if unverified:
            safe_print(
                f"  [info] {unverified} pair(s) not verified — overlap skipped for those"
            )

        for i, pair in enumerate(ann.qa_pairs, 1):
            safe_print(f"  [{i}/{len(ann.qa_pairs)}] {pair.question[:60]!r}")
            try:
                result = _eval_pair(
                    video_id=video_id,
                    question=pair.question,
                    gt_ranges=pair.ground_truth_timestamps,
                    top_k=top_k,
                    verified=pair.verified,
                )
            except Exception as exc:
                safe_print(f"    [error] {exc}")
                result = {
                    "question": pair.question,
                    "answer": None,
                    "chunks_retrieved": 0,
                    "overlap": None,
                    "faithfulness_supported": None,
                    "faithfulness_reasoning": None,
                    "hallucination_rate": None,
                    "hallucination_total_claims": None,
                    "hallucination_unsupported": None,
                    "verified": pair.verified,
                    "error": str(exc),
                }
            result["video_id"] = video_id
            result["video_type"] = vtype
            pair_results.append(result)

            faith = result.get("faithfulness_supported")
            hall = result.get("hallucination_rate")
            safe_print(
                f"    chunks={result['chunks_retrieved']}  "
                f"overlap={_fmt(result.get('overlap'))}  "
                f"faithful={'yes' if faith else 'no' if faith is not None else 'n/a'}  "
                f"halluc={_fmt(hall, pct=True)}"
            )

    report = _build_report(pair_results, annotations, top_k, run_ts)

    _log_mlflow(report, top_k)

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    safe_print(f"\n[run_eval] Report written to {json_path} and {md_path}")

    o = report["overall"]
    safe_print("\n-- Summary --")
    safe_print(f"  Pairs evaluated   : {o['n_pairs']}")
    safe_print(f"  Avg overlap (IoU) : {_fmt(o['avg_overlap'])}")
    safe_print(f"  Faithfulness rate : {_fmt(o['faithfulness_rate'], pct=True)}")
    safe_print(f"  Hallucination rate: {_fmt(o['avg_hallucination_rate'], pct=True)}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full offline evaluation: retrieve → answer → 3 metrics → report.",
    )
    parser.add_argument(
        "--dataset",
        default="eval/datasets",
        metavar="DIR",
        help="Directory containing *.json annotation files (default: eval/datasets).",
    )
    parser.add_argument(
        "--output",
        default="eval/reports",
        metavar="DIR",
        help="Directory to write report.json and report.md (default: eval/reports).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        metavar="N",
        help="Chunks to retrieve per question (default: 6).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_eval(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        top_k=args.top_k,
    )
