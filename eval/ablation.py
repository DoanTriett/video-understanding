"""eval/ablation.py — Ablation study: compare pipeline configurations.

## Status of each ablation

### A. chunk_type filter  ← RUNS NOW, no re-processing needed
  Config "all"          : all retrieved chunks (speech + screen_share)
  Config "speech_only"  : filter retrieved chunks to chunk_type=="speech" only
  Uses existing Qdrant data; no pipeline re-run required.

### B. Diarization ablation  ← BLOCKED, needs pipeline change + GPU re-run
  run_meeting_pipeline() calls diarize() unconditionally — no skip flag.
  To unblock:
    1. Add `skip_diarization: bool = False` param to run_meeting_pipeline()
       in backend/app/pipeline/meeting/pipeline.py
    2. When True, skip Steps 1-2 and build speaker_turns from whisper_segments
       directly (speaker="UNKNOWN" for all turns).
    3. Re-process all meeting videos (est. ~30-45 min/video on RTX 4050).
  Effort: ~1h code + re-run time. DO NOT proceed without Tony's approval.

### C. OCR ablation  ← BLOCKED, needs pipeline change + GPU re-run
  run_lecture_pipeline() calls extract_slide_texts() unconditionally.
  To unblock:
    1. Add `skip_ocr: bool = False` param to run_lecture_pipeline()
       in backend/app/pipeline/lecture/pipeline.py
    2. When True, pass empty ocr_results to build_lecture_chunks().
    3. Re-process all lecture videos.
  Effort: ~1h code + re-run time. DO NOT proceed without Tony's approval.

### D. Fixed-duration chunking  ← BLOCKED, needs new code + Qdrant re-index
  chunk_fixed_duration() does not exist.  Meeting chunker uses speaker-turn
  strategy only; lecture chunker uses per-slide boundaries only.
  To unblock:
    1. Write chunk_fixed_duration() here (no production change needed).
    2. Re-embed chunks using text_embedder and re-index into a separate Qdrant
       collection (e.g. chunks_text_fixed_30s) to avoid polluting the live index.
    3. Run ablation against the new collection.
  Effort: ~2h code + re-index time. Requires Tony approval to re-use GPU.

Usage (from repo root):
    python eval/ablation.py --dataset eval/datasets [--output eval/reports]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from eval.schema import AnnotationFile  # noqa: E402


# ── ablation A: chunk_type filter ─────────────────────────────────────────────


def _eval_config(
    video_id: str,
    question: str,
    top_k: int,
    chunk_type_filter: str | None,
) -> dict | None:
    """Retrieve → (optionally filter by chunk_type) → answer → judge.

    Returns None if no chunks remain after filtering.
    """
    chunks = retrieve(video_id, question, top_k)
    if not chunks:
        return None

    if chunk_type_filter:
        chunks = [c for c in chunks if c.get("chunk_type") == chunk_type_filter]
    if not chunks:
        return None

    context = build_context(chunks)
    answer = generate_answer(question, context)
    faith = judge_faithfulness(answer, context)
    hall = compute_hallucination_rate(answer, context)
    return {
        "n_chunks": len(chunks),
        "faithfulness": 1.0 if faith["supported"] else 0.0,
        "hallucination_rate": hall["hallucination_rate"],
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2%}" if pct else f"{v:.3f}"


# ── main ───────────────────────────────────────────────────────────────────────


def run_ablation(dataset_dir: Path, output_dir: Path, top_k: int) -> None:
    annotations: list[AnnotationFile] = load_annotations(dataset_dir)
    annotations = [
        a for a in annotations if a.video_id != "00000000-0000-0000-0000-000000000000"
    ]

    if not annotations:
        safe_print(f"[ablation] No annotation files found in {dataset_dir}")
        return

    configs = {
        "all_chunks": None,  # no filter
        "speech_only": "speech",  # meeting only — filters out screen_share
    }

    # Collect per-pair results for each config.
    results: dict[str, list[dict]] = {k: [] for k in configs}

    total_pairs = sum(len(a.qa_pairs) for a in annotations)
    safe_print(
        f"\n[ablation] chunk_type filter ablation — "
        f"{len(annotations)} video(s), {total_pairs} pairs, top_k={top_k}\n"
    )

    for ann in annotations:
        for pair in ann.qa_pairs:
            safe_print(f"  video={ann.video_id[:8]}  q={pair.question[:55]!r}")
            for cfg_name, chunk_filter in configs.items():
                r = _eval_config(ann.video_id, pair.question, top_k, chunk_filter)
                if r is None:
                    safe_print(f"    [{cfg_name}] no chunks — skipped")
                    continue
                results[cfg_name].append(r)
                safe_print(
                    f"    [{cfg_name}] chunks={r['n_chunks']}  "
                    f"faithful={'yes' if r['faithfulness'] else 'no'}  "
                    f"halluc={_fmt(r['hallucination_rate'], pct=True)}"
                )

    # ── comparison table ───────────────────────────────────────────────────────
    safe_print("\n-- Ablation A: chunk_type filter ---------------------------------")
    safe_print(f"  {'config':<15} {'n':>4}  {'faithfulness':>13}  {'halluc_rate':>12}")
    safe_print("  " + "-" * 50)
    for cfg_name, rows in results.items():
        n = len(rows)
        faith = _mean([r["faithfulness"] for r in rows])
        hall = _mean([r["hallucination_rate"] for r in rows])
        safe_print(
            f"  {cfg_name:<15} {n:>4}  {_fmt(faith, pct=True):>13}  {_fmt(hall, pct=True):>12}"
        )

    # ── blocked ablations summary ──────────────────────────────────────────────
    safe_print("\n-- Blocked ablations (need pipeline changes + GPU re-run) --------")
    safe_print("  B. Diarization ablation:")
    safe_print(
        "     Action needed : add skip_diarization flag to run_meeting_pipeline()"
    )
    safe_print(
        "     Then          : re-process meeting videos (~30-45 min/video on RTX 4050)"
    )
    safe_print("     Status        : BLOCKED — awaiting Tony approval")
    safe_print("  C. OCR ablation:")
    safe_print("     Action needed : add skip_ocr flag to run_lecture_pipeline()")
    safe_print("     Then          : re-process lecture videos")
    safe_print("     Status        : BLOCKED — awaiting Tony approval")
    safe_print("  D. Fixed-duration chunking:")
    safe_print(
        "     Action needed : write chunk_fixed_duration() + re-index Qdrant collection"
    )
    safe_print("     Status        : BLOCKED — awaiting Tony approval for GPU re-run")

    # ── write report ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "ablation_A_chunk_type_filter": {
            cfg: {
                "n": len(rows),
                "faithfulness_rate": _mean([r["faithfulness"] for r in rows]),
                "avg_hallucination_rate": _mean(
                    [r["hallucination_rate"] for r in rows]
                ),
            }
            for cfg, rows in results.items()
        },
        "ablation_B_diarization": "BLOCKED — no skip_diarization flag in pipeline",
        "ablation_C_ocr": "BLOCKED — no skip_ocr flag in pipeline",
        "ablation_D_fixed_duration_chunking": "BLOCKED — no chunk_fixed_duration() + re-index needed",
    }
    json_path = output_dir / "ablation.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = _render_md(results, configs)
    md_path = output_dir / "ablation.md"
    md_path.write_text(md, encoding="utf-8")

    safe_print(f"\n[ablation] Report written to {json_path} and {md_path}")


def _render_md(results: dict, configs: dict) -> str:
    lines = ["# Ablation Report\n"]
    lines.append("## A. chunk_type filter (runs on existing data)\n")
    lines.append("| config | n | faithfulness_rate | avg_hallucination_rate |")
    lines.append("|--------|---|-------------------|------------------------|")
    for cfg_name, rows in results.items():
        n = len(rows)
        faith = _mean([r["faithfulness"] for r in rows])
        hall = _mean([r["hallucination_rate"] for r in rows])
        lines.append(
            f"| {cfg_name} | {n} | {_fmt(faith, pct=True)} | {_fmt(hall, pct=True)} |"
        )

    lines.append("")
    lines.append("## B. Diarization ablation — BLOCKED")
    lines.append(
        "Requires adding `skip_diarization` flag to `run_meeting_pipeline()` "
        "and re-processing meeting videos on GPU (~30–45 min/video)."
    )
    lines.append("")
    lines.append("## C. OCR ablation — BLOCKED")
    lines.append(
        "Requires adding `skip_ocr` flag to `run_lecture_pipeline()` "
        "and re-processing lecture videos on GPU."
    )
    lines.append("")
    lines.append("## D. Fixed-duration chunking — BLOCKED")
    lines.append(
        "Requires writing `chunk_fixed_duration()` + re-indexing "
        "a separate Qdrant collection. No production code change needed, "
        "but GPU re-run required for re-embedding."
    )
    lines.append("")
    lines.append("---")
    lines.append(
        "*Ablation A runs on already-processed data in Qdrant/Postgres. "
        "Ablations B–D each require a full GPU pipeline re-run.*"
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study for the Video Understanding QA pipeline.",
    )
    parser.add_argument("--dataset", default="eval/datasets", metavar="DIR")
    parser.add_argument("--output", default="eval/reports", metavar="DIR")
    parser.add_argument("--top-k", type=int, default=6, metavar="N")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ablation(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.output),
        top_k=args.top_k,
    )
