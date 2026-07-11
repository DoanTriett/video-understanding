"""eval/metrics/retrieval.py — Timestamp-overlap retrieval metric.

For each QA pair that has ground_truth_timestamps, this module:
  1. Calls retrieve() (real Qdrant call, no mock) for (video_id, question, top_k).
  2. For each ground-truth range, computes max IoU with any retrieved chunk.
  3. Averages max-IoU across all GT ranges  →  per-pair overlap score.
  4. Aggregates mean overlap per video_type and overall.

Only pairs with at least one ground_truth_timestamp are scored.
Pairs with an empty ground_truth_timestamps list are skipped and reported.

Usage (from repo root):
    python eval/metrics/retrieval.py
    python eval/metrics/retrieval.py --dataset-dir eval/datasets --top-k 6
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# ── make backend imports available ────────────────────────────────────────────
_BACKEND = Path(__file__).parent.parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.pipeline.shared.retriever import retrieve  # noqa: E402

# ── eval imports ───────────────────────────────────────────────────────────────
_EVAL_ROOT = Path(__file__).parent.parent.parent
if str(_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_EVAL_ROOT))

from eval.loader import load_annotations  # noqa: E402
from eval.schema import AnnotationFile, GroundTruthTimestamp  # noqa: E402


# ── core metric ────────────────────────────────────────────────────────────────


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection-over-union for two 1-D intervals [a_start, a_end] ∩ [b_start, b_end]."""
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if intersection == 0.0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0.0 else 0.0


def compute_timestamp_overlap(
    retrieved_chunks: list[dict],
    ground_truth_ranges: list[GroundTruthTimestamp],
) -> float:
    """Mean of per-GT-range max-IoU across all retrieved chunks.

    For each ground-truth time range, finds the retrieved chunk with the highest
    IoU and records that score.  Then returns the mean across all GT ranges.

    Returns 0.0 when ground_truth_ranges is empty or retrieved_chunks is empty.

    Retrieved chunk fields used: ``start`` (float), ``end`` (float).
    Both come directly from retriever.retrieve() output.
    """
    if not ground_truth_ranges or not retrieved_chunks:
        return 0.0

    per_gt_scores: list[float] = []
    for gt in ground_truth_ranges:
        best = max(
            _iou(c["start"], c["end"], gt.start, gt.end) for c in retrieved_chunks
        )
        per_gt_scores.append(best)

    return sum(per_gt_scores) / len(per_gt_scores)


# ── evaluation loop ────────────────────────────────────────────────────────────


def run_retrieval_eval(dataset_dir: str | Path, top_k: int = 6) -> None:
    """Load annotations, call retrieve() for each QA pair, print overlap table."""
    annotations: list[AnnotationFile] = load_annotations(dataset_dir)

    if not annotations:
        print(f"[retrieval] No annotation files found in {dataset_dir}")
        return

    # bucket: video_type (or "unknown") -> list of overlap scores
    scores_by_type: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []

    skipped_not_verified = 0
    skipped_no_gt = 0
    total_pairs = 0

    print(f"\n[retrieval] Loaded {len(annotations)} file(s), top_k={top_k}\n")
    print(f"{'video_id':<38} {'question':<45} {'chunks':>6} {'overlap':>8}")
    print("-" * 102)

    for annotation in annotations:
        video_id = annotation.video_id
        vtype = annotation.video_type or "unknown"

        verified_pairs = [p for p in annotation.qa_pairs if p.verified]
        unverified_count = len(annotation.qa_pairs) - len(verified_pairs)
        if unverified_count:
            skipped_not_verified += unverified_count
            print(
                f"  [warn] {video_id}: {unverified_count} QA pair(s) skipped -- not verified"
            )

        for pair in verified_pairs:
            total_pairs += 1

            if not pair.ground_truth_timestamps:
                skipped_no_gt += 1
                continue

            # Real retrieve() call — verified by presence of chunk_id in output.
            chunks = retrieve(video_id, pair.question, top_k)

            overlap = compute_timestamp_overlap(chunks, pair.ground_truth_timestamps)
            scores_by_type[vtype].append(overlap)
            all_scores.append(overlap)

            q_raw = (
                pair.question[:43] + ".." if len(pair.question) > 45 else pair.question
            )
            # Encode to the console's charset, replacing unencodable chars, for display only.
            q_short = q_raw.encode(
                sys.stdout.encoding or "utf-8", errors="replace"
            ).decode(sys.stdout.encoding or "utf-8", errors="replace")
            print(f"{video_id:<38} {q_short:<45} {len(chunks):>6} {overlap:>8.3f}")

    # ── summary ───────────────────────────────────────────────────────────────
    print("-" * 102)
    print(
        f"\n[retrieval] Summary  "
        f"(scored={len(all_scores)}, skipped_not_verified={skipped_not_verified}, "
        f"skipped_no_gt={skipped_no_gt}, verified_total={total_pairs})\n"
    )

    if all_scores:
        print(f"  {'video_type':<15} {'n':>4}  {'mean_overlap':>12}")
        print(f"  {'-'*36}")
        for vtype, scores in sorted(scores_by_type.items()):
            mean = sum(scores) / len(scores)
            print(f"  {vtype:<15} {len(scores):>4}  {mean:>12.3f}")
        overall = sum(all_scores) / len(all_scores)
        print(f"  {'OVERALL':<15} {len(all_scores):>4}  {overall:>12.3f}")
    else:
        print(
            "  No scored pairs -- either no verified=true pairs exist, "
            "or all verified pairs have empty ground_truth_timestamps."
        )


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute timestamp-overlap retrieval metric against eval/datasets/.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).parent.parent / "datasets"),
        metavar="DIR",
        help="Directory containing *.json annotation files (default: eval/datasets/).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        metavar="N",
        help="Number of chunks to retrieve per question (default: 6).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_retrieval_eval(dataset_dir=args.dataset_dir, top_k=args.top_k)
