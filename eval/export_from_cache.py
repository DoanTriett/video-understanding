"""eval/export_from_cache.py — Export real Q&A pairs from the Redis semantic cache.

Cache structure (from backend/app/semantic_cache.py):
  - Per-video index:  sc:idx:{video_id}  →  Redis SET of entry keys
  - Per-entry key:    sc:{video_id}:{uuid}  →  JSON blob:
        {
            "question":   str,
            "embedding":  list[float],
            "answer":     str,
            "citations":  list[{chunk_id, speaker, start, end, text}],
            "created_at": str  (ISO-8601)
        }

The cache stores citations with start/end timestamps, so ground_truth_timestamps
can be fully populated from cache entries.

All exported pairs carry:
    video_type = None   (unknown without a DB lookup)

These files are valid AnnotationFile instances and load cleanly via eval/loader.py.
Manually set video_type and review expected_answer after export.

Usage:
    python eval/export_from_cache.py [--redis-url URL] [--output-dir DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import redis

# Allow running as `python eval/export_from_cache.py` from repo root.
sys.path.insert(0, str(Path(__file__).parent))
from schema import AnnotationFile, GroundTruthTimestamp, QAPair  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────────


def _extract_video_id(index_key: str) -> str:
    return index_key[len("sc:idx:") :]


def _build_qa_pair(entry: dict) -> QAPair:
    citations: list[dict] = entry.get("citations", [])
    timestamps = [
        GroundTruthTimestamp(
            start=float(c.get("start", 0.0)),
            end=float(c.get("end", 0.0)),
        )
        for c in citations
    ]
    return QAPair(
        question=entry["question"],
        expected_answer=entry["answer"],
        ground_truth_timestamps=timestamps,
        notes=None,
    )


# ── main logic ─────────────────────────────────────────────────────────────────


def export(redis_url: str, output_dir: Path, dry_run: bool) -> None:
    client = redis.from_url(redis_url, decode_responses=True)

    try:
        client.ping()
    except redis.exceptions.ConnectionError as exc:
        print(f"[error] Cannot connect to Redis at {redis_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    import json

    index_keys: list[str] = list(client.scan_iter("sc:idx:*"))
    if not index_keys:
        print("[export] No semantic cache entries found. Nothing to export.")
        return

    print(f"[export] Found {len(index_keys)} video(s) with cached Q&A.")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    total_with_citations = 0
    total_without_citations = 0

    for idx_key in sorted(index_keys):
        video_id = _extract_video_id(idx_key)
        entry_keys: set[str] = client.smembers(idx_key)

        pairs: list[QAPair] = []
        stale_keys: list[str] = []

        for ek in sorted(entry_keys):
            raw = client.get(ek)
            if raw is None:
                stale_keys.append(ek)
                continue
            entry: dict = json.loads(raw)
            pair = _build_qa_pair(entry)
            pairs.append(pair)

            citations = entry.get("citations", [])
            if citations:
                total_with_citations += 1
            else:
                total_without_citations += 1
                print(
                    f"  [warn] video={video_id} "
                    f"question={pair.question[:60]!r} "
                    "-- cache entry has no citations (ground_truth_timestamps empty)"
                )

        if stale_keys:
            print(
                f"  [info] {len(stale_keys)} stale key(s) skipped for video={video_id}"
            )

        if not pairs:
            print(f"  [skip] video={video_id}: no live entries.")
            continue

        annotation = AnnotationFile(
            video_id=video_id,
            video_type=None,  # unknown without DB lookup; set manually after review
            qa_pairs=pairs,
        )

        out_path = output_dir / f"{video_id}.json"
        total_pairs += len(pairs)

        if dry_run:
            print(f"  [dry-run] would write {out_path} ({len(pairs)} pair(s))")
        else:
            out_path.write_text(
                annotation.model_dump_json(indent=2),
                encoding="utf-8",
            )
            print(f"  [ok] {out_path} -- {len(pairs)} pair(s)")

    print()
    print("-- Export summary --------------------------------------------------")
    print(f"  Videos exported  : {len(index_keys)}")
    print(f"  Total QA pairs   : {total_pairs}")
    print(f"  With citations   : {total_with_citations}")
    print(f"  Without citations: {total_without_citations}")
    if total_without_citations:
        print(
            "  [note] Pairs without citations have empty ground_truth_timestamps.\n"
            "         Safe for faithfulness metrics; not for retrieval overlap metrics."
        )
    print(f"  Exported at      : {datetime.now(timezone.utc).isoformat()}")
    print("  video_type=null in all files -- set manually after review.")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cached Q&A pairs from Redis semantic cache to eval/datasets/.",
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        metavar="URL",
        help="Redis connection URL (default: redis://localhost:6379/0).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "datasets"),
        metavar="DIR",
        help="Directory to write {video_id}.json files (default: eval/datasets/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(
        redis_url=args.redis_url,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
