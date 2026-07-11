"""eval/loader.py — Load and validate annotation files from eval/datasets/.

Usage:
    from eval.loader import load_annotations
    annotations = load_annotations("eval/datasets")   # raises ValidationError on bad files
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError  # noqa: F401  (re-exported for callers)

from eval.schema import AnnotationFile


def load_annotations(directory: str | Path) -> list[AnnotationFile]:
    """Load all *.json files in *directory* as AnnotationFile instances.

    Files are processed in alphabetical order.  On the first malformed file,
    pydantic.ValidationError is raised with full field-level details — no silent
    skipping.  Fix the file and re-run.

    Returns an empty list if the directory contains no *.json files.
    """
    path = Path(directory)
    results: list[AnnotationFile] = []
    for f in sorted(path.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        # ValidationError propagates to the caller — not caught here.
        results.append(AnnotationFile.model_validate(data))
    return results
