# Ablation Report

## A. chunk_type filter (runs on existing data)

| config | n | faithfulness_rate | avg_hallucination_rate |
|--------|---|-------------------|------------------------|
| all_chunks | 20 | 65.00% | 33.57% |
| speech_only | 12 | 16.67% | 60.52% |

## B. Diarization ablation — BLOCKED
Requires adding `skip_diarization` flag to `run_meeting_pipeline()` and re-processing meeting videos on GPU (~30–45 min/video).

## C. OCR ablation — BLOCKED
Requires adding `skip_ocr` flag to `run_lecture_pipeline()` and re-processing lecture videos on GPU.

## D. Fixed-duration chunking — BLOCKED
Requires writing `chunk_fixed_duration()` + re-indexing a separate Qdrant collection. No production code change needed, but GPU re-run required for re-embedding.

---
*Ablation A runs on already-processed data in Qdrant/Postgres. Ablations B–D each require a full GPU pipeline re-run.*
