# Eval Report

**Run:** `2026-07-13T19:22:22.887203+00:00`  |  **top_k:** 6

## Overall
| metric | value |
|--------|-------|
| QA pairs evaluated (total) | 20 |
| — of which verified (used for overlap) | 0 |
| — used for faithfulness / hallucination | 20 (all pairs, incl. unverified) |
| Avg retrieval overlap (IoU) | N/A — no verified QA pairs |
| Faithfulness rate (answer-level) | 40.00% (n=20 pairs) |
| Avg hallucination rate (claim-level) | 44.32% (n=20 pairs) |

## By Video Type
| video_type | n_videos | overlap (n=verified) | faithful (n=total) | halluc (n=total) |
|------------|----------|----------------------|--------------------|------------------|
| unknown | 4 | N/A (0 verified) | 41.04% (n=20) | 43.46% (n=20) |

## Per Video
| video_id | type | pairs | verified | overlap | faithful | halluc |
|----------|------|-------|----------|---------|----------|--------|
| 1c5478f9-61f0-4f… | unknown | 4 | 0 | n/a | 0.00% | 79.17% |
| 2db6e69b-d88a-43… | unknown | 3 | 0 | n/a | 66.67% | 25.00% |
| 42982b9a-f6b0-47… | unknown | 5 | 0 | n/a | 60.00% | 20.93% |
| b1730449-7d27-4f… | unknown | 8 | 0 | n/a | 37.50% | 48.75% |

---
*overlap scored on **verified** pairs only — unverified ground-truth timestamps come from retriever citations (circular), so are excluded from retrieval metrics.*
*faithfulness and hallucination use **all** pairs (no ground-truth timestamp needed).*
