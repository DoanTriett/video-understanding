# Eval Report

**Run:** `2026-07-09T18:47:37.492356+00:00`  |  **top_k:** 6

## Overall
| metric | value |
|--------|-------|
| QA pairs evaluated (total) | 20 |
| — of which verified (used for overlap) | 0 |
| — used for faithfulness / hallucination | 20 (all pairs, incl. unverified) |
| Avg retrieval overlap (IoU) | N/A — no verified QA pairs |
| Faithfulness rate (answer-level) | 50.00% (n=20 pairs) |
| Avg hallucination rate (claim-level) | 36.01% (n=20 pairs) |

## By Video Type
| video_type | n_videos | overlap (n=verified) | faithful (n=total) | halluc (n=total) |
|------------|----------|----------------------|--------------------|------------------|
| unknown | 4 | N/A (0 verified) | 57.50% (n=20) | 33.64% (n=20) |

## Per Video
| video_id | type | pairs | verified | overlap | faithful | halluc |
|----------|------|-------|----------|---------|----------|--------|
| 1c5478f9-61f0-4f… | unknown | 4 | 0 | n/a | 25.00% | 50.00% |
| 2db6e69b-d88a-43… | unknown | 3 | 0 | n/a | 100.00% | 16.67% |
| 42982b9a-f6b0-47… | unknown | 5 | 0 | n/a | 80.00% | 24.32% |
| b1730449-7d27-4f… | unknown | 8 | 0 | n/a | 25.00% | 43.57% |

---
*overlap scored on **verified** pairs only — unverified ground-truth timestamps come from retriever citations (circular), so are excluded from retrieval metrics.*
*faithfulness and hallucination use **all** pairs (no ground-truth timestamp needed).*
