# eval/ — Offline Evaluation for the Video Understanding QA Pipeline

## Goal

Measure the end-to-end quality of the QA pipeline without a live server:

- **Retrieval quality** — hit-rate@k, MRR (requires golden chunk ids in the dataset)
- **Answer quality** — exact match, token F1, semantic similarity (BERTScore or cosine sim)

## Directory layout

```
eval/
├── datasets/          # Evaluation datasets (JSONL); committed only as .gitkeep
│   └── .gitkeep
├── run_eval.py        # Main harness (see --help)
├── schema.py          # TODO: dataset schema definition (see below)
└── README.md          # This file
```

## How to run

```bash
python eval/run_eval.py \
    --dataset eval/datasets/sample.jsonl \
    --output  eval/results/run_001.json \
    --top-k   6
```

`--help` shows all options.

## Dataset schema

TODO: schema to be defined in `eval/schema.py` (next step).

Each line of the JSONL file will be a JSON object.  Placeholder fields:

```jsonc
{
  "video_id": "...",       // matches a video already processed in the DB
  "question": "...",
  "reference_answer": "...",
  "golden_chunk_ids": []   // optional; used for retrieval hit-rate/MRR
}
```

## Pipeline integration points

The harness calls the same functions used by the `/ask` endpoint:

| Component | Function | Location |
|-----------|----------|----------|
| Retrieval | `retrieve(video_id, question, top_k)` | `backend/app/pipeline/shared/retriever.py` |
| Context formatting | `build_context(chunks)` | `backend/app/pipeline/shared/retriever.py` |
| Answer generation | `generate_answer(question, context)` | `backend/app/llm.py` |

Retrieved chunks have the following fields: `chunk_id`, `speaker`, `start`, `end`,
`text`, `chunk_type`, `score`.

## Notes

- MLflow integration is deferred (Phase 4 of the end-to-end roadmap).
- Do **not** install `mlflow`, `jiwer`, or `pyannote.metrics` for this phase.
- `run_eval.py` and `ablation.py` call `retrieve()` / `generate_answer()` directly
  in-process (not via the `/ask` HTTP endpoint), so the API's rate limiter never
  actually sees these calls. Both scripts still set `TESTING=true` before any
  backend import as a defensive default in case that changes later. This means
  eval results reflect the pipeline's raw throughput, not the `/ask` rate limit
  a real client would experience in production (10 req/min by default).
