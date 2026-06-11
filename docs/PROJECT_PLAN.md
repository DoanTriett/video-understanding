# Video Understanding System — End-to-End Implementation Roadmap

> A step-by-step roadmap to evolve the current FastAPI + Celery pipeline into a real,
> observable, reproducible product with a full MLOps stack.
>
> Tooling: **MLflow** (experiments + model registry), **GitHub Actions** (CI/CD),
> **Prometheus + Grafana** (metrics), **Loki** (logs), **OpenTelemetry + Tempo** (traces),
> **MinIO** (object storage), **PostgreSQL** (metadata), **Qdrant** (vectors),
> **Redis** (broker/cache), **Next.js** (frontend), **Docker Compose** (local),
> **Kubernetes + Helm** (cloud).

---

## 0. Current State (baseline)

| Area | What exists today |
|------|-------------------|
| API | `backend/app/main.py`, `backend/app/api/videos.py` (upload / status / transcript / chunks) |
| Async | `backend/workers/tasks.py`, `backend/workers/celery_app.py` (Celery + Redis) |
| Pipeline | faster-whisper `large-v3`, pyannote `speaker-diarization-3.1`, CLIP `ViT-B/32`, merge/chunk/screen-detect |
| State | Redis used as both broker and pseudo-DB (`backend/app/store.py`) |
| Files | Local disk (`settings.upload_dir`) |
| Infra | `docker-compose.yml` runs only Redis + Qdrant |
| Missing | RAG/QA serving, frontend, tests, MLflow, CI/CD, metrics/logs/traces, object storage, metadata DB, secrets mgmt |

**Target repo layout after this roadmap:**

```
video-understanding/
├── backend/
│   ├── app/
│   │   ├── api/            # videos.py, qa.py, health.py
│   │   ├── pipeline/       # shared/ + meeting/ (+ indexer, text_embedder)
│   │   ├── db/             # SQLAlchemy models, session, migrations
│   │   ├── storage.py      # MinIO/S3 abstraction
│   │   ├── observability/  # logging, metrics, tracing setup
│   │   └── config.py
│   ├── workers/
│   ├── ml/eval/            # WER / DER / retrieval eval + MLflow logging
│   ├── tests/
│   ├── Dockerfile
│   ├── Dockerfile.worker
│   └── pyproject.toml
├── frontend/               # Next.js app
├── infra/                  # prometheus, grafana, loki, tempo, otel configs
├── deploy/
│   ├── helm/video-understanding/
│   └── terraform/          # optional
├── .github/workflows/      # ci, build, eval, deploy
├── docker-compose.yml      # full local stack
└── docs/                   # PROJECT_PLAN.md, RUNBOOK.md
```

---

## Phase 1 — Foundations & Hardening

**Goal:** secure secrets, set up tooling, move durable state to Postgres, move files to MinIO.

### Step 1.1 — Remove hardcoded secrets
1. In `backend/app/config.py`, delete the hardcoded `huggingface_token` default value. Make it required from env:
   ```python
   huggingface_token: str = ""   # MUST come from env, never commit a real value
   openai_api_key: str = ""
   ```
2. Create `backend/.env.example` listing every variable (no real values):
   ```
   REDIS_URL=redis://localhost:6379/0
   POSTGRES_URL=postgresql+psycopg://vu:vu@localhost:5432/vu
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   MINIO_ENDPOINT=localhost:9000
   MINIO_ACCESS_KEY=minioadmin
   MINIO_SECRET_KEY=minioadmin
   MINIO_BUCKET=videos
   MLFLOW_TRACKING_URI=http://localhost:5000
   HUGGINGFACE_TOKEN=
   OPENAI_API_KEY=
   DEVICE=cuda
   ```
3. Add `.gitignore` entries: `.env`, `uploads/`, `mlruns/`, `__pycache__/`, `*.pt`, `venv/`.
4. **Rotate the leaked HF token** in your Hugging Face account (the old one is in git history).

### Step 1.2 — Dev tooling
1. Create `backend/pyproject.toml` with `ruff`, `mypy`, `pytest`, `pytest-cov` config.
2. Add dev deps to a `backend/requirements-dev.txt`: `ruff`, `mypy`, `pytest`, `pytest-cov`, `pytest-asyncio`, `httpx`, `pre-commit`, `types-redis`.
3. Add `.pre-commit-config.yaml` at repo root (hooks: ruff, ruff-format, mypy, end-of-file-fixer).
4. Run `pre-commit install`.

### Step 1.3 — PostgreSQL metadata layer
1. Add deps: `sqlalchemy>=2`, `psycopg[binary]`, `alembic`.
2. Create `backend/app/db/session.py` (engine + `SessionLocal` from `settings.postgres_url`).
3. Create `backend/app/db/models.py` with tables: `videos` (id, filename, type, status, progress, error, created_at, object_key), `chunks` (id, video_id FK, speaker, text, start, end, chunk_type), optionally `jobs`.
4. `alembic init backend/alembic`; generate first migration; run `alembic upgrade head`.
5. Refactor `backend/app/store.py`: keep Redis only for live progress/cache; write the source-of-truth rows to Postgres. Update `videos.py` + `tasks.py` to read/write via the DB session.

### Step 1.4 — MinIO object storage
1. Add dep: `boto3` (S3-compatible client works with MinIO).
2. Create `backend/app/storage.py` with `upload_fileobj`, `download_to_path`, `presigned_url` using MinIO endpoint/keys from settings.
3. In `videos.py` upload handler: stream the uploaded file to MinIO (`object_key = f"{video_id}/source{ext}"`) instead of `aiofiles` local write; store `object_key` in Postgres.
4. In `tasks.py`: download the object to a temp `work_dir` at the start, upload artifacts (transcript.json, chunks.json, frames) back to MinIO.

**Phase 1 done when:** secrets come only from env, `alembic upgrade head` works, an upload lands in MinIO and a row appears in Postgres.

---

## Phase 2 — Complete the Product (RAG / QA Serving)

**Goal:** index chunks into Qdrant and add an `/ask` endpoint that answers with citations.

### Step 2.1 — Text embedder
1. Add dep: `sentence-transformers`.
2. Create `backend/app/pipeline/shared/text_embedder.py` with a lazy-loaded `BAAI/bge-small-en-v1.5` model and `embed_text(text) -> list[float]` (384-dim) + `embed_texts(list)` batch.

### Step 2.2 — Qdrant indexing
1. Create `backend/app/pipeline/shared/indexer.py`:
   - Ensure two collections exist: `chunks_text` (size 384, cosine) and `chunks_visual` (size 512, cosine).
   - `index_chunks(video_id, chunks)`: embed each chunk's text; upsert points with payload `{video_id, chunk_id, speaker, start, end, text, chunk_type}`. For `screen_share` chunks, also upsert the CLIP vector into `chunks_visual`.
2. Call `index_chunks(...)` at the end of `run_meeting_pipeline` (or in `tasks.py` after the pipeline returns).

### Step 2.3 — Retrieval + answer endpoint
1. Create `backend/app/api/qa.py` with `POST /videos/{video_id}/ask` taking `{ "question": str, "top_k": int = 6 }`.
2. Logic:
   - Embed the question (text embedder); also CLIP-embed for visual search.
   - Qdrant search in `chunks_text` (filter by `video_id`) and `chunks_visual`; merge + dedupe by `chunk_id` (hybrid).
   - Build a context string with `[speaker @ mm:ss]` prefixes.
   - Call the LLM (see 2.4) to produce an answer that cites timestamps.
   - Return `{ answer, citations: [{chunk_id, speaker, start, end, text}] }`.
3. Register the router in `main.py`.

### Step 2.4 — LLM provider abstraction
1. Add dep: `openai`.
2. Create `backend/app/llm.py` with an interface `generate_answer(question, context) -> str` and an OpenAI implementation (model + key from settings). Keep it pluggable so a local model can be swapped later.

**Phase 2 done when:** after a video finishes, `POST /videos/{id}/ask` returns a grounded answer with timestamp citations.

---

## Phase 3 — Frontend (Next.js)

**Goal:** a usable UI for upload, progress, transcript, and chat.

### Step 3.1 — Scaffold
1. `npx create-next-app@latest frontend --ts --tailwind --app --eslint`.
2. Add `frontend/.env.local`: `NEXT_PUBLIC_API_URL=http://localhost:8000`.

### Step 3.2 — Pages/components
1. **Upload page**: drag-drop file + video-type select → `POST /videos/upload`.
2. **Job view**: poll `GET /videos/{id}/status` every ~2s, show a progress bar.
3. **Transcript viewer**: render segments from `GET /videos/{id}/transcript` with clickable timestamps.
4. **Chat panel**: send questions to `POST /videos/{id}/ask`; render answer + citation chips that jump to the timestamp.
5. Create a small `frontend/lib/api.ts` client wrapper.

### Step 3.3 — Containerize
1. Add `frontend/Dockerfile` (multi-stage: build → `next start`).

**Phase 3 done when:** you can upload, watch progress, read the transcript, and chat — all in the browser.

---

## Phase 4 — MLflow (Experiments + Model Registry)

**Goal:** reproducible quality evaluation and governed model versions.

### Step 4.1 — Tracking server
1. Run MLflow with a Postgres backend store and MinIO artifact store (added to compose in Phase 6). Set `MLFLOW_TRACKING_URI` in env.

### Step 4.2 — Eval datasets
1. Create `backend/ml/eval/datasets/` with a few short clips + ground-truth: reference transcripts (for WER), RTTM speaker labels (for DER), and a Q→relevant-chunk file (for retrieval).

### Step 4.3 — Eval scripts (log to MLflow)
1. Add deps: `mlflow`, `jiwer`, `pyannote.metrics`.
2. `backend/ml/eval/eval_asr.py`: run transcriber on clips → compute **WER** with `jiwer` → `mlflow.log_metric("wer", ...)` + params (model name, compute_type).
3. `backend/ml/eval/eval_diarization.py`: run diarizer → compute **DER** with `pyannote.metrics` → log.
4. `backend/ml/eval/eval_retrieval.py`: index eval set → run queries → compute **recall@k / MRR** → log.
5. Wrap each in `with mlflow.start_run(run_name=...)`, tag the git SHA.

### Step 4.4 — Model registry
1. Register the chosen model identifiers/versions (whisper / pyannote / embedder) in the MLflow Model Registry; promote a version to stage `Production`.
2. Read the active model name/version from config so a swap is tracked and reproducible.

**Phase 4 done when:** `python -m ml.eval.eval_asr` (etc.) produces runs visible in the MLflow UI with WER/DER/retrieval metrics.

---

## Phase 5 — Observability (Metrics, Logs, Traces)

**Goal:** see latency, errors, throughput, logs, and end-to-end traces.

### Step 5.1 — Metrics
1. Add deps: `prometheus-fastapi-instrumentator`, `celery-exporter` (run as a service).
2. In `main.py`, instrument FastAPI and expose `/metrics`.
3. Add custom metrics in `backend/app/observability/metrics.py`: `jobs_in_progress` (Gauge), `pipeline_stage_seconds` (Histogram, label=stage), `job_failures_total` (Counter). Emit them from `tasks.py` around each stage.

### Step 5.2 — Logs (Loki)
1. Switch to structured JSON logging (`structlog` or stdlib `logging` + JSON formatter) in `backend/app/observability/logging.py`.
2. Promtail (compose/k8s) ships container logs → Loki. Include `trace_id` in log records.

### Step 5.3 — Traces (OpenTelemetry → Tempo)
1. Add deps: `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-celery`, `opentelemetry-exporter-otlp` (several already vendored in `venv`).
2. `backend/app/observability/tracing.py`: configure the OTLP exporter to the OTel Collector → Tempo. Auto-instrument FastAPI + Celery; add manual spans per pipeline stage.

### Step 5.4 — Dashboards & alerts
1. Provision Grafana datasources (Prometheus, Loki, Tempo) and dashboards under `infra/grafana/`: API latency/error rate, Celery throughput, pipeline stage timings, queue depth.
2. Prometheus alert rules (`infra/prometheus/alerts.yml`): high job failure rate, stuck jobs (no progress > N min), worker down.

**Phase 5 done when:** Grafana shows API/worker metrics, logs are searchable in Loki, and a single upload produces an end-to-end trace in Tempo.

---

## Phase 6 — Containerization & Local Orchestration

**Goal:** one command brings up the entire stack locally.

### Step 6.1 — Dockerfiles
1. `backend/Dockerfile` — API (python slim, install reqs, `uvicorn app.main:app`).
2. `backend/Dockerfile.worker` — GPU base (e.g. `nvidia/cuda` runtime) + ffmpeg, runs `celery -A workers.celery_app worker`.
3. `frontend/Dockerfile` (from Phase 3).

### Step 6.2 — Full docker-compose
Expand `docker-compose.yml` to include: `api`, `worker`, `frontend`, `redis`, `qdrant`, `postgres`, `minio`, `mlflow`, `prometheus`, `grafana`, `loki`, `promtail`, `tempo`, `otel-collector`, `celery-exporter`. Add healthchecks and `depends_on`.

### Step 6.3 — Infra config
Create `infra/` with: `prometheus/prometheus.yml`, `prometheus/alerts.yml`, `grafana/provisioning/...` + dashboards, `loki/loki-config.yml`, `promtail/promtail-config.yml`, `tempo/tempo.yml`, `otel/otel-collector-config.yml`.

**Phase 6 done when:** `docker compose up` starts everything and the frontend can drive a full upload→ask flow with dashboards live.

---

## Phase 7 — CI/CD (GitHub Actions)

**Goal:** automated quality gates and image delivery.

### Step 7.1 — CI (`.github/workflows/ci.yml`)
- Triggers: pull_request.
- Jobs: `ruff check` + `ruff format --check`, `mypy`, `pytest --cov` (spin up Redis/Qdrant as services), frontend `npm ci && npm run build && npm test`.

### Step 7.2 — Build & push (`.github/workflows/build.yml`)
- Triggers: push to `main`, tags `v*`.
- Build api/worker/frontend images, push to **GHCR** tagged with SHA + semver.
- Run **Trivy** image scan; fail on HIGH/CRITICAL.

### Step 7.3 — Eval gate (`.github/workflows/eval.yml`)
- Trigger: pull_request (manual `workflow_dispatch` too).
- Run `ml/eval/*` against the small labeled set, log to MLflow, and post WER/DER/retrieval as a PR comment; optionally fail if metrics regress beyond a threshold.

### Step 7.4 — Deploy (`.github/workflows/deploy.yml`)
- Trigger: release tag, with a protected `production` environment (manual approval).
- `helm upgrade --install` against the cluster using the new image tags.

**Phase 7 done when:** a PR runs lint/type/test + eval automatically, and merging to a tag publishes images and (optionally) deploys.

---

## Phase 8 — Kubernetes + Helm

**Goal:** cloud-grade deployment.

### Step 8.1 — Helm chart
1. `deploy/helm/video-understanding/` umbrella chart.
2. Templates: Deployments for `api`, `worker` (GPU `nodeSelector` + `resources.limits."nvidia.com/gpu": 1`, CPU fallback via values), `frontend`; Services; ConfigMaps; HPA for api/worker.
3. Dependencies (`Chart.yaml`): Redis, PostgreSQL, Qdrant, MinIO, MLflow, `kube-prometheus-stack`, Loki, Tempo.

### Step 8.2 — Secrets
1. Use **external-secrets** (or **sealed-secrets**) for HF/OpenAI/DB/MinIO credentials — never plain secrets in git.

### Step 8.3 — Networking & TLS
1. Ingress (nginx) routing `/` → frontend and `/api` → API.
2. **cert-manager** for TLS certificates.

### Step 8.4 — Observability wiring
1. `ServiceMonitor`s so Prometheus scrapes api/worker `/metrics`.
2. Grafana dashboards as ConfigMaps (auto-loaded by the sidecar).

### Step 8.5 — Local k8s + optional IaC
1. Document `kind`/`minikube` bring-up in the runbook.
2. Optional `deploy/terraform/` for managed cluster + object-store buckets.

**Phase 8 done when:** `helm upgrade --install` brings the full app up on a cluster with TLS, autoscaling, and dashboards.

---

## Phase 9 — Tests & Docs

**Goal:** confidence and onboarding.

### Step 9.1 — Tests (`backend/tests/`)
1. Unit: `merge_transcript_with_diarization`, `chunk_meeting`, indexer payload building.
2. Contract: FastAPI `TestClient` for upload/status/transcript/chunks/ask (mock the pipeline + LLM).
3. Integration: upload → mocked pipeline → ask, using ephemeral Redis/Qdrant (testcontainers or compose service).
4. Target a coverage threshold (e.g. 70%) enforced in CI.

### Step 9.2 — Docs
1. Keep this `docs/PROJECT_PLAN.md` updated.
2. Add `docs/RUNBOOK.md`: local (`docker compose up`) and k8s (`helm install`) bring-up, env vars, troubleshooting.
3. Update root `README.md` with architecture diagram and quickstart.

**Phase 9 done when:** `pytest` is green in CI and a new dev can follow the runbook from zero to running.

---

## Suggested order & milestones

1. **M1 – Solid base:** Phase 1 + Phase 9.1 (minimal tests).
2. **M2 – Working product:** Phase 2 + Phase 3.
3. **M3 – Reproducible ML:** Phase 4.
4. **M4 – Observable:** Phase 5 + Phase 6.
5. **M5 – Automated:** Phase 7.
6. **M6 – Production:** Phase 8 + finish Phase 9.

## Tool choices (popular / widely used)
- Experiments & registry: **MLflow**
- CI/CD: **GitHub Actions** + **GHCR** + **Trivy**
- Metrics: **Prometheus + Grafana** (`kube-prometheus-stack`)
- Logs: **Loki + Promtail**
- Traces: **OpenTelemetry + Tempo** (Jaeger alternative)
- Object storage: **MinIO** (S3-compatible)
- Metadata DB: **PostgreSQL**; Vectors: **Qdrant**; Broker/cache: **Redis**
- Embeddings: **CLIP** (visual) + **bge-small** (text); LLM: **OpenAI** (pluggable)
- Frontend: **Next.js + Tailwind**
- Orchestration: **Docker Compose** (local) + **Kubernetes + Helm** (cloud); secrets via **external-secrets**
