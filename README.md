# Video Understanding System

An end-to-end multimodal AI system for uploading meeting or lecture videos, generating structured summaries, and asking grounded questions with clickable timestamp citations.

**Live frontend:** [https://video-understanding.vercel.app](https://video-understanding.vercel.app)

**API health:** [https://video-understanding-production.up.railway.app/health](https://video-understanding-production.up.railway.app/health)

## Problem and Idea

Long meetings and lectures contain useful information across speech, speakers, slides, and screen shares, but finding one specific moment usually means scrubbing through the entire recording. Plain transcription improves searchability but loses speaker and visual context.

This project narrows the problem to two structured video types:

- Meetings are organized around speaker turns and occasional screen sharing.
- Lectures are organized around slide changes, OCR text, and spoken explanations.

The system converts those signals into time-aligned chunks, retrieves the most relevant evidence for a question, and returns an answer whose citations seek directly to the supporting moments in the video.

## Architecture

```mermaid
flowchart LR
    U[User] --> FE[Next.js frontend<br/>Vercel]
    FE --> API[FastAPI API<br/>Railway]

    API --> OBJ[MinIO-compatible<br/>object storage]
    API --> PG[(PostgreSQL<br/>Neon)]
    API --> REDIS[(Redis Cloud<br/>broker + progress + cache)]
    API --> QD[(Qdrant<br/>Docker container)]
    API --> OAI[OpenAI API<br/>gpt-4.1-mini]

    REDIS --> GPU[Celery GPU worker<br/>RunPod]
    GPU --> OBJ
    GPU --> PG
    GPU --> QD

    GPU --> AUDIO[Faster-Whisper<br/>+ pyannote]
    GPU --> VISION[OpenCV/SSIM<br/>+ PaddleOCR + CLIP]

    QD --> API
    PG --> API
    API --> FE
```

Processing flow:

1. The API stores the uploaded video and creates a PostgreSQL record.
2. A Celery job is published through Redis Cloud.
3. The RunPod worker downloads the video and runs the meeting or lecture pipeline.
4. Chunks are persisted in PostgreSQL and their embeddings are indexed in Qdrant.
5. A question is embedded, retrieved against the video's chunks, and formatted as time-ordered context.
6. OpenAI generates a grounded answer; the API returns the answer and source chunks as clickable citations.
7. Similar questions can reuse a seven-day Redis semantic-cache entry.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS 4, TanStack Query, Zustand, React Player, shadcn/ui |
| API and jobs | Python 3.11, FastAPI, Pydantic, SQLAlchemy, Alembic, Celery, Redis, SlowAPI |
| Audio | FFmpeg, Faster-Whisper `large-v3`, pyannote `speaker-diarization-3.1` |
| Vision | OpenCV, SSIM, PaddleOCR, CLIP `ViT-B/32` |
| Retrieval and LLM | BGE `bge-small-en-v1.5`, Qdrant, OpenAI API (`gpt-4.1-mini`) |
| Data | PostgreSQL, Redis, MinIO-compatible S3 object storage |
| MLOps and observability | MLflow, Prometheus, Grafana, structured JSON logging |
| Delivery | Docker, Docker Compose, GitHub Actions, GHCR, Trivy, Helm |
| Production hosting | Vercel, Railway, Neon, Redis Cloud, RunPod, Dockerized Qdrant |

## Key Design Decisions

### Scope meetings and lectures separately

A general-purpose video parser would make temporal alignment and modality selection much harder. Meeting chunks follow speaker turns and screen-share overlap; lecture chunks follow slide boundaries and combine OCR with speech.

### Keep API serving separate from GPU inference

The Railway API handles uploads, status, retrieval, and response serving. Expensive transcription, diarization, OCR, visual embedding, and indexing run asynchronously in a RunPod Celery worker. Redis Cloud connects both sides without requiring the public API to own a GPU.

### Run GPU-heavy models sequentially

Faster-Whisper, pyannote, and CLIP are loaded and released by stage. Explicit cleanup reduces VRAM contention and allows the pipeline to run on a constrained single-GPU worker.

### Use each data store for one primary responsibility

- PostgreSQL is the durable source of truth for videos, chunks, and summaries.
- Redis is ephemeral infrastructure for Celery, live progress, and semantic caching.
- Qdrant stores retrieval vectors and citation payloads.
- MinIO-compatible storage holds large video and transcript objects.

### Move generation from Ollama to OpenAI

Answer generation and auto-summary now use the OpenAI API with `gpt-4.1-mini` by default. The migration preserves the original grounded-answer prompt, timestamp contract, meeting/lecture summary schemas, JSON validation, and one retry on invalid summary JSON. It also removes local LLM VRAM pressure from the GPU video pipeline.

### Return evidence, not only generated text

Retrieved chunks retain `video_id`, `chunk_id`, speaker, start/end time, text, and chunk type. The answer prompt requires timestamp citations, while the response separately returns the exact chunks used so the frontend can seek to their start times.

## Eval Results

The checked-in evaluation is an internal LLM-as-a-Judge study over cached QA interactions, not a production benchmark. These reports were generated before the Ollama-to-OpenAI migration and must be rerun before the numbers can be attributed to `gpt-4.1-mini`.

### Stored baseline

| Metric | Result |
| --- | ---: |
| QA pairs | 20 |
| Answer-level faithfulness | 40.00% |
| Average claim-level hallucination rate | 44.32% |
| Verified timestamp pairs | 0 |
| Retrieval timestamp overlap | N/A |

### Retrieval-context ablation

| Configuration | Evaluated pairs | Faithfulness | Avg. hallucination rate |
| --- | ---: | ---: | ---: |
| `all_chunks` | 20 | 65.00% | 33.57% |
| `speech_only` | 12 | 16.67% | 60.52% |

Within this small snapshot, retaining screen-share/slide-enriched chunks performed better than filtering retrieval to speech chunks. The unequal sample counts and unverified cache-derived annotations mean the result should be treated as directional evidence only.

See [the full report](eval/reports/report.md) and [the ablation report](eval/reports/ablation.md).

## Known Limitations

- The current online retriever searches the 384-dimensional text collection only. Visual questions are detected, and 512-dimensional CLIP image vectors are indexed, but CLIP text encoding is not yet wired into `chunks_visual` search.
- The frontend exposes an auto-detect option that maps to `unknown`, while the worker currently processes only explicit `meeting` or `lecture` types. Select a type manually for reliable processing.
- Evaluation has only 20 cache-derived QA pairs and zero manually verified timestamp annotations. WER, DER, retrieval overlap, and post-OpenAI-migration results are not available.
- Full-summary input is truncated at 30,000 characters, so very long videos may have incomplete summaries.
- Browser state and chat history are session-only; refreshing the page does not restore the active video or conversation.
- The current deployment uses a single on-demand GPU worker and self-hosted Qdrant container, without high availability or automatic GPU failover.
- Retrieval uses dense similarity without BM25, a reranker, or semantic topic segmentation; short generic chunks can outrank longer relevant chunks.

## Local Setup and Testing

### Prerequisites

- Python 3.11
- Node.js 22 and npm
- Docker Desktop with Docker Compose
- FFmpeg available on `PATH`
- An OpenAI API key
- A Hugging Face token with access to `pyannote/speaker-diarization-3.1`
- An NVIDIA GPU is recommended for full video processing; CPU fallback is much slower

### 1. Start local data services

```bash
docker compose up -d postgres redis qdrant minio
```

Open the MinIO console at <http://localhost:9001>, sign in with the development credentials from `docker-compose.yml`, and create a bucket named `videos`.

### 2. Configure and run the backend

From `backend/`, create `.env` from `.env.example` and set at least:

```dotenv
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1-mini
HUGGINGFACE_TOKEN=your_token
POSTGRES_URL=postgresql://vu:123@localhost:5432/videodb
REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
QDRANT_PORT=6333
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=videos
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,https://video-understanding.vercel.app
```

Create the environment, install dependencies, and apply migrations:

```bash
cd backend
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# bash/zsh:   source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
alembic upgrade head
```

Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

In another activated terminal, start the worker. Use `--pool=solo` on Windows:

```bash
celery -A workers.celery_app worker --loglevel=info --pool=solo
```

On Linux, the default prefork pool can be used by omitting `--pool=solo`.

### 3. Run the frontend

```bash
cd frontend
npm ci
```

Create `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Then start the app:

```bash
npm run dev
```

Open <http://localhost:3000>. Choose `meeting` or `lecture` explicitly when uploading.

### 4. Run checks

Backend checks from `backend/`:

```bash
ruff check .
ruff format --check .
mypy .
pytest tests -m "not integration" --cov=app --cov-report=term-missing
pytest tests -m "integration and not requires_openai" -v
```

The second pytest command requires the local PostgreSQL, Redis, and Qdrant services. OpenAI integration tests are excluded unless a real key and paid API access are intentionally provided.

Frontend checks from `frontend/`:

```bash
npm run lint
npm run build
```

To rerun the offline evaluation from the repository root after configuring all shared services and OpenAI:

```bash
python eval/run_eval.py --dataset eval/datasets --output eval/reports --top-k 6
python eval/ablation.py --dataset eval/datasets --output eval/reports --top-k 6
```

## Deployment

| Component | Host | Notes |
| --- | --- | --- |
| Frontend | [Vercel](https://video-understanding.vercel.app) | Production Next.js build; `NEXT_PUBLIC_API_URL` points to Railway at build time. |
| API | [Railway](https://video-understanding-production.up.railway.app) | CPU FastAPI container built from `backend/Dockerfile`. |
| PostgreSQL | Neon | Managed durable database for videos, chunks, and summaries. |
| Redis | Redis Cloud | Shared Celery broker/result backend, progress store, and semantic cache. |
| GPU worker | RunPod | On-demand NVIDIA GPU container built from `backend/Dockerfile.worker`. |
| Vector database | Qdrant in Docker | Self-hosted from the official Qdrant image; not Qdrant Cloud. |
| Object storage | MinIO-compatible S3 service | Source videos and transcript artifacts, served through presigned URLs. |
| LLM | OpenAI API | `gpt-4.1-mini` by default for answers, summaries, and judge calls. |

Production secrets are injected by each provider and are never committed. The API and worker must use the same Neon, Redis Cloud, Qdrant, object-storage, Hugging Face, and OpenAI configuration.

The repository also contains Docker Compose for local orchestration and a minimal Helm chart for an alternative Kubernetes deployment. The Helm workflow is not the active production source of truth.

## Future Improvements

- Rerun the full evaluation after the OpenAI migration and record provider/model metadata with every report.
- Build a manually verified meeting/lecture dataset with ground-truth timestamps, WER, DER, recall@k, and MRR.
- Wire CLIP text embeddings into visual Qdrant search and fuse text/visual results with a reranker.
- Add BM25 or sparse-dense hybrid retrieval and topic-aware semantic chunking.
- Replace the non-functional auto-detect path with a real video-type classifier or remove the option.
- Add persistent user sessions, chat history, and resumable/retryable uploads.
- Add RunPod worker autoscaling, warm model snapshots, health checks, and failed-job recovery.
- Add Qdrant replication/backups and production-grade object-storage lifecycle policies.
- Complete OpenTelemetry tracing, Grafana dashboards, alerting, and centralized logs.
- Add authentication, per-user authorization, quotas, and stronger production rate limiting.

## Repository Layout

```text
video-understanding/
├── backend/              # FastAPI, Celery, ML pipelines, DB, tests
├── frontend/             # Next.js application
├── eval/                 # Dataset schema, metrics, reports, ablations
├── deploy/helm/          # Optional Kubernetes deployment
├── infra/                # Prometheus and Grafana configuration
├── docs/                 # Original plan and end-to-end roadmap
└── docker-compose.yml    # Local stack
```
