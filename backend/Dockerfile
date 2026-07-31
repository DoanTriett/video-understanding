# syntax=docker/dockerfile:1
# Build context must be the repository root:
#   docker build -f backend/Dockerfile .   (from repo root)
#
# Runtime env (POSTGRES_URL, REDIS_URL, MINIO_*, etc.) is injected by the orchestrator.

FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# CPU-only PyTorch. The API has no GPU; pipeline work runs in the worker image.
# Full requirements.txt is still installed because app.api.videos imports workers.tasks
# and eagerly imports pipeline modules at startup.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --force-reinstall torch \
      --index-url https://download.pytorch.org/whl/cpu

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libpq5 \
      libsndfile1 \
      libglib2.0-0 \
      libgl1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY backend/. .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
