# syntax=docker/dockerfile:1
# tour_render worker: OpenCV + ffmpeg, no PyTorch. Build from the repo root:
#   docker build -f docker/render.Dockerfile -t tour-worker-render .
# Default: the HTTP service (python -m tour_worker.service) on :8000 - needs WORKER_API_KEY.
# RunPod serverless: same image, container start command `python -u handler.py`.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /bin/uv

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1

WORKDIR /app
# dependencies first (cached while only the code changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra http --no-install-project
COPY LICENSES.md handler.py ./
COPY tour_worker ./tour_worker
RUN uv sync --frozen --no-dev --extra http

RUN useradd --create-home --uid 10001 worker \
 && mkdir -p /models /var/lib/tour-worker/jobs \
 && chown worker /var/lib/tour-worker/jobs
USER worker

ENV PATH=/opt/venv/bin:$PATH \
    MODELS_DIR=/models \
    JOBS_DIR=/var/lib/tour-worker/jobs \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

CMD ["python", "-m", "tour_worker.service"]
