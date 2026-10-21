# syntax=docker/dockerfile:1
# tour_segment worker: OneFormer on CUDA 12.6 PyTorch, with the pinned weights baked in (no job
# downloads them). Build from the repo root:
#   docker build -f docker/segment.Dockerfile -t tour-worker-segment .
#   ... --build-arg BAKE_ADE20K=1            also bake the ADE20K checkpoint (SEGMENT_MODEL=ade20k)
#   ... --build-arg TORCH_EXTRA=segment      CPU PyTorch instead (local development, no GPU)
# Default: the HTTP service on :8000 (needs WORKER_API_KEY).
# RunPod serverless (GPU): same image, container start command `python -u handler.py`.
# The PyTorch cu126 wheels bring their own CUDA runtime; the base image only needs to expose the
# host's NVIDIA driver (the NVIDIA container runtime does that, as on RunPod).
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ARG TORCH_EXTRA=segment-gpu
ARG BAKE_ADE20K=0

COPY --from=ghcr.io/astral-sh/uv:0.11.22 /uv /bin/uv

# python3.12 is Ubuntu 24.04's own; uv makes the venv itself (no python3-venv needed)
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra "${TORCH_EXTRA}" --extra http --no-install-project
COPY LICENSES.md handler.py ./
COPY tour_worker ./tour_worker
RUN uv sync --frozen --no-dev --extra "${TORCH_EXTRA}" --extra http

# Pinned OneFormer weights (sha256-checked by ensure_model). COCO is the production default.
ENV MODELS_DIR=/models HF_HUB_DISABLE_TELEMETRY=1
RUN /opt/venv/bin/python -c "from tour_worker.vision.segment import MODELS, ensure_model; \
import os; ensure_model(os.path.join('/models', MODELS['coco']['dir']), 'coco')" \
 && if [ "${BAKE_ADE20K}" = "1" ]; then /opt/venv/bin/python -c "from tour_worker.vision.segment import MODELS, ensure_model; \
import os; ensure_model(os.path.join('/models', MODELS['ade20k']['dir']), 'ade20k')"; fi \
 && rm -rf /models/*/.cache /root/.cache

RUN useradd --create-home --uid 10001 worker \
 && mkdir -p /var/lib/tour-worker/jobs \
 && chown worker /var/lib/tour-worker/jobs
USER worker

# HF_HUB_OFFLINE: a missing checkpoint fails the job at once instead of downloading at run time
ENV PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    SEGMENT_MODEL=coco \
    JOBS_DIR=/var/lib/tour-worker/jobs \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

CMD ["python", "-m", "tour_worker.service"]
