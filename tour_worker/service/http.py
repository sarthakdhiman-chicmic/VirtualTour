"""The worker as an HTTP service that speaks RunPod's serverless API, so the backend has one client
for both: only the base URL and the key differ.

  RunPod     https://api.runpod.ai/v2/<endpoint_id>   (handler.py runs the jobs)
  service    http://<host>:<port>                      (this app runs them, tour_worker.service.jobs)

  POST /run           {"input": {...}, "webhook"?: url}  -> {"id", "status": "IN_QUEUE"}
  POST /runsync       same body; waits up to RUNSYNC_TIMEOUT_S -> the /status body
  GET  /status/{id}   -> {"id", "status", "delayTime"?, "executionTime"?, "output"?, "error"?}
  POST /cancel/{id}   -> {"id", "status"} (only a queued job can be cancelled)
  GET  /health        -> {"jobs": {...}, "workers": {...}, "version", ...} (no key needed)

Unlike RunPod, a request that cannot run (bad job_type, missing fields, a plan that fails its
checks) is refused at submit with HTTP 400 and the worker's failure shape
{"status": "failed", "error": {"code", "message"}} - the same codes the job would fail with.
Every other HTTP error uses that shape too (401 unauthorized, 404 job_not_found)."""
from __future__ import annotations

import hmac
import logging
import socket
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from tour_worker.config import Settings, get_settings
from tour_worker.contract.jobs import JOB_TYPES
from tour_worker.errors import failure
from tour_worker.service.jobs import CANCELLED, COMPLETED, FAILED, IN_PROGRESS, IN_QUEUE, JobStore

log = logging.getLogger("tour_worker.service")


def package_version() -> str:
    try:
        from importlib.metadata import version
        return version("tour-worker")
    except Exception:                    # noqa: BLE001 - running from a checkout, not installed
        return "unknown"


class RunRequest(BaseModel):
    """RunPod's /run body. `policy` (RunPod's execution timeout / TTL) is accepted and ignored."""
    model_config = ConfigDict(extra="ignore")
    input: Any
    webhook: Optional[str] = None
    policy: Optional[dict] = None


class ServiceError(Exception):
    def __init__(self, http_status: int, code: str, message: str):
        self.http_status, self.code, self.message = http_status, code, message


def _error(http_status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(failure(code, message), status_code=http_status)


def create_app(settings: Settings | None = None, store: JobStore | None = None) -> FastAPI:
    """The app. Refuses (RuntimeError) to build without WORKER_API_KEY unless
    WORKER_ALLOW_NO_AUTH=1. store: inject one (tests); default from settings."""
    settings = settings or get_settings()
    if not settings.api_key and not settings.allow_no_auth:
        raise RuntimeError("WORKER_API_KEY is not set. Set it, or set WORKER_ALLOW_NO_AUTH=1 "
                           "for local development without authentication.")
    if not settings.api_key:
        log.warning("WORKER_ALLOW_NO_AUTH=1: the service accepts requests without a key")
    store = store or JobStore(concurrency=settings.job_concurrency, jobs_dir=settings.jobs_dir,
                              ttl_s=settings.job_ttl_s)
    expected = f"Bearer {settings.api_key}".encode() if settings.api_key else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.start_janitor()
        log.info("tour-worker %s: %d job(s) at once, records %s, TTL %ss", package_version(),
                 store.concurrency, settings.jobs_dir or "in memory", settings.job_ttl_s)
        yield
        store.shutdown(wait=False)

    app = FastAPI(title="tour-worker", version=package_version(), lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.settings = store, settings

    def require_key(request: Request) -> None:
        if expected is None:
            return
        got = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(got, expected):
            raise ServiceError(401, "unauthorized", "send Authorization: Bearer <WORKER_API_KEY>")

    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, e: ServiceError):
        return _error(e.http_status, e.code, e.message)

    @app.exception_handler(RequestValidationError)
    async def _bad_body(request: Request, e: RequestValidationError):
        # never echo the body back (it has presigned URLs): locations and messages only
        msg = "; ".join(f"{'.'.join(str(p) for p in err.get('loc', ())) or 'body'}: {err.get('msg')}"
                        for err in e.errors())
        return _error(400, "invalid_request", msg or "the body must be JSON: {\"input\": {...}}")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, e: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "http_error")
        return _error(e.status_code, code, str(e.detail))

    def _submit(body: RunRequest):
        from tour_worker.jobs.router import validate_request
        if not isinstance(body.input, dict):
            raise ServiceError(400, "invalid_request", "input must be a JSON object")
        if body.webhook is not None and not body.webhook.startswith(("https://", "http://")):
            raise ServiceError(400, "invalid_request", "webhook must be an http(s) URL")
        bad = validate_request(body.input)
        if bad is not None:
            raise ServiceError(400, bad["error"]["code"], bad["error"]["message"])
        return store.submit(body.input, webhook=body.webhook)

    def _status_or_404(job_id: str, body: dict | None) -> dict:
        if body is None:
            raise ServiceError(404, "job_not_found", f"no job {job_id} (unknown, or older than the TTL)")
        return body

    @app.post("/run", dependencies=[Depends(require_key)])
    def run(body: RunRequest):
        job = _submit(body)
        return {"id": job.id, "status": IN_QUEUE}      # what it was when queued (it may be running now)

    @app.post("/runsync", dependencies=[Depends(require_key)])
    def runsync(body: RunRequest):
        job = _submit(body)
        return store.wait(job.id, settings.runsync_timeout_s)

    @app.get("/status/{job_id}", dependencies=[Depends(require_key)])
    def status(job_id: str):
        return _status_or_404(job_id, store.status(job_id))

    @app.post("/cancel/{job_id}", dependencies=[Depends(require_key)])
    def cancel(job_id: str):
        return _status_or_404(job_id, store.cancel(job_id))

    @app.get("/health")
    def health():
        c = store.counts()
        running = c[IN_PROGRESS]
        return {
            "jobs": {"inQueue": c[IN_QUEUE], "inProgress": running, "completed": c[COMPLETED],
                     "failed": c[FAILED], "cancelled": c[CANCELLED]},
            "workers": {"running": running, "idle": max(0, store.concurrency - running),
                        "concurrency": store.concurrency},
            "version": package_version(),
            "jobTypes": list(JOB_TYPES),
            "segmentModel": _segment_model(settings),
            "workerId": socket.gethostname(),
        }

    return app


def _segment_model(settings: Settings) -> str | None:
    """The OneFormer checkpoint tour_segment would use ("<hf id>@<revision>")."""
    try:
        from tour_worker.vision.segment import model_id
        return model_id(settings.segment_model)
    except Exception:                    # noqa: BLE001 - e.g. an unknown SEGMENT_MODEL
        return None
