"""One entry point for every job: dispatch by job_type, validate the request
(tour_worker.contract.jobs), run it in a temporary folder and return its result - or, for any
failure, {"status": "failed", "error": {"code", "message"}}.

Jobs: tour_render (jobs/render.py), tour_segment (jobs/segment.py), tour_plan (jobs/plan.py).
Codes: invalid_job_type, invalid_request (request shape), the plan's own codes (invalid_plan,
unsupported_camera_type, unsupported_easing), and the run's (image_fetch_failed, encode_failed,
verification_failed, planning_failed, ...; unknown for anything unexpected)."""
from __future__ import annotations

import tempfile

from pydantic import ValidationError

from tour_worker.config import Settings, get_settings
from tour_worker.contract.jobs import (
    JOB_TYPES,
    TourPlanRequest,
    TourRenderRequest,
    TourSegmentRequest,
)
from tour_worker.errors import failure, from_exception
from tour_worker.io.storage import build_storage

_REQUESTS = {"tour_render": TourRenderRequest, "tour_segment": TourSegmentRequest,
             "tour_plan": TourPlanRequest}


def _validation_message(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}" for err in e.errors())


def _request_failure(job) -> dict | None:
    """The failure for a request of the wrong shape (job_type, required fields), or None."""
    job_type = job.get("job_type") if isinstance(job, dict) else None
    if job_type not in JOB_TYPES:
        return failure("invalid_job_type", f"job_type must be one of {', '.join(JOB_TYPES)}")
    try:
        _REQUESTS[job_type].model_validate(job)
    except ValidationError as e:
        return failure("invalid_request", _validation_message(e))
    return None


def validate_request(job) -> dict | None:
    """Check a request before queueing it (the HTTP service: callers fail fast): the failure
    run_job would return for it without fetching anything, or None. Checks the request's shape
    and, for tour_render, the plan (schema, plan_hash, camera / easing / bounds rules). No I/O."""
    bad = _request_failure(job)
    if bad is None and job["job_type"] == "tour_render":
        from tour_worker.contract.plan import parse_plan, plan_failure
        try:
            parse_plan(job["plan"])
        except ValueError as e:              # pydantic's ValidationError is a ValueError
            bad = plan_failure(e)
    return bad


def run_job(job: dict, settings: Settings | None = None, storage=None) -> dict:
    """Run one job (the RunPod event's `input`). storage: override (tests); default from settings."""
    settings = settings or get_settings()
    job = job if isinstance(job, dict) else {}
    bad = _request_failure(job)
    if bad is not None:
        return bad
    job_type = job["job_type"]
    storage = storage or build_storage(settings, job_type)
    try:
        with tempfile.TemporaryDirectory() as workdir:
            if job_type == "tour_segment":
                from tour_worker.jobs.segment import run_segment
                return run_segment(job, storage, workdir, settings=settings)
            if job_type == "tour_plan":
                from tour_worker.jobs.plan import run_plan
                return run_plan(job, storage, workdir)
            from tour_worker.jobs.render import run_tour
            return run_tour(job, storage, workdir, settings=settings)
    except Exception as e:                  # e.g. the segmentation model failed to load
        return from_exception(e)
