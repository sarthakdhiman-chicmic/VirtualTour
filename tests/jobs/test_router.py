"""tour_worker.jobs.router: dispatch, request validation and the single failure shape."""
import pytest

from tour_worker.contract.jobs import JobFailure
from tour_worker.jobs.router import run_job


@pytest.mark.parametrize("job, code", [
    ({}, "invalid_job_type"),
    ({"job_type": "virtual_tour"}, "invalid_job_type"),
    ({"job_type": "tour_render", "plan": {}}, "invalid_request"),                       # no output_prefix
    ({"job_type": "tour_render", "plan": {}, "output_prefix": ""}, "invalid_request"),
    ({"job_type": "tour_render", "output_prefix": "s3://b/o/"}, "invalid_request"),     # no plan
    ({"job_type": "tour_segment", "images": [], "output_prefix": "s3://b/s/"}, "invalid_request"),
    ({"job_type": "tour_segment", "images": [{"photo_id": "p", "uri": "s3://b/p.jpg"}]}, "invalid_request"),
    ({"job_type": "tour_render", "plan": {"plan_version": "1.5"}, "output_prefix": "s3://b/o/"}, "invalid_plan"),
    ({"job_type": "tour_plan", "photos": [{"photo_id": "p", "uri": "s3://b/p.jpg"}]}, "invalid_request"),  # no listing_id
    ({"job_type": "tour_plan", "listing_id": "L", "photos": []}, "invalid_request"),
])
def test_failures_have_one_shape(job, code):
    res = run_job(job)
    JobFailure.model_validate(res)
    assert res["error"]["code"] == code and res["error"]["message"]


def test_a_crash_inside_a_job_is_a_failure_not_an_exception(monkeypatch):
    import tour_worker.jobs.segment as seg

    def boom(settings):
        raise RuntimeError("model_load_failed: no weights")

    monkeypatch.setattr(seg, "get_segmenter", boom)
    res = run_job({"job_type": "tour_segment", "images": [{"photo_id": "p", "uri": "s3://b/p.jpg"}],
                   "output_prefix": "s3://b/s/"})
    assert res == {"status": "failed", "error": {"code": "model_load_failed", "message": "model_load_failed: no weights"}}


@pytest.mark.parametrize("job, code", [
    ({}, "invalid_job_type"),
    ("not a dict", "invalid_job_type"),
    ({"job_type": "tour_render", "plan": {}}, "invalid_request"),
    ({"job_type": "tour_segment", "images": [], "output_prefix": "s3://b/s/"}, "invalid_request"),
    ({"job_type": "tour_render", "plan": {"plan_version": "1.5"}, "output_prefix": "s3://b/o/"}, "invalid_plan"),
])
def test_validate_request_matches_run_job(job, code):
    from tour_worker.jobs.router import validate_request
    bad = validate_request(job)
    assert bad["error"]["code"] == code
    assert bad == run_job(job if isinstance(job, dict) else {})


def test_validate_request_accepts_the_examples_and_checks_the_plan_hash():
    import copy
    import json
    import os
    from tour_worker.config import REPO_ROOT
    from tour_worker.jobs.router import validate_request
    for name in ("tour_render.json", "tour_segment.json", "tour_plan.json"):
        req = json.load(open(os.path.join(REPO_ROOT, "examples", name)))["input"]
        assert validate_request(req) is None
    req = json.load(open(os.path.join(REPO_ROOT, "examples", "tour_render.json")))["input"]
    bad = copy.deepcopy(req)
    bad["plan"]["total_duration_s"] += 1                    # the plan no longer matches its hash
    assert validate_request(bad)["error"]["code"] == "invalid_plan"


def test_tour_plan_is_dispatched_to_its_job(monkeypatch):
    import tour_worker.jobs.plan as plan_job
    seen = []
    monkeypatch.setattr(plan_job, "run_plan", lambda job, storage, workdir: seen.append(job) or {"status": "completed"})
    job = {"job_type": "tour_plan", "listing_id": "L", "photos": [{"photo_id": "p", "uri": "s3://b/p.jpg"}]}
    assert run_job(job, storage=object()) == {"status": "completed"} and seen == [job]
