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
