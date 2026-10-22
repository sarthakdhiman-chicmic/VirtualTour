"""examples/*.json stay valid requests (current plan version, correct plan_hash)."""
import json
import os

from tour_worker.config import REPO_ROOT
from tour_worker.contract.jobs import TourPlanRequest, TourRenderRequest, TourSegmentRequest
from tour_worker.contract.plan import parse_plan


def _load(name):
    return json.load(open(os.path.join(REPO_ROOT, "examples", name)))["input"]


def test_tour_render_example_is_a_valid_plan_15_request():
    req = _load("tour_render.json")
    TourRenderRequest.model_validate(req)
    plan = parse_plan(req["plan"])
    assert plan.plan_version == "1.5" and plan.music is not None


def test_tour_segment_example_is_valid():
    TourSegmentRequest.model_validate(_load("tour_segment.json"))


def test_tour_plan_example_is_valid():
    TourPlanRequest.model_validate(_load("tour_plan.json"))
