"""The plan schema lives in two places: tour_worker/contract/plan.py (what the renderer accepts) and
tour_worker/planner/schemas.py (what the planner writes). These tests fail when they diverge."""
import json
import os

from tour_worker.contract import plan as contract
from tour_worker.planner import schemas as planner

# planner model -> contract model. Every field the planner writes must exist in the contract
# (the contract ignores unknown keys, so a missing one would vanish silently from the render).
PAIRS = [
    (planner.ShotPlan, contract.Plan), (planner.Shot, contract.Shot), (planner.CameraParams, contract.Camera),
    (planner.OutputParams, contract.OutputConfig), (planner.Keyframe, contract.Keyframe),
    (planner.Segment, contract.Segment), (planner.Timing, contract.Timing), (planner.Punch, contract.Punch),
    (planner.MotionRegion, contract.MotionRegion), (planner.TransitionParams, contract.Transition),
    (planner.Voiceover, contract.Voiceover), (planner.Branding, contract.Branding),
    (planner.AgentCard, contract.AgentCard), (planner.RoomLabels, contract.RoomLabels),
]
# models added for plan 1.6 must be identical, field for field
IDENTICAL = [(planner.BezierSegment, contract.BezierSegment), (planner.Scene, contract.Scene)]


def test_every_planner_field_exists_in_the_contract():
    for p, c in PAIRS:
        missing = set(p.model_fields) - set(c.model_fields)
        assert not missing, f"{p.__name__} fields missing from contract {c.__name__}: {missing}"


def test_the_16_models_are_identical():
    for p, c in IDENTICAL:
        assert list(p.model_fields) == list(c.model_fields), (p.__name__, c.__name__)
        for name in p.model_fields:
            assert p.model_fields[name].is_required() == c.model_fields[name].is_required(), (p.__name__, name)


def test_both_schemas_read_a_camera_dict_the_same_way():
    bez = {"type": "bezier_path", "segments": [{"eye": [[0, 0, 0]] * 4, "target": [[1, 0, 0]] * 4}], "hfov_deg": [70.0, 60.0]}
    comp = {"type": "compound", "segments": [{"t0": 0.0, "t1": 1.0, "cx": 0.5, "cy": 0.5, "zoom": 1.1}], "hfov_deg": 70.0}
    for cam in (bez, comp):
        a = planner.CameraParams.model_validate(cam)
        b = contract.Camera.model_validate(cam)
        assert type(a.segments[0]).__name__ == type(b.segments[0]).__name__
        assert a.hfov_deg == b.hfov_deg
        assert a.segments[0].model_dump() == b.segments[0].model_dump()


def test_a_planner_16_plan_reads_back_unchanged_in_the_contract():
    p = json.load(open(os.path.join(os.path.dirname(__file__), "..", "fixtures", "plans", "plan_1_6_box_room.json")))
    c = contract.parse_plan(p)
    assert c.scene.model_dump(exclude_none=True) == p["scene"]
    for cs, ps in zip(c.shots, p["shots"]):
        assert cs.camera.model_dump(include={"type", "segments", "hfov_deg", "easing"}) == \
            {k: ps["camera"][k] for k in ("type", "segments", "hfov_deg", "easing")}
