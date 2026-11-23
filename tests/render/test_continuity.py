"""Cut continuity (render/continuity.py, client review 2026-11-20 items 1-9): every shot's entry
and exit in % of frame width per second, and each cut judged on speed, direction and beat."""
import copy
import json
import os

from tour_worker.contract.plan import compute_plan_hash, parse_plan
from tour_worker.render import continuity as c
from tour_worker.render.continuity import Motion

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures", "plans")


def _plan(name):
    return json.load(open(os.path.join(FIXTURES, name)))


def _parse(p):
    p["provenance"] = {"scoring_config_version": "t", "planner_config_version": "t", "plan_hash": None}
    p["provenance"]["plan_hash"] = compute_plan_hash(p)
    return parse_plan(p)


def _kf(cx0, cy0, z0, cx1, cy1, z1):
    return {"type": "keyframe_path", "move": "test", "keyframes": [
        {"t": 0, "cx": cx0, "cy": cy0, "zoom": z0}, {"t": 1, "cx": cx1, "cy": cy1, "zoom": z1}]}


def _two_shots(cam_a, cam_b, music=None):
    p = _plan("plan_1_5_turkey_trot.json")
    p["shots"] = copy.deepcopy(p["shots"][:2])
    p["shots"][0]["camera"], p["shots"][1]["camera"] = cam_a, cam_b
    for s in p["shots"]:
        s.pop("motion_regions", None)
        s["transition_in"] = {"type": "cut", "duration_s": 0.0}
    p["total_duration_s"] = round(sum(s["duration_s"] for s in p["shots"]), 2)
    p["music"] = music
    return _parse(p)


def test_directions_read_as_the_camera_moves():
    assert Motion(3.0, 0.0, 0.0).direction() == "right"
    assert Motion(-3.0, 0.0, 0.0).direction() == "left"
    assert Motion(0.0, 0.0, 3.0).direction() == "forward"
    assert Motion(0.0, 0.0, -3.0).direction() == "back"
    assert Motion(0.1, 0.0, 0.1).direction() == "still"


def test_a_pan_right_then_a_pan_left_is_a_reversal():
    plan = _two_shots(_kf(0.4, 0.5, 1.5, 0.6, 0.5, 1.5), _kf(0.65, 0.5, 1.5, 0.35, 0.5, 1.5))
    row = c.report(plan)[0]
    assert row["out"]["direction"] == "right" and row["in"]["direction"] == "left"
    assert row["turn_deg"] > 170 and "direction_turn" in row["breaks"]
    assert c.violations([row])[0].startswith("cut_")


def test_a_push_in_into_a_pull_out_is_forward_back():
    plan = _two_shots(_kf(0.5, 0.5, 1.1, 0.5, 0.5, 1.4), _kf(0.5, 0.5, 1.4, 0.5, 0.5, 1.1))
    row = c.report(plan)[0]
    assert row["out"]["direction"] == "forward" and row["in"]["direction"] == "back"
    assert row["axial_reversal"] and "forward_back" in row["breaks"]


def test_the_old_time_profile_makes_the_incoming_shot_enter_faster():
    """The same right pan across the cut, no path_s: the 1.6 / 0.8 time profile (plan <= 1.6
    default) enters faster than it leaves (item 2)."""
    plan = _two_shots(_kf(0.4, 0.5, 1.5, 0.6, 0.5, 1.5), _kf(0.4, 0.5, 1.5, 0.6, 0.5, 1.5))
    row = c.report(plan)[0]
    assert row["turn_deg"] is not None and row["turn_deg"] < 1
    assert row["breaks"] == ["speed_jump"] and row["speed_ratio"] > 1.5


def test_with_continuous_cuts_the_same_move_carries_through_the_cut():
    """Items 2-3: path_s at the move's pace (linear between the tour's ends) - the same pan
    continued across the cut leaves and enters at the same speed, so the cut is within limits."""
    a = {**_kf(0.35, 0.5, 1.5, 0.65, 0.5, 1.5), "easing": "ease_in"}
    b = {**_kf(0.35, 0.5, 1.5, 0.65, 0.5, 1.5), "easing": "ease_out"}
    plan = _two_shots(a, b)
    d0, d1 = plan.shots[0].duration_s, plan.shots[1].duration_s
    p = plan.model_dump(mode="json")
    p["shots"][0]["camera"]["path_s"] = d0 + 1.0            # cut a second before its path ends
    p["shots"][1]["camera"]["path_s"] = d1
    p["shots"][1]["camera"]["keyframes"][1]["cx"] = 0.35 + 0.3 * d1 / (d0 + 1.0)   # the same pace
    row = c.report(_parse(p))[0]
    assert row["breaks"] == [] and row["speed_ratio"] < 1.05 and row["turn_deg"] < 1


def test_ramp_profiles_hold_a_steady_pace_between_the_ramps():
    from tour_worker.render.camera_path import ramp_profile
    for e in ("linear", "ease_in", "ease_out", "ease_in_out"):
        assert ramp_profile(e, 0.0) == 0.0 and abs(ramp_profile(e, 1.0) - 1.0) < 1e-9
        ps = [ramp_profile(e, i / 1000) for i in range(1001)]
        assert all(b >= a for a, b in zip(ps, ps[1:]))
    slope = lambda e, u: (ramp_profile(e, u + 1e-4) - ramp_profile(e, u - 1e-4)) / 2e-4   # noqa: E731
    assert slope("ease_in", 1e-3) < 0.02 and abs(slope("ease_in", 0.99) - slope("ease_in", 0.6)) < 1e-6
    assert slope("ease_out", 0.999) < 0.02 and abs(slope("ease_out", 0.01) - slope("ease_out", 0.4)) < 1e-6


def test_beat_offset_is_measured_against_the_beat_the_sync_chose():
    p = _plan("plan_1_5_turkey_trot.json")
    for s in p["shots"]:
        s["transition_in"] = {"type": "cut", "duration_s": 0.0}
    plan = _parse(p)
    fps = plan.output.fps
    cuts = [round(sum(x.duration_s for x in plan.shots[:k + 1]) * fps) / fps for k in range(len(plan.shots) - 1)]
    p["music"]["cut_beats_s"] = [cuts[0] + 2 / fps] + cuts[1:]
    rows = c.report(_parse(p))
    assert rows[0]["beat_offset_frames"] == -2.0 and "off_beat" in rows[0]["breaks"]


def test_3d_shots_are_measured_in_the_same_unit():
    plan = parse_plan(_plan("plan_1_6_box_room.json"))
    rows = c.report(plan)
    assert rows and all(r["out"] is not None or plan.shots[r["cut"] - 1].camera.type == "video_clip" for r in rows)
