"""Plan 1.3 keyframe_path: camera path math and contract validation."""
import copy

import pytest

from modules.camera_path import (
    PROFILE_MIN_SLOPE, hermite_profile, monotone_eval, monotone_slopes,
)
from modules.plan_io import compute_plan_hash, parse_plan

PATHS = [
    [{"t": 0, "cx": 0.5, "cy": 0.5, "zoom": 1.03}, {"t": 1, "cx": 0.53, "cy": 0.48, "zoom": 1.1}],
    [{"t": 0, "cx": 0.5, "cy": 0.52, "zoom": 1.03}, {"t": 0.35, "cx": 0.38, "cy": 0.47, "zoom": 1.28},
     {"t": 0.7, "cx": 0.62, "cy": 0.47, "zoom": 1.28}, {"t": 1, "cx": 0.58, "cy": 0.49, "zoom": 1.08}],
    [{"t": 0, "cx": 0.45, "cy": 0.5, "zoom": 1.0}, {"t": 0.5, "cx": 0.5, "cy": 0.47, "zoom": 1.08},
     {"t": 1, "cx": 0.52, "cy": 0.44, "zoom": 1.17}],
    [{"t": 0, "cx": 0.4, "cy": 0.5, "zoom": 1.1}, {"t": 0.2, "cx": 0.6, "cy": 0.5, "zoom": 1.1},
     {"t": 0.4, "cx": 0.41, "cy": 0.55, "zoom": 1.3}, {"t": 0.8, "cx": 0.6, "cy": 0.45, "zoom": 1.2},
     {"t": 1, "cx": 0.5, "cy": 0.5, "zoom": 1.05}],
]
T_SAMPLES = [i / 64 for i in range(65)]


def test_profile_endpoints_and_monotonic():
    assert hermite_profile(0) == 0 and abs(hermite_profile(1) - 1) < 1e-12
    assert PROFILE_MIN_SLOPE > 0.76


@pytest.mark.parametrize("kfs", PATHS)
def test_no_overshoot_between_keyframes(kfs):
    ts = [k["t"] for k in kfs]
    for c in ("cx", "cy", "zoom"):
        ys = [k[c] for k in kfs]
        ms = monotone_slopes(ts, ys)
        for a in range(len(ts) - 1):
            lo, hi = min(ys[a], ys[a + 1]), max(ys[a], ys[a + 1])
            for i in range(101):
                x = ts[a] + (ts[a + 1] - ts[a]) * i / 100
                v = monotone_eval(ts, ys, ms, x)
                assert lo - 1e-12 <= v <= hi + 1e-12


def _plan_13(camera: dict, version: str = "1.3") -> dict:
    plan = {
        "plan_version": version, "listing_id": "t", "mode": "cinematic", "preset": "p",
        "output": {"width": 1920, "height": 1080, "fps": 30, "codec": "h264"}, "total_duration_s": 5.0,
        "shots": [{"shot_id": "s01", "order": 1, "asset_id": "a", "image_uri": "x.jpg",
                   "source": {"width": 1920, "height": 1280}, "room_key": "r", "room_type": "r",
                   "narrative_stage": "living", "role": "hero", "duration_s": 5.0, "camera": camera,
                   "transition_in": {"type": "cut", "duration_s": 0.0}}],
        "warnings": [], "provenance": {"scoring_config_version": "0", "planner_config_version": "0"},
    }
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    return plan


KP = {"type": "keyframe_path", "move": "settle", "keyframes": PATHS[0]}


def test_keyframe_path_accepted_in_13():
    plan = parse_plan(_plan_13(KP))
    assert plan.shots[0].camera.type == "keyframe_path"


def test_keyframe_path_rejected_before_13():
    with pytest.raises(ValueError, match="unsupported_camera_type"):
        parse_plan(_plan_13(KP, "1.2"))


def test_unknown_camera_type_rejected():
    with pytest.raises(ValueError, match="unsupported_camera_type"):
        parse_plan(_plan_13({"type": "orbit", "keyframes": PATHS[0]}))


@pytest.mark.parametrize("mutate, msg", [
    (lambda k: k[:1], "2-5 keyframes"),
    (lambda k: k + [dict(k[-1], t=1.0)] * 5, "2-5 keyframes"),
    (lambda k: [dict(k[0], t=0.1), k[1]], "first keyframe"),
    (lambda k: [k[0], dict(k[1], t=0.9)], "last t must be 1"),
    (lambda k: [k[0], dict(k[0], t=0.5), dict(k[0], t=0.5), k[1]], "strictly increasing"),
])
def test_keyframe_rules(mutate, msg):
    cam = copy.deepcopy(KP)
    cam["keyframes"] = mutate(cam["keyframes"])
    with pytest.raises(ValueError, match=msg):
        parse_plan(_plan_13(cam))


def test_path_leaving_image_between_keyframes_rejected():
    # every keyframe is inside, but zoom has zero velocity at its minimum (t=0.5) while cx is
    # still moving through 0.5, so frames next to that keyframe are at zoom ~1 off-centre
    cam = {"type": "keyframe_path", "move": "x", "keyframes": [
        {"t": 0, "cx": 0.30, "cy": 0.5, "zoom": 1.7}, {"t": 0.5, "cx": 0.5, "cy": 0.5, "zoom": 1.0},
        {"t": 1, "cx": 0.70, "cy": 0.5, "zoom": 1.7}]}
    with pytest.raises(ValueError, match="out of bounds"):
        parse_plan(_plan_13(cam))


# ---- projection "rotate" (addendum 1.3.1) ----------------------------------------------------
from modules.camera_path import apply_homography, frame_margin_out_px, frame_spp, rotate_homography  # noqa: E402

ROT_CASES = [(0.5, 0.5, 1.2, 75), (0.56, 0.62, 1.25, 75), (0.43, 0.41, 1.1, 60), (0.6, 0.5, 1.3, 90)]


def test_rotate_equals_crop_at_centre():
    for z in (1.01, 1.2, 1.33):
        assert abs(frame_margin_out_px(0.5, 0.5, z, 1920, 1280, 1920, 1080, "rotate")
                   - frame_margin_out_px(0.5, 0.5, z, 1920, 1280, 1920, 1080)) < 1e-9
        assert abs(frame_spp(0.5, 0.5, z, 1920, 1280, 1920, 1080, "rotate")
                   - frame_spp(0.5, 0.5, z, 1920, 1280, 1920, 1080)) < 1e-6


@pytest.mark.parametrize("cx, cy, z, hfov", ROT_CASES)
def test_rotate_aims_at_target_and_keeps_horizon_level(cx, cy, z, hfov):
    m = rotate_homography(cx, cy, z, 1920, 1280, 1920, 1080, hfov)
    u, v = apply_homography(m, 960, 540)
    assert abs(u - cx * 1920) < 1e-6 and abs(v - cy * 1280) < 1e-6
    # no roll: the output's horizontal centre line maps to a line through the principal row
    # only when pitch is 0; with pitch, verticals lean but the frame's centre column stays vertical
    # in the source when yaw is 0
    if abs(cx - 0.5) < 1e-12:
        top, bottom = apply_homography(m, 960, 0), apply_homography(m, 960, 1080)
        assert abs(top[0] - bottom[0]) < 1e-6


def test_unknown_projection_rejected():
    cam = dict(KP, projection="fisheye")
    with pytest.raises(ValueError, match="unsupported_projection"):
        parse_plan(_plan_13(cam))


def test_rotate_path_bounds_checked():
    # inside as a crop, outside once the camera turns toward the corner
    cam = {"type": "keyframe_path", "move": "x", "projection": "rotate", "keyframes": [
        {"t": 0, "cx": 0.5, "cy": 0.5, "zoom": 1.3}, {"t": 1, "cx": 0.62, "cy": 0.66, "zoom": 1.3}]}
    parse_plan(_plan_13(dict(cam, projection="crop", keyframes=[dict(k, cx=0.6, cy=0.6) if k["t"] else k
                                                                   for k in cam["keyframes"]])))
    with pytest.raises(ValueError, match="out of bounds"):
        parse_plan(_plan_13(cam))


# ---- camera type "compound" (addendum 1.3.2) ---------------------------------------------------
from modules.camera_path import CompoundPath, corner_rms_px, hermite_slope_max  # noqa: E402

COMPOUNDS = [
    ({"cx": 0.5, "cy": 0.5, "zoom": 1.1}, [{"t0": 0, "t1": 0.4, "cx": 0.39, "cy": 0.6, "zoom": 1.3},
     {"t0": 0.26, "t1": 0.76, "cx": 0.61, "cy": 0.6, "zoom": 1.3}, {"t0": 0.62, "t1": 1, "cx": 0.5, "cy": 0.5, "zoom": 1.012}],
     {"s0": 0.5, "s1": 0.5}),
    ({"cx": 0.4, "cy": 0.66, "zoom": 1.32}, [{"t0": 0, "t1": 0.68, "cx": 0.6, "cy": 0.5, "zoom": 1.24},
     {"t0": 0.56, "t1": 1, "cx": 0.5, "cy": 0.5, "zoom": 1.012}], {"s0": 0.4, "s1": 0.7}),
    ({"cx": 0.5, "cy": 0.5, "zoom": 1.03}, [{"t0": 0, "t1": 1, "cx": 0.52, "cy": 0.48, "zoom": 1.14}], {"s0": 1, "s1": 1}),
]


def _frame_speeds(path, n=200):
    fr = path.frames(n)
    return [corner_rms_px(a, b, 1920, 1280, 1920, 1080) for a, b in zip(fr, fr[1:])]


@pytest.mark.parametrize("start, segs, timing", COMPOUNDS)
def test_compound_endpoints_and_speed_profile(start, segs, timing):
    p = CompoundPath(start, segs, timing, 1920, 1280, 1920, 1080)
    a, b = p.at(0), p.at(1)
    assert max(abs(x - y) for x, y in zip(a, (start["cx"], start["cy"], start["zoom"]))) < 1e-12
    assert max(abs(x - y) for x, y in zip(b, (segs[-1]["cx"], segs[-1]["cy"], segs[-1]["zoom"]))) < 1e-9
    sp = _frame_speeds(p)
    mean = p.distance / 199
    # the speed on screen follows the timing curve: s0 * mean at the start, s1 * mean at the cut,
    # never above max(hermite') * mean, and never zero in between (no stop mid-shot)
    assert abs(sp[0] - timing["s0"] * mean) < 0.05 * mean + 0.02
    assert abs(sp[-1] - timing["s1"] * mean) < 0.05 * mean + 0.02
    assert max(sp) <= hermite_slope_max(timing["s0"], timing["s1"]) * mean * 1.03
    assert min(sp) > 0.25 * mean


def test_compound_slope_above_1_slows_the_middle():
    # why the planner caps s0, s1 at 1: at 3 the timing curve stops dead mid-shot
    start, segs, _ = COMPOUNDS[2]
    sp = _frame_speeds(CompoundPath(start, segs, {"s0": 3, "s1": 3}, 1920, 1280, 1920, 1080))
    assert min(sp) < 0.01 * max(sp)


CP = {"type": "compound", "move": "swoop", "start": COMPOUNDS[0][0], "segments": COMPOUNDS[0][1],
      "timing": COMPOUNDS[0][2]}


def test_compound_accepted_in_13_and_rejected_before():
    assert parse_plan(_plan_13(CP)).shots[0].camera.type == "compound"
    with pytest.raises(ValueError, match="unsupported_camera_type"):
        parse_plan(_plan_13(CP, "1.2"))


@pytest.mark.parametrize("mutate, msg", [
    (lambda c: dict(c, segments=[]), "1-4 segments"),
    (lambda c: dict(c, segments=[dict(c["segments"][0], t0=0.1)] + c["segments"][1:]), "first segment t0"),
    (lambda c: dict(c, segments=c["segments"][:-1] + [dict(c["segments"][-1], t1=0.9)]), "last t1"),
    (lambda c: dict(c, segments=[c["segments"][0], dict(c["segments"][1], t1=0.3), c["segments"][2]]),
     "non-decreasing"),
    (lambda c: dict(c, timing={"s0": 3.5, "s1": 1}), "timing s0"),
    (lambda c: {k: v for k, v in c.items() if k != "timing"}, "needs start, segments and timing"),
])
def test_compound_rules(mutate, msg):
    with pytest.raises(ValueError, match=msg):
        parse_plan(_plan_13(mutate(copy.deepcopy(CP))))


def test_compound_out_of_bounds_rejected():
    cam = copy.deepcopy(CP)
    cam["segments"][0].update(cx=0.2, zoom=1.3)   # window leaves the left edge
    with pytest.raises(ValueError, match="out of bounds"):
        parse_plan(_plan_13(cam))
