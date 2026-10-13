"""Plan 1.3 keyframe_path: camera path math, Python/TypeScript parity and contract validation."""
import copy
import json
import os
import subprocess
import tempfile

import pytest

from modules.camera_path import (
    KeyframePath, PROFILE_MIN_SLOPE, hermite_profile, monotone_eval, monotone_slopes,
)
from modules.plan_io import compute_plan_hash, parse_plan

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTION = os.path.join(REPO, "remotion")

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


def test_python_typescript_parity():
    with tempfile.TemporaryDirectory() as td:
        bundle = os.path.join(td, "math.cjs")
        subprocess.run([os.path.join(REMOTION, "node_modules", ".bin", "esbuild"), "src/math.ts", "--bundle",
                        "--format=cjs", "--platform=node", f"--outfile={bundle}"], cwd=REMOTION, check=True,
                       capture_output=True)
        script = (f"const m=require({json.dumps(bundle)});const P={json.dumps(PATHS)};const T={json.dumps(T_SAMPLES)};"
                  "console.log(JSON.stringify(P.map(k=>T.map(t=>{const c=m.keyframePathAt(k,t);return [c.cx,c.cy,c.zoom]}))))")
        out = json.loads(subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout)
    worst = 0.0
    for kfs, ts_vals in zip(PATHS, out):
        path = KeyframePath(kfs)
        for t, js in zip(T_SAMPLES, ts_vals):
            worst = max(worst, max(abs(a - b) for a, b in zip(path.at(t), js)))
    assert worst < 1e-12, worst


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
