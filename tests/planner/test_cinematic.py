"""planner/cinematic.py: every camera it plans keeps the contract on every frame, and the
selection rules hold (category effects, duration fit, variety, fallback)."""

import math

import pytest

from tour_worker.render.camera_path import (
    CompoundPath,
    KeyframePath,
    corner_rms_px,
    frame_margin_out_px,
    frame_spp,
    roll_fit_scale,
)
from tour_worker.planner.cinematic import (
    MIN_DURATION_S,
    REF_DIAG,
    V_MAX_REF,
    Geometry,
    ShotInput,
    plan_cameras,
)

FPS = 30


def shot(
    i,
    room,
    stage="living",
    role="hero",
    dur=6.0,
    cls="wide",
    w=1920,
    h=1280,
    spread=0.45,
    px=0.55,
    py=0.55,
):
    return ShotInput(
        shot_id=f"s{i:02d}",
        src_w=w,
        src_h=h,
        room_type=room,
        narrative_stage=stage,
        role=role,
        duration_s=dur,
        peak_x=px,
        peak_y=py,
        spread=spread,
        coverage_class=cls,
    )


def tour(n=8, dur=6.0, **kw):
    rooms = [
        "living_room",
        "kitchen",
        "dining_room",
        "primary_bedroom",
        "bedroom",
        "living_room",
        "kitchen",
        "bedroom",
    ]
    classes = ["wide", "wide", "medium", "tight", "tight", "medium", "wide", "tight"]
    return [shot(i + 1, rooms[i % 8], cls=classes[i % 8], dur=dur, **kw) for i in range(n)]


def frames_for(shots):
    out, cum, prev = [], 0.0, 0
    for s in shots:
        cum += s.duration_s
        b = round(cum * FPS)
        out.append(b - prev)
        prev = b
    return out


def frames_of(cam, s, n, ow, oh):
    if cam.type == "compound":
        p = CompoundPath(
            cam.start.model_dump(include={"cx", "cy", "zoom"}),
            [x.model_dump() for x in cam.segments],
            cam.timing.model_dump(exclude_none=True),
            s.src_w,
            s.src_h,
            ow,
            oh,
        )
        return p.frames(n), [0.0] * n
    kp = KeyframePath([k.model_dump(exclude_none=True) for k in cam.keyframes])
    return kp.frames(n), kp.rolls(n)


def check_contract(shots, res, ow=1920, oh=1080, speed=1.0):
    for s, cam, n in zip(shots, res.cameras, frames_for(shots), strict=False):
        proj = cam.projection or "crop"
        fr, rolls = frames_of(cam, s, n, ow, oh)
        g = Geometry(s.src_w, s.src_h, ow, oh, proj)
        for (cx, cy, z), r in zip(fr, rolls, strict=False):
            assert (
                frame_margin_out_px(cx, cy, z, s.src_w, s.src_h, ow, oh, proj) >= 1.0 - 1e-6
            ), s.shot_id
            dens = frame_spp(cx, cy, z, s.src_w, s.src_h, ow, oh, proj) / roll_fit_scale(
                math.radians(r), ow, oh
            )
            assert dens >= g.min_spp - 1e-6, s.shot_id
        peak = max(
            corner_rms_px(a, b, s.src_w, s.src_h, ow, oh) for a, b in zip(fr, fr[1:], strict=False)
        )
        assert peak <= V_MAX_REF * math.hypot(ow, oh) / REF_DIAG * speed + 1e-3, (s.shot_id, peak)
        if cam.move == "reveal_aerial_approach":
            assert fr[-1][2] > fr[0][2], f"{s.shot_id}: aerials close in on the house"
        elif cam.move.startswith("push"):
            assert fr[-1][2] > fr[0][2], f"{s.shot_id}: a push-in ends closer than it starts"
        elif cam.move in ("reveal_gimbal_track", "reveal_pedestal"):         # constant zoom
            assert fr[-1][2] <= fr[0][2] + 1e-9, f"{s.shot_id} must not end closer"
        else:
            assert fr[-1][2] < fr[0][2], f"{s.shot_id} must end wider than it starts"


def test_every_frame_keeps_the_contract():
    shots = tour()
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    assert not res.warnings
    check_contract(shots, res)


def test_speed_factor_slows_every_shot():
    shots = tour()
    fast = plan_cameras(shots, frames_for(shots), 1920, 1080, 1.0)
    slow = plan_cameras(shots, frames_for(shots), 1920, 1080, 0.8)
    for a, b in zip(fast.notes, slow.notes, strict=False):
        if a["variant"] == b["variant"]:
            assert b["speed_max_px_f"] <= 0.8 * a["speed_max_px_f"] * 1.02 + 1e-3


def test_category_effects():
    shots = [
        shot(1, "aerial", stage="exterior_open"),
        shot(2, "front_exterior", stage="exterior_open"),
        shot(3, "living_room"),
        shot(4, "primary_bathroom", stage="private", cls="tight"),
        shot(5, "bedroom", stage="private", cls="tight"),
    ]
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    v = [n["variant"] for n in res.notes]
    assert v[0] == "aerial_approach" and v[1] == "rise"
    assert res.cameras[0].type == "keyframe_path" and any(
        k.roll_deg for k in res.cameras[0].keyframes
    )
    assert v[3] in ("pull_back", "pan", "tilt_up", "vertical_glide")  # bathroom table
    check_contract(shots, res)


def test_short_shots_never_get_long_effects():
    shots = tour(dur=4.0)
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    for n in res.notes:
        assert MIN_DURATION_S[n["variant"]] <= 4.0 or n["variant"] == "pull_back", n
    check_contract(shots, res)


def test_variety_no_variant_twice_running():
    shots = tour(n=8)
    v = [n["variant"] for n in plan_cameras(shots, frames_for(shots), 1920, 1080).notes]
    assert all(a != b for a, b in zip(v, v[1:], strict=False)), v


@pytest.mark.parametrize("ow, oh", [(1280, 720), (3840, 2160), (1080, 1920), (1440, 1080)])
def test_other_outputs(ow, oh):
    shots = tour(n=5)
    res = plan_cameras(shots, frames_for(shots), ow, oh)
    check_contract(shots, res, ow, oh)


@pytest.mark.parametrize("w, h", [(1280, 853), (853, 1280), (1920, 1080), (4000, 3000)])
def test_any_photo_size_plans_without_failing(w, h):
    shots = tour(n=5, w=w, h=h)
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    assert len(res.cameras) == 5
    check_contract(shots, res)


def test_deterministic():
    shots = tour()
    a = plan_cameras(shots, frames_for(shots), 1920, 1080, 0.8)
    b = plan_cameras(shots, frames_for(shots), 1920, 1080, 0.8)
    assert [c.model_dump() for c in a.cameras] == [c.model_dump() for c in b.cameras]


@pytest.mark.parametrize("dur", [3.5, 6.0])
def test_aerial_always_zooms_in_toward_the_centre(dur):
    shots = [
        shot(1, "aerial", stage="exterior_open", dur=dur, px=0.1, py=0.9),
        shot(2, "living_room"),
    ]
    res = plan_cameras(shots, frames_for(shots), 1920, 1080)
    cam = res.cameras[0]
    assert cam.move == "reveal_aerial_approach" and not res.warnings
    k = cam.keyframes
    assert k[-1].zoom > k[0].zoom + 0.05
    assert (
        abs(k[-1].cx - 0.5) < 0.02 and abs(k[-1].cy - 0.5) < 0.02
    )  # centre, not the saliency peak
    assert k[0].roll_deg and k[-1].roll_deg == 0  # banked -> level
    check_contract(shots, res)


def test_lateral_glides_replace_the_spiral():
    """slider (2026-10-14) and gimbal_track (2026-10-19) glide one way across the room (crop)."""
    shots = tour(n=8)
    res = plan_cameras(shots, frames_for(shots), 1920, 1080, 0.8)
    variants = [n["variant"] for n in res.notes]
    assert "arc" not in variants
    sliders = [c for c in res.cameras if c.move in ("reveal_slider", "reveal_gimbal_track")]
    assert sliders, variants
    for cam in sliders:
        k = cam.keyframes
        assert cam.projection is None  # crop: reads as the camera moving
        dx = [b.cx - a.cx for a, b in zip(k, k[1:], strict=False)]
        assert all(x > 0 for x in dx) or all(x < 0 for x in dx)  # one direction, no reversal
        assert abs(k[-1].cx - k[0].cx) > 0.05  # a visible glide, not a wobble
    check_contract(shots, res, speed=1.0)
