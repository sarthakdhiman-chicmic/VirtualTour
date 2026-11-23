"""Motion blur (client review 2026-11-20, item 7; tour_worker/render/frames.py): a frame at or below
the speed threshold is byte-for-byte the frame without blur, a fast one is blurred, blur is
deterministic, and a shot that cannot blur renders exactly as without it."""
import os
import tempfile

import cv2
import numpy as np
import pytest

from tour_worker.contract.plan import compute_plan_hash, parse_plan
from tour_worker.render import frames as frames_mod
from tour_worker.render.frames import ShotWarp, blur_offsets
from tour_worker.render.timeline import compute_timeline

BLUR = {"shutter": 0.5, "min_px_f": 2.0, "max_samples": 5}


def _plan(blur: bool, kfs, path_s=None):
    out = {"width": 640, "height": 360, "fps": 30, "codec": "h264"}
    if blur:
        out["motion_blur"] = BLUR
    plan = {"plan_version": "1.3", "listing_id": "t", "mode": "cinematic", "preset": "p",
            "output": out, "total_duration_s": 1.0,
            "shots": [{"shot_id": "s01", "order": 1, "asset_id": "a", "image_uri": "x.jpg",
                       "source": {"width": 960, "height": 640}, "room_key": "r", "room_type": "r",
                       "narrative_stage": "living", "role": "hero", "duration_s": 1.0,
                       "camera": {"type": "keyframe_path", "move": "x", "keyframes": kfs,
                                  **({"path_s": path_s, "easing": "linear"} if path_s else {})},
                       "transition_in": {"type": "cut", "duration_s": 0.0}}],
            "warnings": [], "provenance": {"scoring_config_version": "0", "planner_config_version": "0"}}
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    return parse_plan(plan)


STILL = [{"t": 0, "cx": 0.5, "cy": 0.5, "zoom": 1.5}, {"t": 1, "cx": 0.5, "cy": 0.5, "zoom": 1.5}]
FAST = [{"t": 0, "cx": 0.36, "cy": 0.5, "zoom": 1.5}, {"t": 1, "cx": 0.64, "cy": 0.5, "zoom": 1.5}]


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur((rng.random((640, 960, 3)) * 255).astype(np.uint8), (0, 0), 2)
    with tempfile.TemporaryDirectory() as wd:
        os.makedirs(os.path.join(wd, "images"))
        cv2.imwrite(os.path.join(wd, "images", "x.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        yield os.path.join(wd, "images")


def _shot(images, blur, kfs, warnings=None):
    plan = _plan(blur, kfs)
    n = compute_timeline(plan)[0]["render_frames"]
    return ShotWarp(plan, plan.shots[0], n, images, warnings)


def test_offsets_are_symmetric_inside_the_shutter():
    o = blur_offsets(4, 0.5)
    assert o == [-0.1875, -0.0625, 0.0625, 0.1875] and sum(o) == 0


def test_a_still_camera_is_never_blurred(images):
    sw, ref = _shot(images, True, STILL), _shot(images, False, STILL)
    assert sw.blur_n == [1] * sw.n
    for f in (0, 11, sw.n - 1):
        assert sw.frame(f).tobytes() == ref.frame(f).tobytes()


def test_fast_frames_blur_and_slow_frames_stay_exact(images):
    sw, ref = _shot(images, True, FAST), _shot(images, False, FAST)
    fast = [f for f, k in enumerate(sw.blur_n) if k > 1]
    slow = [f for f, k in enumerate(sw.blur_n) if k == 1]
    assert fast, "a 0.28-wide slide in one second must blur"
    assert all(3 <= sw.blur_n[f] <= BLUR["max_samples"] for f in fast)
    f = fast[len(fast) // 2]
    blurred, plain = sw.frame(f), ref.frame(f)
    assert blurred.tobytes() != plain.tobytes()
    # same picture, softer along the motion: the mean barely moves, horizontal detail drops
    assert abs(blurred.astype(float).mean() - plain.astype(float).mean()) < 1.0
    assert np.abs(np.diff(blurred.astype(float), axis=1)).mean() < np.abs(np.diff(plain.astype(float), axis=1)).mean()
    for g in slow[:3]:
        assert sw.frame(g).tobytes() == ref.frame(g).tobytes()


def test_blur_is_deterministic(images):
    a, b = _shot(images, True, FAST), _shot(images, True, FAST)
    f = int(np.argmax(a.blur_n))
    assert a.frame(f).tobytes() == b.frame(f).tobytes()


def test_a_shot_that_cannot_blur_renders_as_without_blur(images, monkeypatch):
    def broken(self, f, samples):
        raise RuntimeError("boom")

    monkeypatch.setattr(frames_mod.ShotWarp, "_blurred", broken)
    warnings = []
    sw, ref = _shot(images, True, FAST, warnings), _shot(images, False, FAST)
    assert sw.blur_n is None
    assert warnings == ["enhancement_dropped:s01:motion_blur:error_RuntimeError"]
    assert all(sw.frame(f).tobytes() == ref.frame(f).tobytes() for f in (0, sw.n // 2, sw.n - 1))


def test_the_blur_budget_keeps_only_the_fastest_frames():
    from types import SimpleNamespace

    from tour_worker.contract.plan import MotionBlur
    from tour_worker.render.direct import blur_budget

    mb = MotionBlur(**BLUR)
    extents = [float(e) for e in range(1, 101)]                  # 100 frames, all over the threshold
    sw = SimpleNamespace(blur_n=[5] * 100, blur_extent=extents, ow=1920, oh=1080)
    warnings = []
    out = blur_budget([sw], 100, mb, warnings)                   # 400 extra draws; budget 25
    assert out.max_samples == 3
    floor = out.min_px_f * out.shutter                           # 1080p: extent at the threshold
    assert sum(e > floor for e in extents) * (out.max_samples - 1) <= 25
    assert sum(e > floor for e in extents) == 12                 # the 12 fastest frames
    assert warnings == [f"motion_blur_budget:max_samples=3,min_px_f={out.min_px_f}"]
    calm = SimpleNamespace(blur_n=[1] * 95 + [3] * 5, blur_extent=extents, ow=1920, oh=1080)
    assert blur_budget([calm], 100, mb, []) is mb                # within budget: unchanged


def test_continuous_cut_shots_blur_on_their_own_timing(images):
    """A shot with path_s (continuous cuts): the shutter's sub-frame cameras follow the steady pace,
    so a frame's blur is centred on that frame's own camera (the sharp frame lies inside the smear)."""
    plan = _plan(True, FAST, path_s=1.0)
    n = compute_timeline(plan)[0]["render_frames"]
    sw = ShotWarp(plan, plan.shots[0], n, images, [])
    assert sw.blur_n is not None and max(sw.blur_n) > 1
    f = n // 2
    (cx, cy, z), _ = sw.state_at(f)
    (cx0, _, _), _ = sw.frames[f]
    assert abs(cx - cx0) < 1e-9
    (a, _, _), _ = sw.state_at(f - 0.25)
    (b, _, _), _ = sw.state_at(f + 0.25)
    assert a < cx < b and abs((cx - a) - (b - cx)) < 1e-6        # a steady pace: symmetric about the frame
