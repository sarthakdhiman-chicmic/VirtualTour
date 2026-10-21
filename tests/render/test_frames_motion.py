"""tour_worker.render.frames with motion regions: drawn frames, faint regions, determinism."""
import hashlib
import os
import tempfile

import cv2
import numpy as np
import pytest

from tests.perceptual_fixtures import (
    CURTAIN_BBOX,
    FREQ,
    _frames,
    _plan,
    _region,
    _shot,
    _workdir,
)
from tour_worker.contract.plan import compute_plan_hash, parse_plan


def test_zero_amplitude_matches_the_plain_warp():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _workdir(a, True)
        _workdir(b, False)
        moving = _frames(parse_plan(_plan([_region(amp=0.0)])), a)
        # the same camera without regions (a tiny roll keyframe, as before the direct renderer)
        plain = _plan(None, version="1.3")
        plain["shots"][0]["camera"]["keyframes"][0]["roll_deg"] = 0.0001
        plain["provenance"]["plan_hash"] = compute_plan_hash(plain)
        for fa, fb in zip(moving, _frames(parse_plan(plain), b)):
            assert np.abs(fa.astype(int) - fb.astype(int)).mean() < 0.6


def test_frames_with_motion_move_and_are_deterministic():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
            tempfile.TemporaryDirectory() as still:
        for td in (a, b, still):
            _workdir(td, True)
        fa = _frames(parse_plan(_plan([_region()])), a)
        fb = _frames(parse_plan(_plan([_region()])), b)
        fs = _frames(parse_plan(_plan([_region(amp=0.0)])), still)
        digest = [hashlib.md5(f.tobytes()).hexdigest() for f in fa]
        assert digest == [hashlib.md5(f.tobytes()).hexdigest() for f in fb]
        assert any(np.abs(fm.astype(int) - fz.astype(int)).max() > 20 for fm, fz in zip(fa, fs))


def test_region_with_missing_mask_is_dropped_with_a_warning():
    with tempfile.TemporaryDirectory() as td:
        _workdir(td, False)
        warnings = []
        sw = _shot(parse_plan(_plan([_region()])), td, warnings)
        assert sw.regions == [] and warnings == ["motion_region_dropped:s01:obj_000:missing"]


def _time_slice_motion(effect, amp):
    """Draw 3 s through the real frame path (ShotWarp); stack one row inside the region over time (a
    time-slice). A static row gives straight stripes; return how far the stripes wander (px)."""
    with tempfile.TemporaryDirectory() as td:
        _workdir(td, True, leafy=effect == "canopy_flutter")
        plan = _plan([_region(effect=effect, anchor="none", amp=amp, freq=FREQ[effect])], dur=3.0)
        plan["shots"][0]["camera"] = {"type": "keyframe_path", "move": "hold", "keyframes": [
            {"t": 0, "cx": 0.5, "cy": 0.5, "zoom": 1.0}, {"t": 1, "cx": 0.5, "cy": 0.5, "zoom": 1.0}]}
        plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
        frames = _frames(parse_plan(plan), td)
    row = int(0.55 * 360)
    x0, x1 = int((CURTAIN_BBOX[0] + 0.03) * 640), int((CURTAIN_BBOX[2] - 0.03) * 640)
    sl = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[row, x0:x1].astype(np.float64) for f in frames])
    shifts = []
    for line in sl:                                     # best shift of each line vs the first
        errs = [np.abs(np.roll(line, k)[8:-8] - sl[0][8:-8]).mean() for k in range(-8, 9)]
        shifts.append(abs(int(np.argmin(errs)) - 8))
    return max(shifts), np.abs(sl - sl[0]).max(axis=0).mean(), np.abs(np.diff(sl, axis=0)).mean()


@pytest.mark.parametrize("effect, amp", [("calm_water", 6.0), ("canopy_flutter", 5.0)])
def test_time_slice_shows_visible_motion(effect, amp):
    """The acceptance rule from review: motion that a viewer can see, not just non-zero maths."""
    rejected = _time_slice_motion("water_ripple", 3.0)  # what review called "static"
    wander, change, activity = _time_slice_motion(effect, amp)
    if effect == "calm_water":                          # reflections sway sideways, clearly
        assert wander >= 1 and change > 2 * rejected[1]
    else:                                               # branches sway: the row visibly shifts
        assert wander >= 1 and change > rejected[1]


def test_a_region_too_faint_to_read_is_left_still():
    """Review 2026-10-19: trees whose visible part is trunk / hedge (held still by leaf_weight)
    moved 0-0.6 px. The render leaves such regions still, with a warning, and the quality gate
    measures visibility the same way (tour_worker/render/frames.region_motion_px)."""
    from tour_worker.render.frames import ShotWarp, visible_motion_px
    from tour_worker.render.timeline import compute_timeline
    with tempfile.TemporaryDirectory() as td:
        _workdir(td, True)
        plan = parse_plan(_plan([_region(effect="canopy_flutter", anchor="none", amp=0.3,
                                         freq=FREQ["canopy_flutter"])]))     # far too gentle
        warnings = []
        sw = ShotWarp(plan, plan.shots[0], compute_timeline(plan)[0]["render_frames"],
                      os.path.join(td, "images"), warnings)
    assert sw.regions == []
    assert sw.motion_px["obj_000"] < visible_motion_px(sw.ow, sw.oh)
    assert any(w.startswith("motion_region_dropped:s01:obj_000:faint_") for w in warnings), warnings


def test_parallel_render_is_byte_identical_to_serial():
    from tour_worker.render.direct import render_direct
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for td in (a, b):
            _workdir(td, True)
        plan = parse_plan(_plan([_region()], dur=5.0))      # 150 frames: two encode chunks
        out_a = render_direct(plan, a, workers=1)
        out_b = render_direct(plan, b, workers=3)
        assert open(out_a, "rb").read() == open(out_b, "rb").read()
