"""tour_worker.motion.region: effects bounded and smooth, motion stays inside its mask."""
import os
import tempfile

import cv2
import numpy as np
import pytest

from tests.perceptual_fixtures import CURTAIN_BBOX, FREQ, SH, SW, _mask, _Spec
from tour_worker.motion.masks import load_feather
from tour_worker.motion.region import Region, apply_regions
from tour_worker.render.frames import homography_maps


@pytest.mark.parametrize("effect, anchor", [("fabric_sway", "top"), ("foliage_sway", "bottom")])
def test_motion_is_anchored_bounded_and_smooth(effect, anchor):
    r = Region(_Spec(effect=effect, anchor=anchor), np.ones((SH, SW), np.float32), SW, SH)
    x = np.full(50, (CURTAIN_BBOX[0] + 0.1) * SW)
    y = np.linspace(CURTAIN_BBOX[1] * SH, CURTAIN_BBOX[3] * SH, 50)
    peaks, steps = [], []
    prev = None
    for f in range(90):
        dx, dy = r.displacement(x, y, f / 30)
        peaks.append(np.hypot(dx, dy).max())
        if prev is not None:
            steps.append(np.hypot(dx - prev[0], dy - prev[1]).max())
        prev = (dx, dy)
    root = 0 if anchor == "top" else -1                 # the rod / the pot stays put
    dx0, dy0 = r.displacement(x[[root]], y[[root]], 0.7)
    assert abs(dx0[0]) < 1e-9 and abs(dy0[0]) < 1e-9
    assert max(peaks) <= 4.0 * 1.6                      # amplitude 4 px, within the envelope
    assert max(peaks) > 1.0                             # it does move
    assert max(steps) < 0.5                             # < 0.5 px per frame: no jitter


def test_motion_stays_inside_the_mask():
    """Exact check on the sampling maps (before JPEG): only output pixels whose source sample
    lies inside the mask are displaced. (On the written JPEG frames, block ringing can touch
    pixels up to one 8x8 block outside the mask by a few levels.)"""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.png")
        cv2.imwrite(p, _mask())
        feather, _ = load_feather(p, SW, SH, (CURTAIN_BBOX[0] * SW, CURTAIN_BBOX[1] * SH,
                                              CURTAIN_BBOX[2] * SW, CURTAIN_BBOX[3] * SH))
    r = Region(_Spec(), feather, SW, SH)
    from tour_worker.render.frames import crop_roll_homography
    m = np.array(crop_roll_homography(0.5, 0.5, 1.05, 0, SW, SH, 640, 360))
    grid = np.meshgrid(np.arange(640.0), np.arange(360.0))
    for t in (0.0, 0.5, 1.3):
        bx, by = homography_maps(m, grid)
        mx, my = bx.copy(), by.copy()
        apply_regions(mx, my, [r], t)
        moved = (mx != bx) | (my != by)
        inside = cv2.remap(_mask(), bx, by, cv2.INTER_NEAREST) > 0
        assert moved.any() and not moved[~inside].any()


def test_water_ripple_moves_everywhere_smoothly_and_boundedly():
    r = Region(_Spec(effect="water_ripple", anchor="none", amp=3.0), np.ones((SH, SW), np.float32), SW, SH)
    x = np.linspace(CURTAIN_BBOX[0] * SW, CURTAIN_BBOX[2] * SW, 40)
    y = np.full(40, (CURTAIN_BBOX[1] + CURTAIN_BBOX[3]) / 2 * SH)
    prev, steps, peaks = None, [], []
    for f in range(90):
        dx, dy = r.displacement(x, y, f / 30)
        peaks.append(np.hypot(dx, dy).max())
        if prev is not None:
            steps.append(np.hypot(dx - prev[0], dy - prev[1]).max())
        prev = (dx, dy)
    assert 0.3 < max(peaks) <= 3.0 * 1.3
    assert max(steps) < 0.4


@pytest.mark.parametrize("effect, amp", [("calm_water", 6.0), ("canopy_flutter", 5.0)])
def test_new_effects_are_bounded_smooth_and_light_is_gentle(effect, amp):
    r = Region(_Spec(effect=effect, anchor="none", amp=amp, freq=FREQ[effect]), np.ones((SH, SW), np.float32), SW, SH)
    x, y = np.meshgrid(np.linspace(CURTAIN_BBOX[0] * SW, CURTAIN_BBOX[2] * SW, 40),
                       np.linspace(CURTAIN_BBOX[1] * SH, CURTAIN_BBOX[3] * SH, 30))
    x, y = x.ravel(), y.ravel()
    prev, steps, peaks = None, [], []
    for f in range(90):
        dx, dy = r.displacement(x, y, f / 30)
        peaks.append(np.hypot(dx, dy).max())
        if prev is not None:
            steps.append(np.hypot(dx - prev[0], dy - prev[1]).max())
        prev = (dx, dy)
        b = r.brightness(x, y, f / 30)
        if effect == "calm_water":                      # smooth light, never a flicker
            assert np.abs(b - 1.0).max() <= 0.04
        else:                                           # leaves turning: a faint shimmer,
            assert np.abs(b - 1.0).max() <= 0.07        # brighter as a gust front passes
    assert 0.4 * amp < max(peaks) <= 1.6 * amp
    assert max(steps) < 0.25 * amp


def test_planner_phases_make_curtains_and_plants_sway_opposite_ways():
    box = [0.2, 0.1, 0.45, 0.8]
    cur = Region(_Spec(effect="fabric_sway", anchor="top", bbox=box), np.ones((SH, SW), np.float32), SW, SH)
    pl = _Spec(effect="foliage_sway", anchor="bottom", bbox=box)
    pl.phase = cur.phase + np.pi
    plant = Region(pl, np.ones((SH, SW), np.float32), SW, SH)
    assert plant.phase == pytest.approx(cur.phase + np.pi)
    # main sway terms at the same frequency: opposite signs whenever both are clearly moving
    t = np.linspace(0, 8, 200)
    a = np.sin(2 * np.pi * 0.3 * t + cur.phase)
    b = np.sin(2 * np.pi * 0.3 * t + plant.phase)
    big = (np.abs(a) > 0.3) & (np.abs(b) > 0.3)
    assert big.any() and np.all(np.sign(a[big]) != np.sign(b[big]))
