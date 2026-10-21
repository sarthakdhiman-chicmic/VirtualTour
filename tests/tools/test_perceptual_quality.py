"""tools/quality/perceptual_quality.py: the fast path within the gate's limits; the gate's drag check."""
import os
import tempfile

import cv2
import numpy as np
import pytest

from tests.perceptual_fixtures import FREQ, SH, SW, _plan, _region, _workdir
from tour_worker.contract.plan import parse_plan
from tour_worker.motion.masks import mask_relpath


@pytest.mark.parametrize("effect, anchor, amp", [("fabric_sway", "top", 4.0), ("foliage_sway", "bottom", 4.0),
                                                 ("canopy_flutter", "none", 5.0), ("calm_water", "none", 6.0)])
def test_fast_path_matches_the_exact_path_within_the_quality_limits(effect, anchor, amp):
    """The render's fast path (coarse fields, moving window) against the exact per-pixel path,
    with the limits of tools/quality/perceptual_quality.py."""
    from tools.quality.perceptual_quality import LIMITS, check_shot, failures
    from tour_worker.render.frames import ShotWarp
    from tour_worker.render.timeline import compute_timeline
    with tempfile.TemporaryDirectory() as td:
        _workdir(td, True)
        plan = parse_plan(_plan([_region(effect=effect, anchor=anchor, amp=amp, freq=FREQ.get(effect, 0.3))]))
        sw = ShotWarp(plan, plan.shots[0], compute_timeline(plan)[0]["render_frames"], os.path.join(td, "images"))
        rep = check_shot(sw, 4, None)
    # accuracy only: the visibility floor (weak_motion) is judged on real tours, not on this
    # small synthetic render (and its random photo has no leaves for canopy_flutter to move)
    assert [f for f in failures(rep) if not f.startswith("weak_motion")] == [], rep
    assert rep["field_err_max_px"] <= LIMITS["field_err_max_px"]
    if effect != "canopy_flutter":
        assert rep["motion_p95_px"]["obj_000"] > 0.5


def test_the_gate_catches_a_solid_that_moves_inside_a_tree():
    """2026-10-19: a leaf_weight rule emptied its own solid map, a lantern frame bent and the gate
    passed. The gate now holds tree solids to an independent reference: none may move."""
    from tools.quality.perceptual_quality import LIMITS, check_shot
    from tour_worker.render.frames import ShotWarp
    from tour_worker.render.timeline import compute_timeline
    with tempfile.TemporaryDirectory() as td:
        _workdir(td, True, leafy=True)
        p = os.path.join(td, "images", "x.jpg")
        img = cv2.imread(p)
        x = int(0.3 * SW)
        img[int(0.2 * SH):int(0.7 * SH), x:x + 10] = (35, 35, 38)        # a dark lantern rod
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        plan = parse_plan(_plan([_region(effect="canopy_flutter", anchor="none", amp=8.0,
                                         freq=FREQ["canopy_flutter"])]))
        sw = ShotWarp(plan, plan.shots[0], compute_timeline(plan)[0]["render_frames"],
                      os.path.join(td, "images"))
        good = check_shot(sw, 3, None)
        assert good["solid_moved_share"] <= LIMITS["solid_moved_share"]
        sw.regions[0].feather = (cv2.imread(os.path.join(td, "images", mask_relpath("s01", "obj_000")),
                                            cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)   # the bug
        bad = check_shot(sw, 3, None)
        assert bad["solid_moved_share"] > LIMITS["solid_moved_share"]
