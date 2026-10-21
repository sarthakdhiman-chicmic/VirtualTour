"""tour_worker.motion.fire: the fire effect, its box and the structure held inside it."""
import os
import tempfile
import types

import cv2
import numpy as np

from tests.perceptual_fixtures import _fireplace_scene, _Spec, _with_firebox
from tour_worker.motion.loader import load_regions
from tour_worker.motion.masks import mask_relpath, sample_bilinear
from tour_worker.motion.region import Region


def test_fire_flicker_moves_the_flames_from_their_base_and_glows():
    from tour_worker.motion.fire import fire_glow
    img, fp = _fireplace_scene(lit=True)
    from tour_worker.vision.fire_detect import flame_mask
    flame = (flame_mask(img, fp) > 0).astype(np.float32)
    glow = fire_glow(flame)
    ys, xs = np.nonzero(flame)
    bb = [xs.min() / 600, ys.min() / 400, (xs.max() + 1) / 600, (ys.max() + 1) / 400]
    spec = _Spec(effect="fire_flicker", anchor="bottom", amp=5.0, bbox=bb, freq=1.0)
    r = Region(spec, glow, 600, 400)
    r.flame, r.flame_h, r.flame_bottom = flame, float(ys.max() - ys.min() + 1), float(ys.max())
    u = np.full(40, float(xs.mean()))
    v = np.linspace(ys.min(), ys.max(), 40)
    peaks, bmin, bmax, prev, steps = [], 9.0, 0.0, None, []
    for f in range(60):
        dx, dy, b = r.fields(u, v, f / 30)
        peaks.append(np.hypot(dx, dy))
        bmin, bmax = min(bmin, b.min()), max(bmax, b.max())
        if prev is not None:
            steps.append(np.hypot(dx - prev[0], dy - prev[1]).max())
        prev = (dx, dy)
    peaks = np.array(peaks)
    assert peaks[:, -1].max() < 0.5                                       # still at the base (logs)
    assert peaks[:, :20].max() > 1.5                                      # the tongues move
    assert peaks.max() <= 5.0 * 1.6 and max(steps) < 2.5                  # bounded, no jumps
    assert 0.7 < bmin and bmax < 1.35 and bmax - bmin > 0.1               # flicker, within range
    # the glow reaches beyond the flames and fades out
    assert glow[ys.min() - 20, int(xs.mean())] > 0.2 and glow[0, 0] == 0.0


def test_fire_tongues_reach_above_the_photos_flame_tips():
    """Review 2026-10-19: "the flames are too small". Points just above the flame tips now sample
    the flames below them (the plume), so the fire licks higher than the photo caught it."""
    from tour_worker.motion.fire import FIRE_PLUME, fire_glow, flame_plume
    img, fp = _fireplace_scene(lit=True)
    from tour_worker.vision.fire_detect import flame_mask
    flame = (flame_mask(img, fp) > 0).astype(np.float32)
    ys, xs = np.nonzero(flame)
    fh = float(ys.max() - ys.min() + 1)
    plume = flame_plume(flame, FIRE_PLUME * fh)
    spec = _Spec(effect="fire_flicker", anchor="bottom", amp=8.0,
                 bbox=[xs.min() / 600, ys.min() / 400, (xs.max() + 1) / 600, (ys.max() + 1) / 400], freq=1.0)
    r = Region(spec, np.maximum(fire_glow(flame), plume), 600, 400)
    r.flame, r.flame_h, r.flame_bottom, r.plume = flame, fh, float(ys.max()), plume
    cx = int(np.median(xs[ys == ys.min()]))
    above = np.array([float(ys.min() - 0.1 * fh)])               # within the (shorter) plume
    assert plume[int(above[0]), cx] > 0.3
    lift = max(-r.fields(np.array([float(cx)]), above, f / 30)[1][0] for f in range(60))
    assert lift > 2.0                                   # the air above the tips draws flame up


def test_fire_moves_only_inside_its_box_and_scales_with_its_size():
    """User 2026-10-21: flames scaled to their own size and the firepit; too much motion warps
    the mantel. Nothing outside the fire's box moves; the strength is capped by flame height."""
    from tour_worker.motion.fire import FIRE_AMP_OF_HEIGHT, fire_box
    img, fp = _fireplace_scene(lit=True)
    flame = (_with_firebox(img, fp)).astype(np.uint8) * 255
    box = fire_box(img, flame)
    assert box[100:170].sum() == 0 and box[:, :225].sum() == 0       # surround / mantel outside
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "images", "masks"))
        cv2.imwrite(os.path.join(td, "images", "x.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(os.path.join(td, "images", mask_relpath("s01", "obj_000")), flame)
        ys, xs = np.nonzero(flame)
        bb = [xs.min() / 600, ys.min() / 400, (xs.max() + 1) / 600, (ys.max() + 1) / 400]
        shot = types.SimpleNamespace(shot_id="s01", motion_regions=[
            _Spec(effect="fire_flicker", anchor="bottom", amp=40.0, bbox=bb, freq=1.0)])
        regions, _ = load_regions(shot, os.path.join(td, "images"), 600, 400, img)
    r = regions[0]
    assert r.amp <= FIRE_AMP_OF_HEIGHT * r.flame_h + 1e-6               # 40 px asked, capped by size
    u, v = np.meshgrid(np.arange(0, 600, 3, dtype=float), np.arange(0, 400, 3, dtype=float))
    for f in range(0, 60, 5):
        dx, dy, _ = r.fields(u.ravel(), v.ravel(), f / 30)
        w = sample_bilinear(r.feather, u.ravel(), v.ravel())
        moved = np.hypot(w * dx, w * dy).reshape(u.shape) > 0.05
        assert not moved[100 // 3:170 // 3].any()                        # the mantel / surround
        assert not moved[:, :225 // 3].any()


def test_a_faint_rod_across_the_opening_is_held():
    """Review 2026-10-21: a dark screen rod across the top of a brick fireplace's opening (dark on
    the dark cavity, so low contrast) warped as the flames licked up into it."""
    from tour_worker.motion.fire import fire_box, fire_structure
    img, fp = _fireplace_scene(lit=True)
    img[206:210, 232:368] = (40, 40, 44)                             # a faint rod inside the box
    flame = _with_firebox(img, fp).astype(np.uint8) * 255
    box = fire_box(img, flame)
    st = fire_structure(img, flame, box)
    assert st[205:211, 260:340].mean() > 0.5                          # the rod is held
    assert not st[240:280, 250:350].all()                             # the flames are not
