"""Sky motion (tour_worker/motion/sky.py, effect sky_drift): the sky drifts and billows as a layer
under everything - nothing outside the sky ever moves, the refined sky keeps roofs and leaves out,
the drift is centred on the shot, frames are deterministic - and the effect's way through
segmentation, the planner and the renderer (a sky that cannot be prepared stays still)."""
import os
import tempfile

import cv2
import numpy as np

from tests.perceptual_fixtures import SH, SW, _plan, _region
from tour_worker.contract.plan import parse_plan
from tour_worker.motion.masks import mask_relpath
from tour_worker.motion.sky import SkyMotion, refine_alpha
from tour_worker.render.frames import ShotWarp
from tour_worker.render.timeline import compute_timeline

SKY_Y = int(SH * 0.45)


def _scene():
    """A blue sky with white clouds over a blue-grey textured slate roof (left; the model's mask
    takes in its top 15 px, as photo 1's did) and a green tree with a gap of sky in it (right). The
    model's mask: everything above SKY_Y."""
    rng = np.random.default_rng(1)
    img = np.zeros((SH, SW, 3), np.uint8)
    for y in range(SKY_Y):
        img[y] = (235 - y // 6, 175 - y // 10, 110)                   # BGR: a bright blue gradient
    for cx, cy, r in ((200, 90, 50), (520, 140, 60), (780, 70, 45)):
        cv2.circle(img, (cx, cy), r, (245, 245, 245), -1)              # clouds
    img = cv2.GaussianBlur(img, (0, 0), 3)
    roof = rng.integers(0, 30, (SH - SKY_Y + 15, SW // 2, 3)).astype(np.int16) + np.array([128, 115, 105])
    img[SKY_Y - 15:, : SW // 2] = np.clip(roof, 0, 255)               # slate: blue-grey, textured, at the mask edge
    img[SKY_Y - 60:, SW // 2:] = (40, 120, 50)                          # tree: green
    img[SKY_Y - 40:SKY_Y - 20, int(SW * 0.7):int(SW * 0.74)] = img[SKY_Y - 100, int(SW * 0.72)]   # sky in a gap
    coarse = np.zeros((SH, SW), np.uint8)
    coarse[:SKY_Y] = 255
    return img, coarse


def test_the_refined_sky_keeps_the_slate_roof_out_and_takes_the_gap_in_the_tree():
    img, coarse = _scene()
    a = refine_alpha(img, coarse)
    # roof pixels inside the model's mask (past the 1-2 px feathered edge, which the plate never carries)
    roof_band = a[SKY_Y - 13:SKY_Y, 20:SW // 2 - 20]
    assert (roof_band > 0.5).mean() < 0.05
    below = a[SKY_Y + 5:, : SW // 2]
    assert (below > 0.5).mean() == 0.0                                  # nothing under the mask edge on the roof
    assert a[SKY_Y - 30, int(SW * 0.72)] > 0.5                          # the sky seen through the tree
    assert a[40, 300] > 0.9                                             # open sky


def test_only_the_sky_moves_and_the_drift_is_centred():
    img, coarse = _scene()
    sky = SkyMotion(img, coarse, effects=("drift",), duration=4.0)
    still = (sky.alpha == 0) & (sky.edge == 0)                          # an edge pixel is part sky
    for t in (0.0, 1.3, 4.0):
        f = sky.frame(t)
        assert np.array_equal(f[still], img[still])                     # roof, tree: untouched
    mid = sky.frame(2.0)                                                # the photo itself, half-way
    inner = cv2.erode((sky.alpha > 0.99).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert np.abs(mid[inner].astype(int) - img[inner].astype(int)).max() <= 1
    assert np.abs(sky.frame(0.0)[inner].astype(int) - img[inner].astype(int)).mean() > 1.0   # moved


def _thin_scene():
    """Thin things the model's sky mask takes in whole (review 2026-10-27: "the thinner objects
    are selected and then moved with the clouds"): a branch reaching deep into the sky, a twig
    across a cloud, a power line, and a blue-grey gutter under a dark eave (its colour passes a
    loose "blue" test). The model's mask: everything above SKY_Y but the eave. Returns the image,
    the mask and each thing's centre line."""
    img = np.zeros((SH, SW, 3), np.uint8)
    for y in range(SKY_Y):
        img[y] = (235 - y // 6, 175 - y // 10, 110)
    for cx, cy, r in ((200, 90, 50), (520, 140, 60)):
        cv2.circle(img, (cx, cy), r, (245, 245, 245), -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    img[SKY_Y:] = (40, 120, 50)                                         # trees along the bottom
    img[SKY_Y - 80:SKY_Y, 40:300] = (60, 50, 45)                         # eave
    img[SKY_Y - 100:SKY_Y - 80, 40:300] = (138, 128, 118)                # gutter: blue-grey metal
    lines = {"branch": ((700, SKY_Y), (760, SKY_Y - 170), 3), "twig": ((480, SKY_Y), (540, 110), 2),
             "wire": ((0, 60), (SW, 80), 1)}
    for (p, q, th) in lines.values():
        cv2.line(img, p, q, (45, 60, 75), th, cv2.LINE_AA)
    coarse = np.zeros((SH, SW), np.uint8)
    coarse[:SKY_Y] = 255
    coarse[SKY_Y - 80:SKY_Y, 40:300] = 0
    centre = {}
    for name, (p, q, _) in lines.items():
        c = np.zeros((SH, SW), np.uint8)
        cv2.line(c, p, q, 255, 1)
        centre[name] = (c > 0) & (np.arange(SH)[:, None] < SKY_Y - 2)
    centre["gutter"] = np.zeros((SH, SW), bool)
    centre["gutter"][SKY_Y - 98:SKY_Y - 82, 42:298] = True
    return img, coarse, centre


def test_thin_things_in_the_models_sky_stay_put():
    img, coarse, centre = _thin_scene()
    a = refine_alpha(img, coarse)
    for name, c in centre.items():
        assert a[c].max() == 0, name                                    # none of it is sky
    sky = SkyMotion(img, coarse, effects=("drift",), duration=4.0)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(int)
    darkest_sky = L[:SKY_Y - 100][a[:SKY_Y - 100] > 0.99].min()
    things = cv2.dilate(np.any(list(centre.values()), axis=0).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    for t in (0.0, 4.0):
        f = sky.frame(t)
        for name, c in centre.items():
            assert np.abs(f[c].astype(int) - img[c]).max() <= 15, name    # still there (sky would be 60+ off)
        Lf = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0].astype(int)
        # nothing dark drifts off into the sky (a copy of a twig carried by the clouds)
        assert ((Lf < darkest_sky - 20) & (a > 0.5) & ~things).sum() == 0


def test_sky_frames_are_deterministic():
    img, coarse = _scene()
    a = SkyMotion(img, coarse, seed=11, duration=3.0).frame(1.7)
    b = SkyMotion(img, coarse, seed=11, duration=3.0).frame(1.7)
    assert a.tobytes() == b.tobytes()


def test_segmentation_makes_one_sky_region_of_all_its_pieces():
    from tour_worker.vision.segment import segments_to_regions
    seg = np.zeros((100, 200), np.int32)
    seg[:30, :80] = 1                    # sky through one opening ...
    seg[:30, 120:] = 2                   # ... and another
    seg[50:, :] = 3
    info = [{"id": 1, "label_id": 0, "score": 0.9}, {"id": 2, "label_id": 0, "score": 0.8},
            {"id": 3, "label_id": 1, "score": 0.9}]
    regions = segments_to_regions(seg, info, {0: "sky-other-merged", 1: "tree-merged"}, (200, 100))
    skies = [r for r in regions if r["class_name"] == "sky"]
    assert len(skies) == 1 and skies[0]["bbox"] == [0, 0, 200, 30]
    assert int((skies[0]["mask"] > 0).sum()) == 30 * 80 + 30 * 80


def test_the_planner_maps_a_sky_to_sky_drift():
    from tour_worker.planner.config import PlannerConfig
    from tour_worker.planner.perceptual import plan_motion_regions
    from tour_worker.planner.schemas import CameraParams, Keyframe
    cfg = PlannerConfig()
    cam = CameraParams(type="keyframe_path", move="x", keyframes=[Keyframe(t=0, cx=0.5, cy=0.5, zoom=1.1),
                                                                    Keyframe(t=1, cx=0.5, cy=0.5, zoom=1.0)])
    det = {"object_id": "seg_000", "class_name": "sky", "confidence": 0.9, "bbox": [0, 0, 1536, 300],
           "mask_url": "s3://b/sky.png", "space": [1536, 1152]}
    regions = plan_motion_regions("L", "s01", cam, [det], 1536, 1152, 1920, 1080, cfg, room_type="front_exterior")
    assert [r.effect for r in regions] == ["sky_drift"]
    assert regions[0].amplitude_px == round(min(max(0.0028 * 1536, 2.0), 10.0), 3)
    small = dict(det, bbox=[0, 0, 200, 100])                           # 1% of the photo: too little sky
    assert plan_motion_regions("L", "s01", cam, [small], 1536, 1152, 1920, 1080, cfg) == []


def _render_sky(with_mask: bool):
    img, coarse = _scene()
    plan = parse_plan(_plan([_region(effect="sky_drift", amp=3.0, anchor="none", bbox=[0, 0, 1, 0.45], freq=0.12)],
                            dur=2.0))
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "masks"))
        cv2.imwrite(os.path.join(d, "x.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if with_mask:
            cv2.imwrite(os.path.join(d, mask_relpath("s01", "obj_000")), coarse)
        warnings = []
        n = compute_timeline(plan)[0]["render_frames"]
        sw = ShotWarp(plan, plan.shots[0], n, d, warnings)
        return sw, [sw.frame(f) for f in (0, n - 1)], warnings


def test_the_renderer_draws_the_moving_sky_under_everything():
    sw, (first, last), warnings = _render_sky(True)
    assert sw.sky is not None and sw.regions == [] and not warnings      # a layer, not a displaced region
    top = slice(0, int(360 * 0.3))                                       # the output's sky band
    assert np.abs(first[top].astype(int) - last[top].astype(int)).mean() > 1.0


def test_a_sky_without_its_mask_stays_still_with_a_warning():
    sw, _, warnings = _render_sky(False)
    assert sw.sky is None and warnings == ["motion_region_dropped:s01:obj_000:missing"]
