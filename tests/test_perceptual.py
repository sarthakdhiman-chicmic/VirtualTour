"""Plan 1.4 perceptual motion: noise, effects, frame drawing (ShotWarp) and plan validation."""
import copy
import hashlib
import os
import tempfile
import types

import cv2
import numpy as np
import pytest

from tour_worker.motion.loader import load_regions
from tour_worker.motion.masks import load_feather, mask_relpath, sample_bilinear
from tour_worker.motion.region import Region, apply_regions
from tour_worker.motion.noise import GradientNoise3D
from tour_worker.contract.plan import compute_plan_hash, parse_plan
from tour_worker.render.frames import ShotWarp, homography_maps
from tour_worker.render.timeline import compute_timeline

SW, SH = 960, 640
CURTAIN_BBOX = [0.2, 0.1, 0.45, 0.8]
FREQ = {"calm_water": 0.2, "canopy_flutter": 1.1, "water_ripple": 0.2}     # the backend's values


def _region(effect="fabric_sway", amp=4.0, anchor="top", bbox=CURTAIN_BBOX, seed=7, freq=0.3):
    return {"id": "obj_000", "effect": effect, "mask_uri": "s3://b/m.png", "bbox": bbox,
            "amplitude_px": amp, "freq_hz": freq, "anchor": anchor, "seed": seed, "label": "curtain"}


def _plan(regions, version="1.4", dur=1.0):
    cam = {"type": "keyframe_path", "move": "reveal_crane",
           "keyframes": [{"t": 0, "cx": 0.5, "cy": 0.55, "zoom": 1.1}, {"t": 1, "cx": 0.5, "cy": 0.5, "zoom": 1.03}]}
    plan = {"plan_version": version, "listing_id": "t", "mode": "cinematic", "preset": "p",
            "output": {"width": 640, "height": 360, "fps": 30, "codec": "h264"}, "total_duration_s": dur,
            "shots": [{"shot_id": "s01", "order": 1, "asset_id": "a", "image_uri": "x.jpg",
                       "source": {"width": SW, "height": SH}, "room_key": "r", "room_type": "living_room",
                       "narrative_stage": "living", "role": "hero", "duration_s": dur, "camera": cam,
                       "transition_in": {"type": "cut", "duration_s": 0.0}, "motion_regions": regions}],
            "warnings": [], "provenance": {"scoring_config_version": "0", "planner_config_version": "0"}}
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    return plan


def _photo():
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur((rng.random((SH, SW, 3)) * 255).astype(np.uint8), (0, 0), 2)
    for k in range(0, SW, 24):                         # vertical stripes: motion is easy to see
        cv2.line(img, (k, 0), (k, SH), (100, 100, 100), 2)    # mid grey: not a "metal" solid
    return img


def _mask():
    m = np.zeros((SH, SW), np.uint8)
    x1, y1, x2, y2 = (int(CURTAIN_BBOX[0] * SW), int(CURTAIN_BBOX[1] * SH),
                      int(CURTAIN_BBOX[2] * SW), int(CURTAIN_BBOX[3] * SH))
    m[y1:y2, x1:x2] = 255
    return m


def _leafy(img):
    """The test photo in foliage colours (canopy_flutter only moves leaves)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0], hsv[..., 1] = 55, np.maximum(hsv[..., 1], 120)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _workdir(td, regions, leafy=False):
    os.makedirs(os.path.join(td, "images", "masks"))
    photo = _leafy(_photo()) if leafy else _photo()
    cv2.imwrite(os.path.join(td, "images", "x.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if regions:
        cv2.imwrite(os.path.join(td, "images", mask_relpath("s01", "obj_000")), _mask())


def _shot(plan, td, warnings=None):
    """The plan's first shot as the renderer draws it (images in <td>/images)."""
    return ShotWarp(plan, plan.shots[0], compute_timeline(plan)[0]["render_frames"], os.path.join(td, "images"),
                    warnings)


def _frames(plan, td):
    sw = _shot(plan, td)
    return [sw.frame(f) for f in range(sw.n)]


# ---- noise ------------------------------------------------------------------------------------

def test_noise_is_seeded_bounded_and_smooth():
    a, b, c = GradientNoise3D(1), GradientNoise3D(1), GradientNoise3D(2)
    xs = np.linspace(0, 8, 400)
    assert np.array_equal(a(xs, xs * 0.3, 0.5), b(xs, xs * 0.3, 0.5))
    assert not np.array_equal(a(xs, xs * 0.3, 0.5), c(xs, xs * 0.3, 0.5))
    v = a(xs, xs * 0.7, xs * 0.2)
    assert np.abs(v).max() <= 1.0
    assert np.abs(np.diff(a(0.3, 0.4, np.linspace(0, 2, 2001)))).max() < 0.01   # continuous in time


# ---- effects ----------------------------------------------------------------------------------

def test_feather_is_zero_at_the_mask_edge_and_full_inside():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.png")
        cv2.imwrite(p, _mask())
        f, _ = load_feather(p, SW, SH, (CURTAIN_BBOX[0] * SW, CURTAIN_BBOX[1] * SH,
                                        CURTAIN_BBOX[2] * SW, CURTAIN_BBOX[3] * SH))
    m = _mask() > 0
    assert f[~m].max() == 0.0
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert f[edge].max() < 0.1
    assert f[m].max() == pytest.approx(1.0)


def test_mask_of_a_different_aspect_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.png")
        cv2.imwrite(p, np.full((500, 500), 255, np.uint8))
        assert load_feather(p, SW, SH, (0, 0, SW, SH)) == (None, "aspect")


class _Spec:
    def __init__(self, **kw):
        self.__dict__.update(_region(**kw))


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


# ---- prewarp ----------------------------------------------------------------------------------

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


# ---- plan validation --------------------------------------------------------------------------

def test_plan_14_validation():
    assert parse_plan(_plan([_region()])).shots[0].motion_regions[0].effect == "fabric_sway"
    with pytest.raises(ValueError, match="need plan_version 1.4"):
        parse_plan(_plan([_region()], version="1.3"))
    with pytest.raises(ValueError, match="unsupported_motion_effect"):
        parse_plan(_plan([_region(effect="explode")]))
    with pytest.raises(ValueError, match="amplitude"):
        parse_plan(_plan([_region(amp=50.0)]))
    bad = copy.deepcopy(_region())
    bad["bbox"] = [0.5, 0.1, 0.4, 0.8]
    with pytest.raises(ValueError, match="bbox"):
        parse_plan(_plan([bad]))


def test_a_mask_that_belongs_to_another_object_is_rejected():
    """A curtain detection whose stored mask is a different object (seen in production data)."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.png")
        other = np.zeros((SH, SW), np.uint8)
        other[500:600, 700:900] = 255                      # far from the curtain bbox
        cv2.imwrite(p, other)
        box = (CURTAIN_BBOX[0] * SW, CURTAIN_BBOX[1] * SH, CURTAIN_BBOX[2] * SW, CURTAIN_BBOX[3] * SH)
        assert load_feather(p, SW, SH, box) == (None, "mismatch")
        # a correct mask with a few stray pixels elsewhere is kept, and the strays never move
        good = _mask()
        good[600:604, 900:904] = 255
        cv2.imwrite(p, good)
        f, reason = load_feather(p, SW, SH, box)
        assert reason == "" and f[600:604, 900:904].max() == 0.0


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


def test_segments_to_regions_splits_stuff_and_maps_classes():
    from tour_worker.vision.segment import segments_to_regions
    seg = np.zeros((200, 300), np.int32)
    seg[20:180, 10:60] = 1        # curtain panel left
    seg[20:180, 240:290] = 1      # curtain panel right (same COCO "curtain" segment)
    seg[100:190, 120:180] = 2     # potted plant
    seg[0:3, 0:3] = 3             # a wall: not a class we animate
    info = [{"id": 1, "label_id": 10, "score": 0.99}, {"id": 2, "label_id": 20, "score": 0.95},
            {"id": 3, "label_id": 30, "score": 0.99}]
    from tour_worker.vision.segment import MODELS
    regs = segments_to_regions(seg, info, {10: "curtain", 20: "potted plant", 30: "wall-other-merged"}, (300, 200),
                               class_map=MODELS["coco"]["class_map"])
    assert sorted(r["class_name"] for r in regs) == ["curtain", "curtain", "potted_plant"]
    assert all(r["space"] == [300, 200] and r["mask"].shape == (200, 300) for r in regs)
    left = [r for r in regs if r["class_name"] == "curtain" and r["bbox"][0] < 100][0]
    assert left["bbox"] == [10, 20, 60, 180] and left["mask"][100, 30] == 255 and left["mask"][100, 260] == 0
    assert [r["object_id"] for r in regs] == ["seg_000", "seg_001", "seg_002"]


def test_run_segment_writes_masks_and_returns_json_only():
    from tour_worker.jobs.segment import run_segment

    class FakeStorage:
        def __init__(self):
            self.put_calls = []

        def get(self, uri, dest):
            cv2.imwrite(dest + ".png", _photo())
            os.replace(dest + ".png", dest)

        def put(self, path, uri):
            self.put_calls.append(uri)

    def fake_segmenter(img):
        m = np.zeros(img.shape[:2], np.uint8)
        m[50:300, 200:400] = 255
        return [{"object_id": "seg_000", "class_name": "curtain", "confidence": 0.99,
                 "bbox": [200, 50, 400, 300], "mask": m, "space": [img.shape[1], img.shape[0]]}]

    st = FakeStorage()
    with tempfile.TemporaryDirectory() as td:
        res = run_segment({"images": [{"photo_id": "img_0", "uri": "s3://b/a.jpg"}],
                           "output_prefix": "s3://b/seg/job1"}, st, td, segmenter=fake_segmenter)
    assert res["status"] == "completed"
    r = res["results"]["img_0"][0]
    assert "mask" not in r and r["mask_url"] == "s3://b/seg/job1/img_0/seg_000.png"
    assert st.put_calls == ["s3://b/seg/job1/img_0/seg_000.png"]


# ---- speed paths: parallel frames, coarse fields, windowed drawing ------------------------------

def test_parallel_render_is_byte_identical_to_serial():
    from tour_worker.render.direct import render_direct
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for td in (a, b):
            _workdir(td, True)
        plan = parse_plan(_plan([_region()], dur=5.0))      # 150 frames: two encode chunks
        out_a = render_direct(plan, a, workers=1)
        out_b = render_direct(plan, b, workers=3)
        assert open(out_a, "rb").read() == open(out_b, "rb").read()


@pytest.mark.parametrize("effect, anchor, amp", [("fabric_sway", "top", 4.0), ("foliage_sway", "bottom", 4.0),
                                                 ("canopy_flutter", "none", 5.0), ("calm_water", "none", 6.0)])
def test_fast_path_matches_the_exact_path_within_the_quality_limits(effect, anchor, amp):
    """The render's fast path (coarse fields, moving window) against the exact per-pixel path,
    with the limits of tools/quality/perceptual_quality.py."""
    from tour_worker.render.frames import ShotWarp
    from tour_worker.render.timeline import compute_timeline
    from tools.quality.perceptual_quality import LIMITS, check_shot, failures
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


# ---- review 2026-10-16: leaves only on trees; curtains and plants sway opposite ways ------------

def test_trunks_and_open_sky_stay_still_in_trees():
    from tour_worker.motion.canopy import leaf_weight
    rng = np.random.default_rng(1)
    img = np.zeros((300, 200, 3), np.uint8)
    img[:] = (230, 180, 120)                                         # sky (BGR: blue)
    crown = (np.clip(rng.normal(0, 40, (150, 200)), -80, 80)[..., None] + [40, 140, 60]).clip(0, 255)
    img[:150] = crown.astype(np.uint8)                               # textured green leaves
    img[150:, 85:115] = (40, 70, 110)                                # smooth brown trunk
    mask = np.zeros((300, 200), np.float32)
    mask[:150] = 1.0
    mask[150:, 60:140] = 1.0                                         # trunk + sky beside it
    w, solid = leaf_weight(img, mask)
    assert w[20:130, 20:180].mean() > 0.6                            # crown moves
    assert w[170:, 90:110].max() < 0.1                               # trunk does not
    assert w[170:, 62:80].max() < 0.1                                # nor the sky in the mask
    assert w[mask == 0].max() == 0


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



def test_plan_15_with_music_is_accepted():
    p = _plan([_region()], version="1.5")
    p["music"] = {"track_id": "t", "uri": "s3://m/t.mp3", "sha256": "0" * 64, "end_s": 1.0, "synced": True}
    p["provenance"]["plan_hash"] = compute_plan_hash(p)
    plan = parse_plan(p)
    assert plan.plan_version == "1.5" and plan.shots[0].motion_regions[0].effect == "fabric_sway"


def test_solid_objects_in_a_tree_stay_rigid_and_hedges_still():
    """Review 2026-10-16: lantern frames inside a tree mask bent; hedges boiled."""
    from tour_worker.motion.canopy import leaf_weight
    rng = np.random.default_rng(2)
    H, W = 600, 400
    img = (np.clip(rng.normal(0, 40, (H, W)), -80, 80)[..., None] + [40, 140, 60]).clip(0, 255).astype(np.uint8)
    img[150:350, 180:186] = (35, 35, 38)                             # a dark grey lantern rod
    mask = np.zeros((H, W), np.float32)
    mask[:450] = 1.0                                                 # the tree (crown above)
    mask[520:560, 50:350] = 1.0                                      # a separate low hedge row
    reach = 8.0
    w, solid = leaf_weight(img, mask, reach)
    assert solid[150:350, 180:186].mean() > 0.75                     # the rod is solid
    # a leaf at distance d from the rod moves at most w x reach: it must never reach the rod
    for d in range(1, 3 * int(reach)):
        assert w[160:340, 185 + d].max() * reach < d
    assert w[60:200, 300:380].mean() > 0.5                           # the crown away from it moves
    assert w[520:560].max() == 0.0                                   # the hedge row is still



def test_scene_summary_lists_every_class_with_its_place():
    from tour_worker.vision.segment import scene_summary
    seg = np.zeros((100, 200), np.int32)
    seg[60:100, 40:160] = 1                    # a bed, lower middle
    seg[0:30, :] = 2                           # ceiling across the top
    seg[10:50, 170:200] = 3                    # a window, right
    seg[0:1, 0:1] = 4                          # a speck: below 0.2%, left out
    info = [{"id": 1, "label_id": 0, "score": 0.9}, {"id": 2, "label_id": 1, "score": 0.95},
            {"id": 3, "label_id": 2, "score": 0.8}, {"id": 4, "label_id": 3, "score": 0.9}]
    labels = {0: "bed", 1: "ceiling-merged", 2: "window-other", 3: "clock"}
    out = scene_summary(seg, info, labels, (200, 100))
    c = out["classes"]
    assert set(c) == {"bed", "ceiling-merged", "window-other"}
    assert c["bed"]["area"] == pytest.approx(0.24) and c["bed"]["bbox"] == [0.2, 0.6, 0.8, 1.0]
    assert c["bed"]["centroid"] == pytest.approx([0.4975, 0.795], abs=1e-3)
    assert c["window-other"]["centroid"][0] > 0.85


# ---- lit fires (tour_worker/vision/fire_detect.py, "fire_flicker") -----------------------------------

def _fireplace_scene(lit=True):
    rng = np.random.default_rng(5)
    img = np.full((400, 600, 3), (150, 170, 190), np.uint8)               # wall
    img[80:330, 180:420] = (120, 150, 185)                                # warm stone surround
    img[80:330, 180:420] += rng.integers(0, 30, (250, 240, 1)).astype(np.uint8)
    img[180:320, 230:370] = (25, 25, 28)                                  # dark firebox + logs
    if lit:
        for x in range(245, 355, 22):                                     # flame tongues on the logs
            cv2.ellipse(img, (x + 8, 255), (9, 28), 0, 0, 360, (40, 150, 245), -1)
            cv2.ellipse(img, (x + 8, 262), (4, 14), 0, 0, 360, (120, 230, 255), -1)
    fireplace = np.zeros((400, 600), bool)
    fireplace[80:330, 180:420] = True
    return img, fireplace


def test_lit_fire_is_found_inside_the_fireplace_and_warm_stone_is_not():
    from tour_worker.vision.fire_detect import flame_mask
    img, fp = _fireplace_scene(lit=True)
    fm = flame_mask(img, fp) > 0
    assert fm[225:290, 240:360].mean() > 0.15                             # the flames
    assert fm[80:170].sum() == 0 and fm[:, :230].sum() == 0               # not the stone surround
    img, fp = _fireplace_scene(lit=False)
    assert flame_mask(img, fp).sum() == 0                                 # unlit: nothing


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


def test_ade20k_class_map_covers_our_classes_and_fireplaces():
    from tour_worker.vision.segment import MODELS, segments_to_regions
    seg = np.zeros((200, 300), np.int32)
    seg[20:180, 10:60] = 1
    seg[100:190, 120:180] = 2
    seg[60:140, 200:280] = 3
    info = [{"id": 1, "label_id": 18, "score": 0.9}, {"id": 2, "label_id": 17, "score": 0.9},
            {"id": 3, "label_id": 49, "score": 0.9}]
    labels = {18: "curtain", 17: "plant", 49: "fireplace"}
    regs = segments_to_regions(seg, info, labels, (300, 200), class_map=MODELS["ade20k"]["class_map"])
    assert sorted(r["class_name"] for r in regs) == ["curtain", "fireplace", "potted_plant"]



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


# ---- review 2026-10-19: coherent tree sway, complete foliage, far trees, bigger fire, far water --

def test_trees_sway_sideways_together_not_in_circles():
    """The earlier field (independent x / y noise at branch scale) read as "a circular warp around
    the trees". The crown now sways sideways as one: mostly horizontal, neighbouring parts moving
    the same way, the sway travelling across the tree."""
    r = Region(_Spec(effect="canopy_flutter", anchor="none", amp=8.0, bbox=[0.2, 0.1, 0.6, 0.6], freq=1.1),
               np.ones((SH, SW), np.float32), SW, SH)
    u, v = np.meshgrid(np.linspace(0.25 * SW, 0.55 * SW, 12), np.linspace(0.15 * SH, 0.55 * SH, 10))
    u, v = u.ravel(), v.ravel()
    dxs, dys, same = [], [], []
    for f in range(0, 150, 3):
        dx, dy = r.displacement(u, v, f / 30)
        dxs.append(dx)
        dys.append(dy)
        big = np.abs(dx) > 0.3 * np.abs(dx).max()
        if big.sum() > 10:
            same.append(max((dx[big] > 0).mean(), (dx[big] < 0).mean()))
    dxs, dys = np.array(dxs), np.array(dys)
    assert np.abs(dys).mean() < 0.25 * np.abs(dxs).mean()               # sideways, not swirling
    assert np.mean(same) > 0.8                                           # parts move together
    assert np.abs(dxs).max() <= 1.6 * 8.0


def test_a_distant_tree_line_moves_but_a_hedge_stays_still():
    from tour_worker.motion.canopy import leaf_weight
    rng = np.random.default_rng(3)
    img = np.zeros((1000, 1000, 3), np.uint8)
    img[:] = (230, 180, 120)
    leaves = (np.clip(rng.normal(0, 40, (1000, 1000)), -80, 80)[..., None] + [40, 140, 60]).clip(0, 255)
    mask = np.zeros((1000, 1000), np.float32)
    mask[450:520, 100:900] = 1.0              # a thin tree line at the horizon (far)
    mask[860:950, 100:900] = 1.0              # a hedge in front of the camera
    img[mask > 0] = leaves[mask > 0].astype(np.uint8)
    w, _ = leaf_weight(img, mask)
    assert w[455:490, 200:800].mean() > 0.4   # the far line moves (not held as a hedge)
    assert w[860:950].max() == 0.0            # the hedge stays still


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


def test_thin_distant_water_keeps_most_of_its_surface_moving():
    """Review 2026-10-19: a lake band lost most of its surface to an edge band sized for the near
    water; the held band now narrows with the ripples' own (perspective) reach."""
    from tour_worker.motion.water import water_guard
    img = np.full((800, 1200, 3), (140, 120, 90), np.uint8)
    feather = np.zeros((800, 1200), np.float32)
    feather[400:440, 50:1150] = 1.0                       # a 40 px lake band
    w, _ = water_guard(img, feather, reach=10.0)
    assert (w[400:440] > 0.5).mean() > 0.45


def test_the_gate_catches_a_solid_that_moves_inside_a_tree():
    """2026-10-19: a leaf_weight rule emptied its own solid map, a lantern frame bent and the gate
    passed. The gate now holds tree solids to an independent reference: none may move."""
    from tour_worker.render.frames import ShotWarp
    from tour_worker.render.timeline import compute_timeline
    from tools.quality.perceptual_quality import LIMITS, check_shot
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


def test_tree_tops_sway_most_and_branches_keep_their_own_timing():
    """2026-10-19 "more natural": a tree bends from its base (the top of each column moves most),
    and branch clusters sway at their own phase, so the crown is not one rigid sheet."""
    from tour_worker.motion.canopy import tree_columns
    mask = np.zeros((SH, SW), bool)
    mask[int(0.1 * SH):int(0.6 * SH), int(0.2 * SW):int(0.6 * SW)] = True
    r = Region(_Spec(effect="canopy_flutter", anchor="none", amp=8.0, bbox=[0.2, 0.1, 0.6, 0.6], freq=1.1),
               mask.astype(np.float32), SW, SH)
    r.col_bottom, r.col_span = tree_columns(mask)
    x = np.full(2, 0.4 * SW)
    top_bottom = np.array([0.15 * SH, 0.55 * SH])
    amp = np.zeros(2)
    for f in range(0, 300, 2):
        amp = np.maximum(amp, np.abs(r.displacement(x, top_bottom, f / 30)[0]))
    assert amp[0] > 1.8 * amp[1]                                     # the top sways most
    u = np.linspace(0.25 * SW, 0.55 * SW, 40)
    v = np.full(40, 0.2 * SH)
    series = np.array([r.displacement(u, v, f / 30)[0] for f in range(0, 300, 2)])
    corr = np.corrcoef(series.T)
    assert corr[0, -1] < 0.97                                        # not one rigid sheet
    # neighbours move together: displacement changes by < 0.2 px per px (no tearing / shear)
    assert (np.abs(np.diff(series, axis=1)) / (u[1] - u[0])).max() < 0.2


# ---- review 2026-10-21: firebox rules, fire scaled to its own size, kept off the mantel ---------

def _with_firebox(img, fp):
    from tour_worker.vision.fire_detect import firebox, flame_mask
    return flame_mask(img, fp, firebox(img, fp)) > 0


def test_fires_behind_glass_are_found_with_the_firebox():
    """Downing's gas fires: the glass lifts the blacks around the flames, so the dark-ring test
    failed them; with the firebox known it decides instead."""
    img, fp = _fireplace_scene(lit=True)
    img[180:320, 230:370] = np.clip(img[180:320, 230:370].astype(int) + 30, 0, 255).astype(np.uint8)  # glass haze
    assert _with_firebox(img, fp)[225:290, 240:360].mean() > 0.15


def test_unlit_logs_marble_and_a_throw_in_front_are_not_fire():
    """S3 review: unlit orange logs, an orange surround, a throw on the table in front."""
    img, fp = _fireplace_scene(lit=False)
    for x in range(245, 355, 22):                                    # orange-brown logs, no glow
        cv2.ellipse(img, (x + 8, 290), (12, 8), 0, 0, 360, (40, 110, 175), -1)
    img[90:110, 190:410] = (60, 140, 220)                            # orange marble band on top
    img[322:340, 250:350] = (50, 150, 240)                           # a throw just in front
    fp[322:340, 250:350] = True
    assert _with_firebox(img, fp).sum() == 0


def test_firelight_on_the_lintel_is_not_a_flame():
    img, fp = _fireplace_scene(lit=True)
    img[184:190, 235:365] = (60, 170, 250)                           # a glowing strip under the lintel
    fm = _with_firebox(img, fp)
    assert fm[225:290, 240:360].mean() > 0.15 and fm[182:192].sum() == 0


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


def test_a_potted_plants_pot_stays_still_and_the_plant_sways_from_its_rim():
    """Review 2026-10-21: the plant mask covers the pot, and the pot swayed with the plant."""
    from tour_worker.motion.foliage import plant_weight
    rng = np.random.default_rng(4)
    img = np.full((400, 300, 3), (200, 205, 210), np.uint8)
    leaves = (np.clip(rng.normal(0, 40, (400, 300)), -80, 80)[..., None] + [40, 140, 60]).clip(0, 255)
    mask = np.zeros((400, 300), np.float32)
    mask[40:260, 60:240] = 1.0                                       # foliage
    mask[260:380, 100:200] = 1.0                                     # the pot below it
    img[40:260, 60:240] = leaves[40:260, 60:240].astype(np.uint8)
    img[260:380, 100:200] = (90, 120, 170)                           # a terracotta pot
    img[80:110, 120:150] = (230, 230, 245)                           # a white flower up in it
    w, pot, pot_top = plant_weight(img, mask, 4.0)
    assert w[262:380, 100:200].max() == 0.0                          # the pot is still
    assert pot_top is not None and 255 <= pot_top <= 265             # sway from the rim
    assert w[60:200, 80:220].mean() > 0.8                            # the plant moves ...
    assert w[85:105, 125:145].mean() > 0.8                           # ... flowers too


def test_tree_points_swing_on_a_line_not_in_circles():
    """Review 2026-10-21: "the trees look like they are doing a circular motion". A point's path
    over time must be a line (its 2-D spread nearly one-dimensional), not a loop."""
    from tour_worker.motion.canopy import tree_columns
    mask = np.zeros((SH, SW), bool)
    mask[64:384, 192:576] = True
    r = Region(_Spec(effect="canopy_flutter", anchor="none", amp=8.0, bbox=[0.2, 0.1, 0.6, 0.6], freq=1.1),
               mask.astype(np.float32), SW, SH)
    r.col_bottom, r.col_span = tree_columns(mask)
    r.amp = 8.0
    ab, al = r.canopy_amps()
    rng = np.random.default_rng(0)
    u = rng.uniform(0.25 * SW, 0.55 * SW, 30)
    v = rng.uniform(0.12 * SH, 0.5 * SH, 30)
    path = np.array([np.stack(r.displacement(u, v, f / 30)) for f in range(0, 600, 2)])  # (T, 2, N)
    for k in range(len(u)):
        p = path[:, :, k] - path[:, :, k].mean(axis=0)
        ev = np.linalg.eigvalsh(np.cov(p.T))
        # the minor axis is only the tiny leaf jitter: well under a tenth of the swing
        assert np.sqrt(ev[0]) < 0.1 * np.sqrt(ev[1]) + 0.5 * al, (k, ev)


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
