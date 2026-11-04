"""tour_worker/render/frames.py: a rotate shot's frames are the exact homography warp of the photo."""
import os
import tempfile

import cv2
import numpy as np
import pytest

from tour_worker.render.camera_path import rotate_homography
from tour_worker.contract.plan import compound_path, compute_plan_hash, parse_plan
from tour_worker.render.frames import ShotWarp
from tour_worker.render.timeline import compute_timeline


def _plan(projection):
    cam = {"type": "compound", "move": "x", "start": {"cx": 0.47, "cy": 0.55, "zoom": 1.2},
           "segments": [{"t0": 0, "t1": 1, "cx": 0.5, "cy": 0.5, "zoom": 1.012}],
           "timing": {"s0": 1, "s1": 1, "turn": 60}, "projection": projection}
    plan = {"plan_version": "1.3", "listing_id": "t", "mode": "cinematic", "preset": "p",
            "output": {"width": 1920, "height": 1080, "fps": 30, "codec": "h264"}, "total_duration_s": 1.0,
            "shots": [{"shot_id": "s01", "order": 1, "asset_id": "a", "image_uri": "x.jpg",
                       "source": {"width": 1920, "height": 1280}, "room_key": "r", "room_type": "r",
                       "narrative_stage": "living", "role": "hero", "duration_s": 1.0, "camera": cam,
                       "transition_in": {"type": "cut", "duration_s": 0.0}}],
            "warnings": [], "provenance": {"scoring_config_version": "0", "planner_config_version": "0"}}
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    return parse_plan(plan)


def test_rotate_shot_frames_are_the_exact_homography_warp():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((1280, 1920, 3)) * 255).astype(np.uint8), (0, 0), 6)
    with tempfile.TemporaryDirectory() as wd:
        os.makedirs(os.path.join(wd, "images"))
        cv2.imwrite(os.path.join(wd, "images", "x.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        plan = _plan("rotate")
        n = compute_timeline(plan)[0]["render_frames"]
        assert n == 30
        sw = ShotWarp(plan, plan.shots[0], n, os.path.join(wd, "images"))
        src = cv2.imread(os.path.join(wd, "images", "x.jpg"))
        cx, cy, z = compound_path(plan.shots[0], 1920, 1080).frames(30)[17]
        ref = cv2.warpPerspective(src, np.array(rotate_homography(cx, cy, z, 1920, 1280, 1920, 1080)), (1920, 1080),
                                  flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        got = sw.frame(17)
        assert np.abs(got.astype(int) - ref.astype(int)).mean() < 3.0
        first = sw.frame(0).tobytes()
        again = ShotWarp(plan, plan.shots[0], n, os.path.join(wd, "images"))
        assert again.frame(0).tobytes() == first                     # deterministic


@pytest.mark.parametrize("projection", [None, "rotate"])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_a_photo_of_another_size_frames_like_the_plan_size(projection, scale):
    """A photo fetched at another size than the plan's source (same aspect) is scaled to it and
    framed the same: the camera works on the scaled photo, not the file's size."""
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur((rng.random((1280, 1920, 3)) * 255).astype(np.uint8), (0, 0), 12)
    img[:640, :960] //= 3                                       # a dark quadrant: framing shows
    plan = _plan(projection)
    n = compute_timeline(plan)[0]["render_frames"]
    with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as other_dir:
        cv2.imwrite(os.path.join(ref_dir, "x.png"), img)
        other = cv2.resize(img, (int(1920 * scale), int(1280 * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(other_dir, "x.png"), other)
        plan.shots[0].image_uri = "x.png"
        warnings = []
        got = ShotWarp(plan, plan.shots[0], n, other_dir, warnings)
        ref = ShotWarp(plan, plan.shots[0], n, ref_dir)
        assert any(w.startswith("image_resized:s01:") for w in warnings)
        for f in (0, n - 1):
            assert np.abs(got.frame(f).astype(int) - ref.frame(f).astype(int)).mean() < 3.0


def test_two_photos_with_the_same_basename_stay_apart():
    from tour_worker.io.images import photo_filename
    plan = _plan(None)
    a = plan.shots[0].model_copy(update={"image_uri": "s3://b/listing1/IMG_0001.jpg"})
    b = plan.shots[0].model_copy(update={"image_uri": "s3://b/listing2/IMG_0001.jpg"})
    c = plan.shots[0].model_copy(update={"image_uri": "s3://b/listing1/IMG_0002.jpg"})
    plan.shots[:] = [a, b, c, a]
    names = [photo_filename(plan, s.image_uri) for s in plan.shots]
    assert names[0] != names[1] and names[0].endswith("_IMG_0001.jpg") and names[0] == names[3]
    assert names[2] == "IMG_0002.jpg"                          # a unique basename is kept as is


def test_a_punch_peaks_at_its_time_after_the_cut_on_a_faded_shot(tmp_path):
    """at_s counts from the cut (the shot's nominal start); a faded shot starts drawing `head`
    frames before it, so frame f is (f - head) / fps s into the shot."""
    from tour_worker.render.frames import punch_scale
    img = cv2.GaussianBlur((np.random.default_rng(2).random((1280, 1920, 3)) * 255).astype(np.uint8), (0, 0), 6)
    cv2.imwrite(str(tmp_path / "x.jpg"), img)
    kb = {"type": "ken_burns", "easing": "linear", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.1},
          "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.1}}
    shot = {"order": 1, "asset_id": "a", "image_uri": "x.jpg", "source": {"width": 1920, "height": 1280},
            "room_key": "r", "room_type": "r", "narrative_stage": "living", "role": "hero", "duration_s": 2.0}
    punched = dict(kb, punches=[{"at_s": 0.5, "amount": 0.05, "attack_s": 0.0, "decay_s": 0.3}])
    plan = {"plan_version": "1.5", "listing_id": "t", "mode": "cinematic", "preset": "p",
            "output": {"width": 1920, "height": 1080, "fps": 30, "codec": "h264"}, "total_duration_s": 4.0,
            "shots": [dict(shot, shot_id="s01", camera=kb, transition_in={"type": "cut", "duration_s": 0.0}),
                      dict(shot, shot_id="s02", order=2, camera=punched,
                           transition_in={"type": "fade", "duration_s": 0.6})],
            "warnings": [], "provenance": {"scoring_config_version": "0", "planner_config_version": "0"}}
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    plan = parse_plan(plan)
    t = compute_timeline(plan)[1]
    assert t["head"] == 9
    sw = ShotWarp(plan, plan.shots[1], t["render_frames"], str(tmp_path))
    zooms = [z for (_, _, z), _ in sw.frames]
    peak = int(np.argmax(zooms))
    assert peak == t["head"] + 15                               # 0.5 s after the cut, at 30 fps
    assert zooms[peak] == pytest.approx(1.1 * punch_scale(plan.shots[1].camera.punches, 0.5))
