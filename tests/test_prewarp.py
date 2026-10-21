"""modules/prewarp.py: a rotate shot's frames are the exact homography warp of the photo."""
import os
import tempfile

import cv2
import numpy as np

from modules.camera_path import rotate_homography
from modules.plan_io import compound_path, compute_plan_hash, parse_plan
from modules.prewarp import ShotWarp
from modules.timeline import compute_timeline


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
