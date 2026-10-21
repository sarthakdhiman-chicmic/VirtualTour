"""Shared fixtures of the perceptual-motion tests: a synthetic photo with a curtain-like mask, a
one-shot plan around it, and a synthetic fireplace."""
import os

import cv2
import numpy as np

from tour_worker.contract.plan import compute_plan_hash
from tour_worker.motion.masks import mask_relpath
from tour_worker.render.frames import ShotWarp
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


class _Spec:
    def __init__(self, **kw):
        self.__dict__.update(_region(**kw))


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


def _with_firebox(img, fp):
    from tour_worker.vision.fire_detect import firebox, flame_mask
    return flame_mask(img, fp, firebox(img, fp)) > 0
