"""Perceptual motion regions (plan 1.4): which objects in a shot move, and how much.

Effects: curtains / blinds sway like fabric hanging from a rod (anchored at the top), potted
plants sway from the pot (anchored at the bottom), trees flutter across the canopy
(canopy_flutter) and water reflections sway on a moving wave surface (calm_water). The worker
renders the displacement inside each mask before the camera warp (modules/perceptual).

A region is kept when the detect class maps to an effect, the detection is confident, the object
is big enough to read, and at least half of it is on screen at the start or the end of the
shot's camera move. At most `perceptual_max_regions` per shot, largest x most confident first.
"""

from __future__ import annotations

import hashlib
import math

from tour_worker.render.camera_path import base_window
from .config import PlannerConfig
from .schemas import CameraParams, MotionRegion


def _window(
    cx: float, cy: float, zoom: float, bw: float, bh: float
) -> tuple[float, float, float, float]:
    w, h = bw / zoom, bh / zoom
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def camera_windows(cam: CameraParams, src_w: int, src_h: int, out_w: int, out_h: int):
    """The camera's view at the first and the last frame, normalised to the photo."""
    bw, bh = base_window(src_w, src_h, out_w, out_h)
    if cam.type == "keyframe_path" and cam.keyframes:
        ends = [cam.keyframes[0], cam.keyframes[-1]]
    elif cam.type == "compound" and cam.start and cam.segments:
        ends = [cam.start, cam.segments[-1]]
    elif cam.start and cam.end:
        ends = [cam.start, cam.end]
    else:
        return [(0.0, 0.0, 1.0, 1.0)]
    return [_window(p.cx, p.cy, p.zoom, bw, bh) for p in ends]


def _overlap(a, b) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def _seed(listing_id: str, shot_id: str, object_id: str) -> int:
    return int(hashlib.sha256(f"{listing_id}|{shot_id}|{object_id}".encode()).hexdigest()[:8], 16)


def plan_motion_regions(
    listing_id: str,
    shot_id: str,
    cam: CameraParams,
    detections: list[dict],
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    cfg: PlannerConfig,
    room_type: str | None = None,
) -> list[MotionRegion]:
    windows = camera_windows(cam, src_w, src_h, out_w, out_h)
    # detect resizes photos whose long side exceeds perceptual_detect_max_side (aspect kept);
    # its bboxes and masks are in that resized space (measured 2026-10-14: 6240x4160 -> 1600x1067)
    k = min(1.0, cfg.perceptual_detect_max_side / max(src_w, src_h))
    dw, dh = src_w * k, src_h * k
    cands = []
    for d in detections or []:
        # OneFormer regions carry their pixel space; detect results use detect's resize rule
        if d.get("space"):
            dw, dh = float(d["space"][0]), float(d["space"][1])
            k = dw / src_w
        else:
            k = min(1.0, cfg.perceptual_detect_max_side / max(src_w, src_h))
            dw, dh = src_w * k, src_h * k
        effect = cfg.perceptual_effects.get(str(d.get("class_name", "")).lower())
        bbox, mask = d.get("bbox"), d.get("mask_url")
        if not effect or not mask or not bbox or len(bbox) != 4:
            continue
        outdoors = (room_type or "") in cfg.perceptual_outdoor_rooms
        if outdoors and effect in cfg.perceptual_still_outdoors:
            continue                                  # shrubs and hedges outdoors stay still
        if float(d.get("confidence") or 0) < cfg.perceptual_min_confidence:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        # a bbox outside detect's pixel space means the detection belongs to a different photo
        if x2 > dw * 1.02 or y2 > dh * 1.02 or x2 <= x1 or y2 <= y1:
            continue
        nb = (x1 / dw, y1 / dh, x2 / dw, y2 / dh)
        area = (nb[2] - nb[0]) * (nb[3] - nb[1])
        if area < cfg.perceptual_min_area_by_effect.get(effect, cfg.perceptual_min_area):
            continue
        visible = max(_overlap(nb, w) for w in windows) / area
        if visible < cfg.perceptual_min_visible_by_effect.get(effect, cfg.perceptual_min_visible):
            continue
        bbox_w_src = (x2 - x1) / k
        # distance from the box height: thin boxes are far (a tree line, a lake band)
        far = 1.0
        if effect in cfg.perceptual_far_effects:
            h = nb[3] - nb[1]
            far = 1.0 + cfg.perceptual_far_boost * min(
                1.0, max(0.0, (cfg.perceptual_far_height - h) / (0.7 * cfg.perceptual_far_height)))
        scale = out_w / max((w[2] - w[0]) * src_w for w in windows)
        target = cfg.perceptual_screen_target_px.get(effect)
        if target is not None:
            # on-screen strength: output px per source px at the widest view of the shot
            amp = min(far * target / scale, far * cfg.perceptual_bbox_cap[effect] * bbox_w_src,
                      cfg.perceptual_max_amplitude_px)
            if amp * scale < cfg.perceptual_min_screen_px[effect]:
                continue                      # too small on screen to read as motion
        else:
            frac, lo, hi = cfg.perceptual_amplitude[effect]
            size = (src_w if effect in cfg.perceptual_amplitude_of_photo
                    else (y2 - y1) / k if effect in cfg.perceptual_amplitude_of_height
                    else bbox_w_src)
            amp = min(max(frac * size, lo), hi)
            if effect == "calm_water":
                # distance: a thin band of water (a lake near the horizon) moves less than water
                # filling the lower frame (review 2026-10-16: the distant lake moved as much as the
                # pool and dragged its docks), but never less than reads on screen, boosted like
                # far trees (review 2026-10-19: distant water was left still - "the stuff didn't
                # get applied to all the objects")
                amp *= min(1.0, far * (nb[3] - nb[1]) / cfg.perceptual_water_full_height)
                floor = cfg.perceptual_min_screen_px.get(effect, 0.0) / scale
                amp = min(max(amp, floor), hi, cfg.perceptual_max_amplitude_px)
        cands.append((area * float(d.get("confidence") or 0), d, effect, nb, amp))
    cands.sort(key=lambda c: (-c[0], str(c[1].get("object_id"))))
    out = []
    base = _seed(listing_id, shot_id, "sway") / 0xFFFFFFFF * 2 * math.pi
    count = {"fabric_sway": 0, "foliage_sway": 0}
    for _, d, effect, nb, amp in cands[: cfg.perceptual_max_regions]:
        oid = str(d.get("object_id") or len(out))
        phase = None
        if effect in count:              # curtains one way, plants the other
            phase = (base + (math.pi if effect == "foliage_sway" else 0.0)
                     + cfg.perceptual_phase_step * count[effect]) % (2 * math.pi)
            count[effect] += 1
        out.append(
            MotionRegion(
                id=oid,
                effect=effect,
                mask_uri=d["mask_url"],
                bbox=[round(v, 5) for v in nb],
                amplitude_px=round(amp, 3),
                freq_hz=cfg.perceptual_freq_hz[effect],
                anchor=cfg.perceptual_anchor[effect],
                seed=_seed(listing_id, shot_id, oid),
                label=str(d.get("class_name")),
                phase=None if phase is None else round(phase, 5),
            )
        )
    return out
