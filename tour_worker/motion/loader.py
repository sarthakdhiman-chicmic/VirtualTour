"""Motion regions of a shot, ready to draw: each region's fetched mask, checked and feathered,
narrowed by its effect's "what stays still" rule (trees: leaves only, solids and hedges held;
plants: the pot held; water: its edge and solid things reaching it held; fire: only its own box,
structure inside it held)."""
from __future__ import annotations

import os

import cv2
import numpy as np

from tour_worker.motion.canopy import build_instances, leaf_weight, tree_columns
from tour_worker.motion.fire import (FIRE_AMP_OF_HEIGHT, FIRE_PLUME, fire_box, fire_glow, fire_structure,
                                     fire_zone_file, flame_plume)
from tour_worker.motion.foliage import plant_weight
from tour_worker.motion.masks import guard, load_feather, mask_relpath
from tour_worker.motion.region import Region
from tour_worker.motion.water import water_guard
from tour_worker.motion.wind import shot_wind


def load_regions(shot, images_dir: str, src_w: int, src_h: int,
                 src: np.ndarray | None = None) -> tuple[list[Region], list[str]]:
    """Regions whose mask was fetched (images/masks/<shot>_<id>.png) and fits the photo.
    src (the photo, BGR): tree regions are narrowed to their leaves with solid objects and hedges
    held still (leaf_weight); water regions hold solid dark objects still (water_guard)."""
    out, warnings = [], []
    wind = shot_wind(shot, src_w)                   # one breeze for every region of the shot
    for spec in shot.motion_regions or []:
        path = os.path.join(images_dir, mask_relpath(shot.shot_id, spec.id))
        x1, y1, x2, y2 = spec.bbox
        if spec.effect == "fire_flicker" and os.path.exists(path):
            path, (x1, y1, x2, y2) = fire_zone_file(path, src_w, src_h)
        feather, reason = (load_feather(path, src_w, src_h, (x1 * src_w, y1 * src_h, x2 * src_w, y2 * src_h))
                           if os.path.exists(path) else (None, "missing"))
        if feather is None:
            warnings.append(f"motion_region_dropped:{shot.shot_id}:{spec.id}:{reason}")
            continue
        solid = cols = pot_top = raw = None
        if src is not None and spec.effect == "canopy_flutter":
            raw = feather.copy()
            cols = tree_columns(feather > 0)
            feather, solid = leaf_weight(src, feather, float(spec.amplitude_px))
        elif src is not None and spec.effect == "foliage_sway":
            feather, solid, pot_top = plant_weight(src, feather, float(spec.amplitude_px))
        elif src is not None and spec.effect == "calm_water":
            feather, solid = water_guard(src, feather, float(spec.amplitude_px))
        flame = plume = None
        if spec.effect == "fire_flicker":
            ys = np.nonzero((feather > 0).any(axis=1))[0]
            fh = float(ys.max() - ys.min() + 1)
            plume = flame_plume(feather, FIRE_PLUME * fh)
            if src is not None:
                # inside the fire's own box only, held back from its edge and from the structure
                # inside it by the motion's reach, so no moving pixel samples the mantel, the
                # firebox walls, the glass frame or the glass's reflections
                box = fire_box(src, feather)
                amp = min(float(spec.amplitude_px), FIRE_AMP_OF_HEIGHT * fh)
                struct = fire_structure(src, feather, box)
                # the tongues lick UP (sampling from below) and only ~0.55 x amp sideways:
                # what lies above or beside them needs a margin of the sideways reach only
                reach = 0.7 * amp + 2.0
                plume *= guard((box < 0.5) | struct, reach)
                solid = ((box < 0.5) | struct) & (cv2.dilate((plume > 0).astype(np.uint8),
                                                             np.ones((9, 9), np.uint8)) > 0)
            flame, feather = feather, np.maximum(fire_glow(feather), plume)
        region = Region(spec, feather, src_w, src_h)
        region.solid = solid                       # for the quality gate's drag check
        if spec.effect not in ("fire_flicker", "calm_water", "water_ripple"):   # fire, water: their own
            region.wind = wind
            region.wind_delay = wind.delay(region.x0 + region.bw / 2)
        if raw is not None:                        # touching crowns: separate trees
            region.instances = build_instances(region, src, raw, wind)
        if cols is not None:
            region.col_bottom, region.col_span = cols
        if pot_top is not None and pot_top > region.y0 + 0.2 * region.bh:
            region.bh = pot_top - region.y0                # sway from the pot's rim
        if flame is not None:
            ys, xs = np.nonzero(flame > 0)
            region.flame, region.flame_h = flame, float(ys.max() - ys.min() + 1)
            region.plume = plume
            region.amp = min(region.amp, FIRE_AMP_OF_HEIGHT * region.flame_h)
            region.flame_bottom = float(ys.max())
            gy, gx = np.nonzero(feather > 1e-3)     # the region spans its glow, not just the flames
            region.x0, region.y0 = float(gx.min()), float(gy.min())
            region.bw, region.bh = float(gx.max() - gx.min() + 1), float(gy.max() - gy.min() + 1)
        out.append(region)
    return out, warnings
