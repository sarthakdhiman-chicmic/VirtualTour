"""foliage_sway (potted plants): anchored at the bottom; a slow sway grows toward the top, plus fine
leaf flutter. The pot itself is held still (plant_weight).

Every effect function takes the Region (tour_worker.motion.region) as `r`."""
from __future__ import annotations

import math

import cv2
import numpy as np

from tour_worker.motion.masks import crop_box, guard, leafiness
from tour_worker.motion.wind import envelope_at

POT_REACH_BOTTOM = 0.15     # a pot: a non-green part reaching the lowest 15% of a plant's mask,
POT_MAX_SHARE = 0.6         # at most this share of it


def feature_px(r) -> float:
    return min(0.35 * r.bw, 0.35 * r.bh)


def displacement(r, u, v, t):
    """(dx, dy) in source px at source points (u, v), time t seconds."""
    a, s = r.anchor_weight(v)
    A, f = r.amp * envelope_at(r, t), r.freq       # the shot's breeze sets how strongly it moves
    sway = np.sin(2 * math.pi * f * t + r.phase + 0.5 * s)
    n1 = r.noise_a(u / (0.35 * r.bw), v / (0.35 * r.bh), 0.6 * f * t)
    flutter = r.noise_b(u / 18.0, v / 18.0, 2.2 * f * t)
    dx = A * a * (0.7 * sway + 0.3 * n1 + 0.25 * flutter)
    dy = A * a * 0.15 * r.noise_b(u / (0.35 * r.bw) + 50.0, v / (0.35 * r.bh), 0.6 * f * t)
    return dx, dy


def fields(r, u, v, t, irregular=None):
    dx, dy = displacement(r, u, v, t)
    return dx, dy, None


def plant_weight(src: np.ndarray, feather: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Potted plants (review 2026-10-21: "we are moving the pots alongside the plants" - the
    segmentation mask covers plant AND pot). The pot (basket, planter, stand) is a non-green part
    of the mask reaching its bottom; it is held still, with the foliage near it held back by the
    motion's reach so nothing drags it. Flowers are not green either, but sit up in the plant:
    they keep moving. Returns (weight, pot mask, the pot's top row or None) - the plant sways
    from the pot's rim, not from the bottom of the pot."""
    box = crop_box(feather, int(reach) + 20)
    if box is None:
        return feather, np.zeros(feather.shape, bool), None
    y0, y1, x0, x1 = box
    crop, m = src[y0:y1, x0:x1], feather[y0:y1, x0:x1]
    inside = m > 0
    ys = np.nonzero(inside.any(axis=1))[0]
    bottom, top = ys.max(), ys.min()
    nonleaf = (leafiness(crop, soften=False) < 0.15) & inside
    nonleaf = cv2.morphologyEx(nonleaf.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(nonleaf, connectivity=8)
    low = bottom - POT_REACH_BOTTOM * (bottom - top + 1)
    pot = np.zeros(inside.shape, bool)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] - 1 >= low and \
                max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) >= max(12.0, 2.0 * reach):
            pot |= lab == i
    # a pot is a minority of the mask in its lower part; a non-green part that is most of the
    # mask or reaches its top is the plant itself (dried stems, coloured foliage, a dark photo)
    if pot.any():
        rows = np.nonzero(pot.any(axis=1))[0]
        if pot.sum() > POT_MAX_SHARE * inside.sum() or rows.min() < top + 0.3 * (bottom - top + 1):
            pot[:] = False
    if pot.any():                                  # the pot is solid below its rim
        rows = np.nonzero(pot.any(axis=1))[0]
        pot_top = rows.min()
        pot[pot_top:] |= inside[pot_top:] & (cv2.dilate(pot.astype(np.uint8), np.ones((1, 15), np.uint8)) > 0)[pot_top:]
    w = m * guard(pot, 1.5 * reach + 2.0)
    out = np.zeros_like(feather)
    out[y0:y1, x0:x1] = w
    full = np.zeros(feather.shape, bool)
    full[y0:y1, x0:x1] = pot
    return out, full, (float(y0 + np.nonzero(pot.any(axis=1))[0].min()) if pot.any() else None)
