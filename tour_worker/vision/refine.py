"""Refinement of the segmentation model's raw water masks (2026-11-11, from the M2F benchmark's
refine step; kept for OneFormer COCO, 2026-11-20):

  water      raw pool / water masks take in raised spa walls, tile rims and some coping.
             refine_water_surface keeps the model's mask (reflections, submerged shelves, steps
             and dark water keep moving) and takes out only what stands above the waterline:
             dark walls with brighter stone directly above them, thin rims and coping along the edge.
             (2026-11-11 v2: the first, chroma-based version froze reflections and shelves and let
             a dark blue spa wall ripple.)

Masks are uint8, 0 / 255, the photo's size. Pure OpenCV / numpy: deterministic."""
from __future__ import annotations

import cv2
import numpy as np


def _ring_outside_fraction(part: np.ndarray, outside: np.ndarray) -> float:
    """The share of a part's 4-px ring that lies outside the mask."""
    ring = cv2.dilate(part, np.ones((9, 9), np.uint8)) - part
    return float((ring & outside).sum()) / max(1, int(ring.sum()))


def _components(m: np.ndarray, min_area: float):
    """The connected parts of m (uint8 0/1) of at least min_area px, as uint8 0/1 masks."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            yield (lab == i).astype(np.uint8)


def _strip_above(part: np.ndarray, depth: int) -> np.ndarray:
    """The pixels up to `depth` px directly above a part (image up), the part itself excluded."""
    k = np.zeros((2 * depth + 1, 1), np.uint8)
    k[depth:] = 1                     # dilating with the lower half of a column moves pixels up
    return (cv2.dilate(part, k) > 0) & (part == 0)


def _stands_above_water(part: np.ndarray, lab: np.ndarray, stone: np.ndarray, depth: int) -> bool:
    """Wall test for a dark part: directly above it is stone (outside the mask, or grey / pale
    coping), neutral in colour and clearly brighter than the part. A reflection or dark water has
    more water above it, or mirrors what stands above it (dark trees over a dark tree reflection)."""
    up = _strip_above(part, depth)
    if up.sum() == 0 or stone[up].mean() < 0.6:
        return False
    a, p = np.median(lab[up], 0), np.median(lab[part > 0], 0)
    chroma_up = float(np.hypot(a[1] - 128.0, a[2] - 128.0))
    return chroma_up <= 18.0 and float(a[0] - p[0]) >= 25.0


def refine_water_surface(bgr: np.ndarray, raw_pool_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(water_surface, structure) of a raw pool / water mask.

    The model's mask is the base, so reflections, submerged shelves and steps, dark water and
    sunlit water all keep moving. Only what stands above the waterline is taken out:
      * walls and tile faces: solidly dark against this pool's own water, with neutral stone
        (coping, deck) directly above them that is clearly brighter. A dark part bigger than a third
        of the water is dark water, never a wall;
      * thin dark rims along the mask's edge, same test;
      * grey or pale coping / paving along the edge that is not water-hued;
      * a thin lip along the edge that isn't water-coloured.
    Thresholds are relative to the pool (its brighter half), so a dark-finish pool, a night pool
    or a bright one is judged against itself. Checked on 104 varied photos (pools, spas, indoor,
    night, lakes, ponds, sea; 2026-11-11)."""
    empty = np.zeros_like(raw_pool_mask)
    if not raw_pool_mask.any():
        return empty, empty.copy()

    H, W = raw_pool_mask.shape
    inside = raw_pool_mask > 0
    ins8 = inside.astype(np.uint8)
    outside = 1 - ins8
    region_area = int(inside.sum())
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # this pool's water: its brighter half (shadows, walls and grout pull a plain median down)
    vv, sv, hv = v[inside], s[inside], h[inside]
    bright = vv >= np.median(vv)
    v_ref = float(np.median(vv[bright]))
    s_ref = max(1.0, float(np.median(sv[bright])))
    h_ref = float(np.median(hv[bright]))
    dh = np.abs(h - h_ref)
    dh = np.minimum(dh, 180.0 - dh)
    water_hued = (dh <= 25) & (s >= 20)
    k = max(3, int(round(0.004 * W)) | 1)            # gap closing, ~6 px on a 1536 px photo
    w = max(5, int(0.02 * W) | 1)                    # the density window, ~31 px
    depth = max(4, int(0.012 * H))                   # the strip above a wall, ~14 px
    min_area = 0.0005 * H * W
    max_wall = 0.35 * region_area

    def clean(m: np.ndarray) -> np.ndarray:
        m = cv2.morphologyEx((m & inside).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) & ins8

    dark_px = ((v < 0.62 * v_ref) & (s < 0.85 * s_ref) & inside).astype(np.uint8)
    pale_px = ((v >= 0.62 * v_ref) & (s < 0.45 * s_ref) & ~water_hued & inside).astype(np.uint8)
    dark = clean(dark_px > 0)
    structure = np.zeros(raw_pool_mask.shape, bool)

    # coping / paving inside the mask: grey or pale, not water-hued, hugging the edge
    for part in _components(clean(pale_px > 0), min_area):
        if _ring_outside_fraction(part, outside) >= 0.15:
            structure |= part > 0
    stone = (~inside) | (pale_px > 0) | structure

    # walls: dense dark cores (>= 75% dark in the window) out to their dark edge
    dens = cv2.boxFilter(dark_px.astype(np.float32), -1, (w, w))
    for part in _components(((dens >= 0.75) & inside).astype(np.uint8), min_area):
        if part.sum() > max_wall:
            continue
        grown = (cv2.dilate(part, np.ones((w, w), np.uint8)) & dark) | part
        if _stands_above_water(grown, lab, stone, depth):
            structure |= grown > 0
    # thin dark rims along the edge, too thin for the density window
    for part in _components(dark, min_area):
        if part.sum() > max_wall or _ring_outside_fraction(part, outside) < 0.2:
            continue
        if _stands_above_water(part, lab, stone, depth):
            structure |= part > 0

    # the lip: a thin band along the edge that isn't water-coloured (stone, grout, shadow line)
    waterish = water_hued | ((h >= 68) & (h <= 145) & (s >= 18))
    lip = (ins8 - cv2.erode(ins8, np.ones((k, k), np.uint8))) > 0
    structure |= lip & ~waterish
    structure = (cv2.dilate(structure.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & inside

    surface = np.where(structure, 0, raw_pool_mask).astype(raw_pool_mask.dtype)
    n, lab_s, st, _ = cv2.connectedComponentsWithStats((surface > 0).astype(np.uint8))
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:       # specks left between structure parts
            surface[lab_s == i] = 0
    rest = raw_pool_mask.copy()
    rest[surface > 0] = 0
    return surface, rest

