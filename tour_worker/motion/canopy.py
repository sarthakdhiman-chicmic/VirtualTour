"""canopy_flutter (trees): each tree in a region sways sideways on one timing signal driven by the
shot's wind (fields() below for the history of reviews); leaves only move (trunks, solids,
hedges held: leaf_weight); touching crowns are split into separate trees (tree_instances).
amplitude_px = peak displacement.

Every effect function takes the Region (tour_worker.motion.region) as `r`."""
from __future__ import annotations

import math

import cv2
import numpy as np

from tour_worker.motion.masks import big_parts, crop_box, guard, leafiness

SWAY_HZ = 0.22              # trees: the crown's sway (the plan's freq_hz drives the leaf shimmer)
LEAF_SHIMMER = 0.03         # leaves turning in the wind: +-3% light at leaf-cluster scale
LEAN = 0.35                 # a gust leans a tree downwind by this share of its sway

HEDGE_BOTTOM = 0.55         # a short mask part ending below this share of the photo height is a
#                             hedge / bush / garden bed (still); a short part wholly in the upper
#                             part of the photo is a distant tree line (moves). 0.72 caught the
#                             hedges in front of a house (review 2026-10-19)
BASE_HOLD = (0.3, 0.12)     # share of each tree column held still at its base (near, far trees)


def leaf_px(r) -> float:
    return max(8.0, 0.012 * r.src_w)                  # leaf clusters


def branch_px(r) -> float:
    return max(40.0, 0.3 * min(r.bw, r.bh))           # branch masses


def canopy_amps(r) -> tuple[float, float]:
    """(branch sway, leaf flutter) peaks: the leaves never move more than a quarter of a
    cluster (beyond that foliage boils), the branches carry the rest of the strength."""
    # 2026-10-19: a quarter cluster stretched leaves by up to 0.46 px per px (visible pixel
    # shift); the leaves' life now comes mostly from the light shimmer (LEAF_SHIMMER)
    al = min(0.1 * leaf_px(r), 0.12 * r.amp)
    return max(0.0, r.amp - al), al


def feature_px(r) -> float:
    return max(8.0, 0.012 * r.src_w)                  # leaf clusters


def step_error(r, step: int, screen_per_src: float) -> float:
    """Predicted bilinear interpolation error (screen px) of a coarse field grid: branch sway and
    leaf flutter, each with its own detail size."""
    ab, al = canopy_amps(r)
    return screen_per_src * (ab * (step / (branch_px(r) * screen_per_src)) ** 2
                             + al * (step / (leaf_px(r) * screen_per_src)) ** 2)


def bend(r, u, v):
    """Trees: 0.3 at the crown's base .. 1 at its top, per column (col_bottom / col_span set
    by the loader; without them, 1)."""
    cb = getattr(r, "col_bottom", None)
    if cb is None:
        return np.ones_like(u)
    i = np.clip(u.astype(np.int64), 0, cb.shape[0] - 1)
    hh = np.clip((cb[i] - v) / np.maximum(r.col_span[i], 1.0), 0.0, 1.0)
    return 0.3 + 0.7 * hh ** 1.5


def _instance_sway(r, inst, t: float) -> float:
    """One tree's timing signal at t: its own sway (stiffness: small trees quicker) scaled by
    the shot's wind as it reaches the tree, plus a downwind lean while a gust passes. ONE value
    for the whole tree: a crown reversing everywhere at once reads as a tree bending; parts
    reversing at different moments read as rotation (review 2026-10-22)."""
    ph = 2 * math.pi * inst.hz * t + inst.phase
    sway = 0.8 * math.sin(ph) + 0.2 * math.sin(1.63 * ph + inst.phase2)
    w = getattr(r, "wind", None)
    if w is None:
        return sway
    tt = t - inst.delay
    return (w.envelope(tt) * sway + LEAN * w.direction * w.gust(tt)) / (1.0 + LEAN)


def fields(r, u, v, t, irregular=None):
    """Trees (dx, dy, light). History: 2026-10-16 leaf jitter read as jelly; 2026-10-19 x/y
    branch noise read as a circular warp; 2026-10-21 angled branch swings made the crown breathe
    ("pushing from behind, bulging"); 2026-10-22 parts reversing at different moments read as
    rotation. Now (2026-10-22):
      - each TREE (a region may hold several: tree_instances) sways sideways on ONE timing
        signal - its own stiffness and phase, the shot's wind (wind.py) reaching it with a delay
        and leaning it downwind in a gust; the whole crown reverses together
      - the top of each crown moves most (bend), branch clusters further or less (a static,
        smooth amount field) - variety in how FAR, never in when
      - neighbouring trees blend over a soft band (their weights), so leaves never tear
      - tiny leaf jitter and a light shimmer, brighter as a gust passes."""
    lb, ll = branch_px(r), leaf_px(r)
    ab, al = canopy_amps(r)
    f = r.freq
    cs = 3.0 * lb
    reach = 1.0 + 0.35 * r.noise_a(u / cs + 60.0, v / cs, 0.0)
    swing = np.zeros_like(u, dtype=np.float64)
    gusty = 0.0
    for inst in instances_of(r):
        wgt = inst.weight_at(u, v)
        swing += wgt * bend_of(inst, u, v) * _instance_sway(r, inst, t)
        w = getattr(r, "wind", None)
        if w is not None:
            gusty = max(gusty, w.gust(t - inst.delay))
    rustle = np.clip(1.6 * (r.noise_a(u / (2.0 * lb) + 140.0, v / (2.0 * lb), 0.12 * t) + 0.1), 0.0, 1.0)
    jitter = al * (0.4 + 0.8 * rustle) * r.noise_b(u / ll, v / ll, f * t)   # leaves, tiny
    dx = ab * reach * swing + jitter
    dy = 0.4 * jitter
    light = 1.0 + LEAF_SHIMMER * (0.5 + 0.7 * rustle + 1.0 * gusty) * r.noise_b(
        u / ll + 80.0, v / ll, 1.6 * f * t)
    return dx, dy, light


class TreeInstance:
    """One tree of a canopy region: its soft weight (over the region's crop box), timing (phase,
    stiffness hz, wind delay) and its own columns for the bend."""

    def __init__(self, weight, box, phase, phase2, hz, delay, cols):
        self.weight, self.box = weight, box            # box = (y0, x0) of the weight crop
        self.phase, self.phase2, self.hz, self.delay = phase, phase2, hz, delay
        self.col_bottom, self.col_span = cols

    def weight_at(self, u, v):
        if self.weight is None:
            return np.ones_like(u, dtype=np.float64)
        from tour_worker.motion.masks import sample_bilinear
        y0, x0 = self.box
        w = sample_bilinear(self.weight, u - x0, v - y0)
        return w.astype(np.float64)


def instances_of(r) -> list[TreeInstance]:
    """The region's trees (set by the loader); a bare region (tests) is one tree."""
    insts = getattr(r, "instances", None)
    if insts:
        return insts
    cols = (getattr(r, "col_bottom", None), getattr(r, "col_span", None))
    hz = SWAY_HZ * min(1.8, max(0.75, (0.33 * r.src_w / max(r.bh, 1.0)) ** 0.4))
    delay = r.wind.delay(r.x0 + r.bw / 2) if getattr(r, "wind", None) is not None else 0.0
    return [TreeInstance(None, (0, 0), r.phase, r.phase2, hz, delay, cols)]


def bend_of(inst: TreeInstance, u, v):
    """0.3 at a crown's base .. 1 at its top, per column of that tree (1 without columns)."""
    cb = inst.col_bottom
    if cb is None:
        return np.ones_like(u)
    i = np.clip(u.astype(np.int64), 0, cb.shape[0] - 1)
    hh = np.clip((cb[i] - v) / np.maximum(inst.col_span[i], 1.0), 0.0, 1.0)
    return 0.3 + 0.7 * hh ** 1.5


def tree_columns(inside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per photo column: the tree mask's bottom row and height, smoothed across ~2% of the width
    (float32): where each part of a crown bends from."""
    W = inside.shape[1]
    any_col = inside.any(axis=0)
    top = np.where(any_col, inside.argmax(axis=0), 0).astype(np.float32)
    bottom = np.where(any_col, inside.shape[0] - 1 - inside[::-1].argmax(axis=0), 0).astype(np.float32)
    k = max(3, int(0.02 * W) | 1)
    smooth = lambda a: cv2.GaussianBlur(a[None, :], (k, 1), 0)[0]    # noqa: E731
    span = np.where(any_col, bottom - top, 0.0).astype(np.float32)
    return smooth(bottom).astype(np.float32), smooth(span).astype(np.float32)


def leaf_weight(src: np.ndarray, feather: np.ndarray, reach: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Tree regions (canopy_flutter), reviews 2026-10-16 / 2026-10-19. Returns (weight, solid):
    - leaves move (the leaf measure: foliage hue, texture); trunks, branches and sky in the mask
      do not;
    - SOLID objects stay rigid: lanterns, poles, railings, trunks and walls inside or next to the
      mask - non-leaf, non-sky parts at least max(30, 4 x reach) px across (smaller non-leaf
      bits are gaps between leaves). Foliage within `reach` (the motion distance) of a solid
      object is held, so no moving pixel samples it and nothing bends;
    - hedges and bushes stay still (user decision): short mask parts (under 12% of the photo
      high) standing low in the photo (HEDGE_BOTTOM); short parts higher up are distant tree
      lines and move; the base of each tree column is held (BASE_HOLD: trunks, undergrowth)."""
    box = crop_box(feather, int(reach) + 40)
    if box is None:
        return feather, np.zeros(feather.shape, bool)
    y0, y1, x0, x1 = box
    crop, m = src[y0:y1, x0:x1], feather[y0:y1, x0:x1]
    leaf = leafiness(crop)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, sat, val = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    sky = (val > 0.55) & ((((h > 85) & (h < 130)) & (sat > 0.08)) | ((sat < 0.15) & (val > 0.75)))
    cand = cv2.morphologyEx(((leafiness(crop, soften=False) < 0.15) & ~sky).astype(np.uint8),
                            cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    solid = big_parts(cand, max(30.0, 4.0 * reach))
    # (2026-10-19, reverted the same day: treating a large busy non-green part as foliage, and a
    # 0.75 weight floor for the whole mask, bent lantern frames, poles and thin pine trunks -
    # "the trees are so messed up". The leaf measure decides what moves.)
    w = m * leaf * guard(solid, 1.5 * reach + 2.0)
    inside = m > 0
    H = src.shape[0]
    hedge_h = 0.12 * H
    n, lab, st, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=8)
    bottom_px = y0 + st[:, cv2.CC_STAT_TOP] + st[:, cv2.CC_STAT_HEIGHT]
    far_part = (bottom_px < HEDGE_BOTTOM * H)
    hedge_part = (st[:, cv2.CC_STAT_HEIGHT] < hedge_h) & ~far_part
    far_part[0] = hedge_part[0] = False
    w[hedge_part[lab]] = 0.0                                           # bushes, hedges: still
    far = far_part[lab] & inside
    # each column of the tree: still at its base (hedge rows, trunks, undergrowth), moving in the
    # crown; a column only a hedge tall stays still. The column profiles are smoothed across
    # ~2% of the width so neighbouring columns never differ sharply (no shearing streaks).
    rows = np.arange(inside.shape[0])[:, None]
    any_col = inside.any(axis=0)
    top = np.where(any_col, inside.argmax(axis=0), inside.shape[0]).astype(np.float32)
    bottom = np.where(any_col, inside.shape[0] - 1 - inside[::-1].argmax(axis=0), 0).astype(np.float32)
    k = max(3, int(0.02 * src.shape[1]) | 1)
    smooth = lambda a: cv2.GaussianBlur(a[None, :], (k, 1), 0)[0]    # noqa: E731
    span = np.where(any_col, bottom - top, 0.0).astype(np.float32)
    bottom_s, span_s = smooth(bottom), smooth(span)
    x = np.clip((bottom_s[None, :] - rows) / (BASE_HOLD[0] * np.maximum(span_s[None, :], 1.0)), 0, 1)
    xf = np.clip((bottom_s[None, :] - rows) / (BASE_HOLD[1] * np.maximum(span_s[None, :], 1.0)), 0, 1)
    tall = np.clip((span_s - 0.8 * hedge_h) / (0.5 * hedge_h), 0, 1)
    near_w = x * x * (3 - 2 * x) * (tall * tall * (3 - 2 * tall))[None, :]
    far_w = xf * xf * (3 - 2 * xf)                     # a distant tree line: only its foot held
    w *= np.where(far, far_w, near_w).astype(np.float32)
    out = np.zeros_like(feather)
    out[y0:y1, x0:x1] = w
    full_solid = np.zeros(feather.shape, bool)
    full_solid[y0:y1, x0:x1] = solid
    return out, full_solid


# ---- separating touching trees (2026-10-22) -----------------------------------------------------
# COCO's "tree" is a stuff class: one mask for every tree in the photo. Parts that do not touch are
# already separate regions; touching crowns (a tree line) are split here, with no extra model:
#   1. the skyline: the top edge of the canopy has a peak per crown and a dip between crowns; a dip
#      deep enough on both sides is a split
#   2. a watershed on the photo from a seed under each peak places the boundary where one crown's
#      colour / texture meets the next (a light maple beside a dark pine)
#   3. each tree's weight is softened over BLEND, so neighbouring trees' motions mix over a band
#      and the leaves between them never tear
SPLIT_DEPTH = (0.08, 12.0)   # a dip must be this share of the canopy's height (and px) below both peaks
PEAK_SEP = 0.08              # peaks closer than this share of the canopy's width are one crown
MIN_TREE_SHARE = 0.06        # a split-off part smaller than this share of the mask stays merged
MAX_TREES = 6
BLEND = 0.02                 # soft band between trees, share of the photo width (min 4 px)


def _skyline_splits(inside: np.ndarray) -> list[int]:
    """Columns (in the crop) where a canopy part's skyline dips between two crowns."""
    cols = np.nonzero(inside.any(axis=0))[0]
    if len(cols) < 30:
        return []
    c0, c1 = cols.min(), cols.max() + 1
    sub = inside[:, c0:c1]
    has = sub.any(axis=0)
    top = np.where(has, sub.argmax(axis=0), sub.shape[0]).astype(np.float32)
    bottom = float(np.nonzero(sub.any(axis=1))[0].max())
    height = np.where(has, bottom - top, 0.0).astype(np.float32)
    k = max(5, int(0.02 * (c1 - c0)) | 1)
    h = cv2.GaussianBlur(height[None, :], (k, 1), 0)[0]
    span = c1 - c0
    win = max(3, int(PEAK_SEP * span))
    peaks = [x for x in range(span)
             if h[x] > 0 and h[x] == h[max(0, x - win):x + win + 1].max()]
    merged = []
    for x in peaks:                                   # plateaus: keep one peak per crown
        if merged and x - merged[-1] <= win:
            if h[x] > h[merged[-1]]:
                merged[-1] = x
            continue
        merged.append(x)
    canopy_h = float(h.max())
    need = max(SPLIT_DEPTH[0] * canopy_h, SPLIT_DEPTH[1])
    splits = []
    for a, b in zip(merged, merged[1:]):
        x = a + int(np.argmin(h[a:b + 1]))
        if min(h[a], h[b]) - h[x] >= need:
            splits.append((min(h[a], h[b]) - h[x], c0 + x))
    splits.sort(reverse=True)                          # the deepest dips first
    return sorted(x for _, x in splits[:MAX_TREES - 1])


def tree_instances(src: np.ndarray, feather: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Separate trees of a canopy mask: [(soft weight over a crop, (y0, x0) of the crop)], weights
    summing to 1 over the mask. One entry (weight None) when nothing should be split."""
    inside_full = feather > 0
    box = crop_box(feather, 4)
    if box is None:
        return [(None, (0, 0))]
    y0, y1, x0, x1 = box
    inside = inside_full[y0:y1, x0:x1]
    total = int(inside.sum())
    n, lab = cv2.connectedComponents(inside.astype(np.uint8), connectivity=8)
    markers = np.zeros(inside.shape, np.int32)
    label = 0
    for i in range(1, n):
        part = lab == i
        if part.sum() < MIN_TREE_SHARE * total:
            continue
        splits = _skyline_splits(part)
        cols = np.nonzero(part.any(axis=0))[0]
        edges = [cols.min()] + splits + [cols.max() + 1]
        for a, b in zip(edges, edges[1:]):
            seg = part[:, a:b]
            if seg.sum() < MIN_TREE_SHARE * total:
                continue
            label += 1
            # the seed: the upper middle of this crown (away from the split columns)
            ys = np.nonzero(seg.any(axis=1))[0]
            top, bot = ys.min(), ys.max()
            m0, m1 = a + int(0.3 * (b - a)), a + max(int(0.7 * (b - a)), int(0.3 * (b - a)) + 1)
            seed = np.zeros_like(part)
            seed[top:top + max(2, int(0.5 * (bot - top + 1))), m0:m1] = True
            markers[seed & part] = label
    if label <= 1:
        return [(None, (0, 0))]
    crop = np.ascontiguousarray(src[y0:y1, x0:x1])
    ws = cv2.watershed(crop, markers.copy())          # the trees are the only basins
    ws[ws < 1] = 0                                    # the watershed's own boundary lines (-1):
    for _ in range(8):                                # give them to the neighbouring tree
        if not (ws == 0).any():
            break
        ws = np.where(ws == 0, cv2.dilate(ws.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32), ws)
    sigma = max(4.0, BLEND * src.shape[1])
    weights = []
    for k in range(1, label + 1):
        weights.append(cv2.GaussianBlur((ws == k).astype(np.float32), (0, 0), sigma))
    tot = np.maximum(np.sum(weights, axis=0), 1e-6)
    return [((w / tot).astype(np.float32), (y0, x0)) for w in weights]


def build_instances(r, src: np.ndarray, feather_raw: np.ndarray, wind) -> list[TreeInstance]:
    """The region's trees with their own timing: phase (from the region's seed), stiffness (each
    tree's own height: small trees sway quicker), wind delay (each tree's centre), columns."""
    parts = tree_instances(src, feather_raw)
    rng = np.random.default_rng((int(r.spec.seed) * 2654435761) & 0xFFFFFFFF)
    # the bend (top moves most) uses the whole canopy's columns: a tree's own columns end at its
    # edge, where the blend band still mixes its motion in - the bend would jump there
    cols = tree_columns(feather_raw > 0)
    out = []
    H, W = feather_raw.shape
    for k, (wgt, (y0, x0)) in enumerate(parts):
        if wgt is None:
            hard = feather_raw > 0
            phase, phase2 = r.phase, r.phase2
        else:
            hard = np.zeros((H, W), bool)
            hard[y0:y0 + wgt.shape[0], x0:x0 + wgt.shape[1]] = (wgt > 0.5) & (feather_raw[
                y0:y0 + wgt.shape[0], x0:x0 + wgt.shape[1]] > 0)
            phase, phase2 = (r.phase, r.phase2) if k == 0 else tuple(rng.uniform(0, 2 * math.pi, 2))
        ys, xs = np.nonzero(hard)
        if not len(ys):
            continue
        tree_h = float(ys.max() - ys.min() + 1)
        hz = SWAY_HZ * min(1.8, max(0.75, (0.33 * r.src_w / max(tree_h, 1.0)) ** 0.4))
        delay = wind.delay(float(xs.mean())) if wind is not None else 0.0
        out.append(TreeInstance(wgt, (y0, x0), phase, phase2, hz, delay, cols))
    return out
