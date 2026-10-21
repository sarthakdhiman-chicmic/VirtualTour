"""canopy_flutter (trees). Review 2026-10-19: the earlier independent x / y branch noise read as "a
circular warp around the trees". The crown sways sideways as one, the sway travelling across it
like wind (neighbouring parts move together, the far side a moment later), gusts swelling and
fading, branch masses lagging a little, leaf shimmer on top; the whole foliage mask moves (trunks
and solids held: leaf_weight). amplitude_px = peak displacement.

Every effect function takes the Region (tour_worker.motion.region) as `r`."""
from __future__ import annotations

import math

import cv2
import numpy as np

from tour_worker.motion.masks import big_parts, crop_box, guard, leafiness

SWAY_HZ = 0.22              # trees: the crown's sway (the plan's freq_hz drives the leaf shimmer)
BRANCH_HZ = 0.55            # each branch cluster's own, quicker sway, at its own phase
LEAF_SHIMMER = 0.03         # leaves turning in the wind: +-3% light at leaf-cluster scale
GUST_PERIOD_S = 6.5         # a gust front crosses each tree this often
SWAY_ACROSS = 0.35          # of a sway cycle across a tree's width: the wind's travel

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


def fields(r, u, v, t, irregular=None):
    """Trees (dx, dy, light). History: 2026-10-16 leaf jitter read as jelly; 2026-10-19 x/y
    branch noise read as a circular warp, then a coherent sideways sway with bend, branch
    phases and leaf shimmer read better but "a simple omni-directional sway". 2026-10-21
    ("varied motions ... like they would in a jungle"), layers:
      - the crown's sway, travelling across the tree; small trees sway quicker than big ones
      - branch clusters each swinging along their OWN line (mostly sideways, some angled up
        to ~55 deg) at their own phase, blending two rates across the crown; every point
        moves back and forth on its line, never in a loop (2026-10-21: loops read as
        "circular motion")
      - gust fronts sweeping across the canopy every few seconds: branches push harder and
        the leaves flip lighter as a front passes
      - rustling patches of leaves, on and off in different places
    Direction, phase and rate fields change slowly across the crown, so neighbouring leaves
    never shear apart (stretch <= ~0.25 px per px, tests)."""
    lb, ll = branch_px(r), leaf_px(r)
    ab, al = canopy_amps(r)
    f = r.freq
    span = max(r.bw, 1.0)
    bd = bend(r, u, v)
    # stiffness: a tree a third of the photo high sways at SWAY_HZ; smaller ones quicker
    hz = SWAY_HZ * min(1.8, max(0.75, (0.33 * r.src_w / max(r.bh, 1.0)) ** 0.4))
    ph = 2 * math.pi * hz * t - 2 * math.pi * SWAY_ACROSS * (u - r.x0) / span + r.phase
    sway = 0.8 * np.sin(ph) + 0.2 * np.sin(1.63 * ph + r.phase2)
    gust = 0.6 + 0.4 * np.clip(0.5 + r.noise_b(u / (0.8 * span) - 0.1 * t, v / (0.8 * span),
                                               0.07 * t), 0, 1)
    # a gust front sweeping across the crown (period GUST_PERIOD_S, off the crown between)
    p = (t / GUST_PERIOD_S + r.phase / (2 * math.pi)) % 1.0
    x_front = r.x0 + (2.2 * p - 0.6) * span
    front = np.exp(-(((u - x_front) / (0.3 * span)) ** 2))
    # branch clusters: own direction (mostly sideways, up to ~55 deg off), own phase, two rates
    cs = 1.5 * lb
    theta = 0.6 * math.pi * r.noise_a(u / cs + 90.0, v / cs, 0.0)
    bph = math.pi * r.noise_a(u / cs + 60.0, v / cs, 0.0)
    mix = np.clip(0.5 + r.noise_b(u / cs + 30.0, v / cs, 0.0), 0.0, 1.0)
    osc = ((1.0 - mix) * np.sin(2 * math.pi * BRANCH_HZ * t + bph)
           + mix * np.sin(2 * math.pi * 1.45 * BRANCH_HZ * t + 1.7 * bph))
    # rustling patches (on / off slowly), leaves flutter there
    rustle = np.clip(1.6 * (r.noise_a(u / (2.0 * lb) + 140.0, v / (2.0 * lb), 0.12 * t) + 0.1), 0.0, 1.0)
    push = gust * (1.0 + 0.5 * front) / 1.25
    # ONE line of motion per point (review 2026-10-21: "a circular motion ... unsettling"): the
    # crown's sway and the cluster's own swing are summed into one signal and applied along
    # the cluster's direction, so every point moves back and forth on a line - never a loop
    # (a sine in x with a cosine in y, or two rates on two axes, traces circles / Lissajous
    # loops). The direction varies slowly across the crown: mostly sideways, some branches
    # angled, which is the variety
    swing = ab * bd * push * (0.6 * sway + 0.4 * osc)
    jitter = al * (0.4 + 0.8 * rustle) * r.noise_b(u / ll, v / ll, f * t)   # leaves, tiny
    dx = swing * np.cos(theta) + jitter
    dy = swing * np.sin(theta) + 0.6 * jitter
    light = 1.0 + LEAF_SHIMMER * (0.5 + 0.7 * rustle + 1.0 * front) * r.noise_b(
        u / ll + 80.0, v / ll, 1.6 * f * t)
    return dx, dy, light


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
