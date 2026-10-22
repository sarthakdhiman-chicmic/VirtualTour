"""Water effects.

  calm_water    a moving wave height field on the water plane, in perspective (thin dense bands
                far away, wider near the camera); the reflection is displaced by the surface slope,
                mostly sideways, so reflected edges sway in horizontal bands; brightness follows
                the slope smoothly (+-4%, no thresholds: no flicker). amplitude_px = peak sideways
                displacement near the camera
  water_ripple  (superseded by calm_water; still accepted by plans 1.4 / 1.5) two layers of
                drifting noise; reads as static

water_guard holds the water's edge and solid things reaching it (coping, railing bars) still.
Every effect function takes the Region (tour_worker.motion.region) as `r`."""
from __future__ import annotations

import math

import cv2
import numpy as np

from tour_worker.motion.masks import big_parts, crop_box, touching

# calm_water wave trains on the water plane: (direction from "toward the camera" deg, wavelength,
# speed in plane units per s, relative amplitude). Plane units: x ~ du / (v - v_h), z ~ 1 / (v - v_h).
CALM_TRAINS = ((-12, 0.090, 0.050, 1.00), (18, 0.060, 0.042, 0.70), (-35, 0.038, 0.034, 0.45),
               (50, 0.024, 0.028, 0.25))
_SX_MAX = sum(a * abs(math.sin(math.radians(d))) * 2 * math.pi / lam for d, lam, _, a in CALM_TRAINS)
_SZ_MAX = sum(a * abs(math.cos(math.radians(d))) * 2 * math.pi / lam for d, lam, _, a in CALM_TRAINS)
CALM_BRIGHTNESS = 0.04
# the approved prototype's slope -> light weights (tanh(0.0009 * raw slopes)), on normalised slopes
_SHADE_Z, _SHADE_X = 0.0009 * 0.6 * _SZ_MAX, 0.0009 * 0.4 * _SX_MAX


# ---- calm_water (the Region holds the water plane: v_h horizon, v_bot, u_mid, ref) ------------

def calm_irregular(r, u, v, t):
    """The slow (80-140 source px) variation of wave strength across the surface."""
    d = np.maximum(v - r.v_h, 1.0)
    return 1.0 + 0.35 * r.noise_a((u - r.u_mid) / d * 9.0, r.ref / d * 2.5, 0.25 * t)


def calm_slopes(r, u, v, t, irregular=None):
    d = np.maximum(v - r.v_h, 1.0)
    x, z = (u - r.u_mid) / d, r.ref / d
    sx, sz = np.zeros_like(u), np.zeros_like(u)
    for k, (ang, lam, spd, amp) in enumerate(CALM_TRAINS):
        a = math.radians(ang)
        kx, kz = math.sin(a) * 2 * math.pi / lam, math.cos(a) * 2 * math.pi / lam
        c = amp * np.cos(kx * x + kz * z - 2 * math.pi * spd / lam * t + 1.3 * k + r.phase)
        sx += c * kx
        sz += c * kz
    if irregular is None:
        irregular = 1.0 + 0.35 * r.noise_a(x * 9.0, z * 2.5, 0.25 * t)
    near = np.clip((v - r.v_h) / r.ref, 0.0, 1.0)
    return sx / _SX_MAX * irregular, sz / _SZ_MAX * irregular, near


def calm_feature_px(r) -> float:
    return 0.0                                     # calm_water: exact (no coarse grid)


def calm_displacement(r, u, v, t):
    sx, sz, near = calm_slopes(r, u, v, t)
    A = _calm_amp(r, t)
    return A * near * sx / 1.35, 0.5 * A * near * sz / 1.35


def _calm_amp(r, t: float) -> float:
    """Water does not follow the shot's wind (2026-10-22: calming it to 0.85 between gusts left
    thin distant lake bands under the readability floor; ripples at this scale barely respond
    to a breeze anyway)."""
    return r.amp


def calm_fields(r, u, v, t, irregular=None):
    """(dx, dy, brightness); irregular: calm_irregular at these points when already known."""
    sx, sz, near = calm_slopes(r, u, v, t, irregular)
    A = _calm_amp(r, t)
    return (A * near * sx / 1.35, 0.5 * A * near * sz / 1.35,
            1.0 + CALM_BRIGHTNESS * np.tanh(_SHADE_Z * sz + _SHADE_X * sx))


# ---- water_ripple ------------------------------------------------------------------------------

def ripple_feature_px(r) -> float:
    return min(max(30.0, 0.08 * r.bw), max(10.0, 0.03 * r.bh)) * 0.5


def ripple_displacement(r, u, v, t):
    a, s = r.anchor_weight(v)
    A, f = r.amp, r.freq
    sx, sy = max(30.0, 0.08 * r.bw), max(10.0, 0.03 * r.bh)   # wide, flat ripples
    n1 = r.noise_a(u / sx + 0.35 * f * t, v / sy, 0.5 * f * t)
    n2 = r.noise_b(u / (0.5 * sx) - 0.5 * f * t, v / (0.5 * sy), 0.9 * f * t)
    dx = A * a * (0.65 * n1 + 0.35 * n2)
    dy = 0.35 * A * a * r.noise_b(u / sx + 20.0, v / sy, 0.7 * f * t)
    return dx, dy


def ripple_fields(r, u, v, t, irregular=None):
    dx, dy = ripple_displacement(r, u, v, t)
    return dx, dy, None


# ---- what stays still --------------------------------------------------------------------------

def water_guard(src: np.ndarray, feather: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Water regions: dark grey solid things reaching the water's edge (railing bars in front of a
    pool, dock posts) and the water's own edge stay rigid, like solid objects in trees
    (leaf_weight): no moving pixel reaches them. Submerged things (steps, benches, lights) ripple."""
    box = crop_box(feather, int(reach) + 40)
    if box is None:
        return feather, np.zeros(feather.shape, bool)
    y0, y1, x0, x1 = box
    hsv = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).astype(np.float32)
    dark_grey = ((hsv[..., 2] / 255 < 0.3) & (hsv[..., 1] / 255 < 0.4)).astype(np.uint8)
    solid = big_parts(cv2.morphologyEx(dark_grey, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
                      max(30.0, 4.0 * reach))
    # (no pool-structure rule: COCO's water masks are the water surface only; ADE20K's "pool"
    # covered coping and tiled walls, and rules to carve them out also froze submerged steps -
    # review 2026-10-19, COCO is the primary model again)
    # the water's own edge too: ripples next to the pool's coping must not pull the coping in
    # (Turkey Trot pool, 2026-10-16: 0.3% of moving pixels dragged it, gate limit 0.2%)
    edge = feather[y0:y1, x0:x1] <= 0
    # the ripples' reach per row: calm_water moves a row by amp x near(v) (perspective: far rows
    # barely move), so the held band along the edge narrows with it (review 2026-10-19: a thin
    # distant lake lost most of its surface to a band sized for the near water)
    ys = np.nonzero((feather > 0).any(axis=1))[0]
    bh = float(ys.max() - ys.min() + 1)
    v_h = ys.min() - 0.9 * bh
    near = np.clip((np.arange(y0, y1, dtype=np.float32) - v_h) / (ys.max() + 1 - v_h), 0.0, 1.0)
    row_reach = (1.5 * reach * near + 2.0)[:, None]
    # ...but only structure that reaches the water's edge is held: coping, walls above the
    # waterline and railing bars meet or cross it; submerged steps, benches and lights are
    # surrounded by water and ripple with it (review 2026-10-19: the pool steps were held still)
    solid = touching(solid, edge, max(8.0, 2.0 * reach))
    out = feather.copy()
    held = solid | edge
    if held.any():
        dist = cv2.distanceTransform((~held).astype(np.uint8), cv2.DIST_L2, 5)
        g = np.clip((dist - 1.0) / np.maximum(row_reach, 1.0), 0, 1)
        out[y0:y1, x0:x1] *= (g * g * (3 - 2 * g)).astype(np.float32)
    full_solid = np.zeros(feather.shape, bool)
    full_solid[y0:y1, x0:x1] = solid
    return out, full_solid
