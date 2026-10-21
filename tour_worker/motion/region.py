"""Perceptual motion (plan 1.4): displacement fields inside object masks - the Region core, the
effect registry and the field evaluation the renderer uses (apply_regions).

Each region moves the photo's pixels in source space; tour_worker.render.frames composes that with
the camera warp in a single resample (no extra blur). Displacement is scaled by a feathered mask (0
at the mask edge, 1 inside), so the boundary never tears, and by an anchor profile, so a curtain
moves from the rod down and a plant from the pot up. Everything is smooth in time (sines +
gradient noise) and seeded, so renders stay deterministic.

The effects live one per module: fabric (fabric_sway), foliage (foliage_sway), canopy
(canopy_flutter), water (calm_water, water_ripple), fire (fire_flicker). Masks are loaded and
narrowed per effect by tour_worker.motion.loader.

Strengths were set by what can be SEEN, not by numbers: a time-slice (one image row stacked over
time) must show wiggling edges, not straight stripes (tests/motion/test_effects.py).
"""
from __future__ import annotations

import math
from typing import Callable, NamedTuple

import cv2
import numpy as np

from tour_worker.motion import canopy, fabric, fire, foliage, water
from tour_worker.motion.noise import GradientNoise3D


class Effect(NamedTuple):
    fields: Callable           # (r, u, v, t, irregular=None) -> (dx, dy, brightness or None)
    feature_px: Callable       # (r) -> size (source px) of the main spatial detail; 0 = exact grid
    displacement: Callable     # (r, u, v, t) -> (dx, dy)


EFFECTS = {
    "fabric_sway": Effect(fabric.fields, fabric.feature_px, fabric.displacement),
    "foliage_sway": Effect(foliage.fields, foliage.feature_px, foliage.displacement),
    "canopy_flutter": Effect(canopy.fields, canopy.feature_px, lambda r, u, v, t: canopy.fields(r, u, v, t)[:2]),
    "calm_water": Effect(water.calm_fields, water.calm_feature_px, water.calm_displacement),
    "water_ripple": Effect(water.ripple_fields, water.ripple_feature_px, water.ripple_displacement),
    # displacement() is the render's own motion (fields() without the light) for every effect:
    # tests of displacement() test what the renderer draws
    "fire_flicker": Effect(fire.fields, fire.feature_px, lambda r, u, v, t: fire.fields(r, u, v, t)[:2]),
}
# an effect outside the registry (the plan validator rejects them) behaves as before: foliage
# displacement, no brightness, an exact grid
_FALLBACK = Effect(foliage.fields, lambda r: 0.0, foliage.displacement)


class Region:
    """One motion region of a shot: its spec, feathered mask weight and per-region state (phases,
    noise, the calm_water plane). The loader adds effect state: solid (held objects), col_bottom /
    col_span (trees), flame / flame_h / flame_bottom / plume (fire)."""

    def __init__(self, spec, feather: np.ndarray, src_w: int, src_h: int):
        self.spec = spec
        self.effect, self.anchor = spec.effect, spec.anchor
        self.amp, self.freq = float(spec.amplitude_px), float(spec.freq_hz)
        x1, y1, x2, y2 = spec.bbox
        self.x0, self.y0 = x1 * src_w, y1 * src_h
        self.bw, self.bh = max(1.0, (x2 - x1) * src_w), max(1.0, (y2 - y1) * src_h)
        self.feather = feather
        self.src_w = src_w
        rng = np.random.default_rng(spec.seed & 0xFFFFFFFF)
        self.phase, self.phase2 = rng.uniform(0, 2 * math.pi, 2)
        if getattr(spec, "phase", None) is not None:   # planner-set: curtains vs plants opposite
            self.phase = float(spec.phase)
        self.noise_a = GradientNoise3D(spec.seed)
        self.noise_b = GradientNoise3D(spec.seed ^ 0x5F3759DF)
        # calm_water plane: a horizon above the far edge of the water
        self.v_bot = self.y0 + self.bh
        self.v_h = self.y0 - 0.9 * self.bh
        self.u_mid = self.x0 + self.bw / 2
        self.ref = self.v_bot - self.v_h

    @property
    def _fx(self) -> Effect:
        return EFFECTS.get(self.effect, _FALLBACK)

    def anchor_weight(self, v):
        if self.anchor == "none":           # water: the whole surface moves
            return np.ones_like(v), np.ones_like(v)
        if self.anchor == "top":            # hangs from a rod: still at the top, freest at the hem
            s = np.clip((v - self.y0) / self.bh, 0.0, 1.0)
            return s ** 1.3, s
        s = np.clip((self.y0 + self.bh - v) / self.bh, 0.0, 1.0)   # rooted: still at the base
        return s ** 1.5, s

    def fields(self, u, v, t, irregular=None):
        """(dx, dy, brightness or None) at source points; calm_water's waves are computed once
        (irregular: calm_irregular at these points when already known)."""
        return self._fx.fields(self, u, v, t, irregular)

    def displacement(self, u, v, t):
        """(dx, dy) in source px at source points (u, v), time t seconds."""
        return self._fx.displacement(self, u, v, t)

    def brightness(self, u, v, t):
        """Multiplicative brightness inside the region, or None (displacement-only effects)."""
        return self.fields(u, v, t)[2]

    def feature_px(self) -> float:
        """Size (source px) of the effect's main spatial detail: how fine a field grid it needs."""
        return self._fx.feature_px(self)

    def step_error(self, step: int, screen_per_src: float) -> float:
        """Predicted bilinear interpolation error (screen px) of a coarse field grid."""
        if self.effect == "canopy_flutter":
            return canopy.step_error(self, step, screen_per_src)
        feat = self.feature_px() * screen_per_src
        return self.amp * screen_per_src * (step / feat) ** 2 if feat > 0 else float("inf")

    # effect helpers, kept on the Region for callers (quality gate, tests)
    def calm_irregular(self, u, v, t):
        return water.calm_irregular(self, u, v, t)

    def leaf_px(self) -> float:
        return canopy.leaf_px(self)

    def branch_px(self) -> float:
        return canopy.branch_px(self)

    def canopy_amps(self) -> tuple[float, float]:
        return canopy.canopy_amps(self)

    def bend(self, u, v):
        return canopy.bend(self, u, v)

    def flame_px(self) -> float:
        return fire.flame_px(self)




# Fields are smooth, so they are computed on a coarse grid of output pixels (every FIELD_STEP px)
# and interpolated bilinearly (exactly, also over the uneven last cell) to every pixel; the mask
# weight itself stays per pixel. step=1 is the exact per-pixel path. Accuracy against step=1 is
# measured by tools/quality/perceptual_quality.py and held by tests (tests/test_perceptual.py).
# (rows, cols) grid spacing per effect, an upper limit: per frame the grid is also chosen from the
# effect's strength and main detail as seen on screen (adaptive_step: predicted error within
# FIELD_ERR_BUDGET). Steps stay in {1, 2, 4} so windows align to 4.
FIELD_STEP = {"canopy_flutter": (2, 2),     # small leaf clusters: a finer grid
              "calm_water": (1, 1)}         # far waves are a few px apart: no grid can hold them
DEFAULT_FIELD_STEP = (4, 4)
# calm_water's slow strength variation (calm_irregular) on a coarse grid; the waves stay exact
SMOOTH_STEP = {"calm_water": (2, 2)}   # 4 px left 0.35 px max error on far water (Turkey Trot pool)
_INTERP_CACHE: dict = {}


def field_step(effect: str) -> tuple[int, int]:
    return FIELD_STEP.get(effect, DEFAULT_FIELD_STEP)


def _coarse_grid(h: int, w: int, step: tuple[int, int]):
    """Coarse sample rows/cols (always incl. the last pixel) and full-res maps into that grid."""
    key = (h, w, step)
    if key not in _INTERP_CACHE:
        ys, xs = _sample_grid(h, w, *step)
        fx = np.interp(np.arange(w), xs, np.arange(len(xs))).astype(np.float32)
        fy = np.interp(np.arange(h), ys, np.arange(len(ys))).astype(np.float32)
        _INTERP_CACHE[key] = (ys, xs, *np.meshgrid(fx, fy))
    return _INTERP_CACHE[key]


ROI_PROBE = 8      # px: grid used to find where on screen a region can be


def _sample_grid(h: int, w: int, step_y: int, step_x: int):
    return (np.unique(np.r_[np.arange(0, h, step_y), h - 1]), np.unique(np.r_[np.arange(0, w, step_x), w - 1]))


def _cell_px(cu: np.ndarray, cv: np.ndarray) -> float:
    """The largest source-px distance between neighbouring grid points."""
    return max([1.0] + [float(np.abs(np.diff(c, axis=ax)).max()) for c in (cu, cv) for ax in (0, 1)
                        if c.shape[ax] > 1])


def _near_bbox(r: Region, cu: np.ndarray, cv: np.ndarray, m: float) -> np.ndarray:
    return (cu >= r.x0 - m) & (cu <= r.x0 + r.bw + m) & (cv >= r.y0 - m) & (cv <= r.y0 + r.bh + m)


def _screen_box(r: Region, map_x, map_y):
    """((r0, r1, c0, c1) rows/cols of these maps that can show the region, or None; screen px per
    source px, at its smallest)."""
    h, w = map_x.shape
    ys, xs = _sample_grid(h, w, ROI_PROBE, ROI_PROBE)
    pu, pv = map_x[np.ix_(ys, xs)], map_y[np.ix_(ys, xs)]
    # the camera's zoom from the median spacing (earlier regions may already displace the maps;
    # the box margins above keep the maximum)
    spacing = max([1e-6] + [float(np.median(np.abs(np.diff(c, axis=ax)))) for c, ax in ((pu, 1), (pv, 0))
                            if c.shape[ax] > 1])
    return probe_box(r, pu, pv, ys, xs, h, w), ROI_PROBE / spacing


def probe_box(r: Region, pu, pv, ys, xs, h: int, w: int):
    """The screen box from maps sampled at probe rows ys / cols xs (at most ROI_PROBE apart). The
    mask is clipped to the region's bbox, so a pixel that moves maps into the bbox; its nearest
    probe point is within one probe cell, so it is flagged with a 2-cell margin and the box
    grows by one probe."""
    hit = _near_bbox(r, pu, pv, 2.0 * _cell_px(pu, pv))
    if not hit.any():
        return None
    rows, cols = np.nonzero(hit.any(axis=1))[0], np.nonzero(hit.any(axis=0))[0]
    return (max(0, int(ys[rows[0]]) - ROI_PROBE), min(h, int(ys[rows[-1]]) + ROI_PROBE + 1),
            max(0, int(xs[cols[0]]) - ROI_PROBE), min(w, int(xs[cols[-1]]) + ROI_PROBE + 1))


def _coarse_eval(r: Region, fn, n_out: int, map_x, map_y, step: tuple[int, int], box, origin, frame_shape):
    """Evaluate fn(u, v) -> n_out arrays (or None entries) on the frame's coarse grid around box
    (r0, r1, c0, c1, frame px) and interpolate to full res. The box is aligned to grid points so
    every pixel has its 4 neighbours inside. Returns (R, [arrays or None]); R: the aligned box as
    slices of these maps (which start at origin). Points farther than a cell from the region's
    bbox (source px) cannot feed a moving pixel: they are left at 0 (1 for the third output)."""
    oy, ox = origin
    ys, xs, gx, gy = _coarse_grid(*frame_shape, step)
    r0, r1, c0, c1 = box
    i0, i1 = np.searchsorted(ys, r0, "right") - 1, np.searchsorted(ys, r1 - 1, "left")
    j0, j1 = np.searchsorted(xs, c0, "right") - 1, np.searchsorted(xs, c1 - 1, "left")
    cu = map_x[np.ix_(ys[i0:i1 + 1] - oy, xs[j0:j1 + 1] - ox)].astype(np.float64)
    cv = map_y[np.ix_(ys[i0:i1 + 1] - oy, xs[j0:j1 + 1] - ox)].astype(np.float64)
    need = _near_bbox(r, cu, cv, 1.5 * _cell_px(cu, cv))
    vals = fn(cu[need], cv[need]) if need.any() else [None] * n_out
    G = (slice(int(ys[i0]), int(ys[i1]) + 1), slice(int(xs[j0]), int(xs[j1]) + 1))
    lx = np.ascontiguousarray(gx[G] - np.float32(j0))
    ly = np.ascontiguousarray(gy[G] - np.float32(i0))
    out = []
    for k, val in enumerate(vals):
        if val is None:
            out.append(None)
            continue
        c = np.full(cu.shape, 1.0 if k == 2 else 0.0, np.float32)
        c[need] = val
        out.append(cv2.remap(c, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    if not need.any() and n_out >= 2:
        out[:2] = [np.zeros(lx.shape, np.float32)] * 2
    R = (slice(G[0].start - oy, G[0].stop - oy), slice(G[1].start - ox, G[1].stop - ox))
    return R, out


FIELD_ERR_BUDGET = 0.02   # screen px: predicted interpolation error a coarse grid may add


def adaptive_step(r: Region, configured: tuple[int, int], screen_per_src: float) -> tuple[int, int]:
    """The coarsest grid (4, 2 or exact) whose interpolation error stays small: bilinear error
    grows with the strength and with (step / detail size)^2 - est. strength x (step / detail)^2,
    calibrated on Downing / Turkey Trot (curtains at 7 px with step 4 over 48 px detail: 0.06 px
    p99 - too much; trees at 10 px flutter with step 2 over ~25 px leaves: 0.35 px max)."""
    for st in (4, 2):
        if st <= min(configured) and r.step_error(st, screen_per_src) <= FIELD_ERR_BUDGET:
            return (min(configured[0], st), min(configured[1], st))
    return (1, 1)


def field_align(regions: list[Region]) -> int:
    """A window whose bounds are multiples of this (or the frame's last row/col) contains every
    coarse grid point its regions use."""
    steps = [s for r in regions for s in field_step(r.effect) + SMOOTH_STEP.get(r.effect, ())]
    assert all(s in (1, 2, 4) for s in steps), "field steps must be 1, 2 or 4 (window alignment)"
    return int(np.lcm.reduce([max(1, s) for s in steps] or [1]))


def apply_regions(map_x: np.ndarray, map_y: np.ndarray, regions: list[Region], t: float,
                  step: int | tuple[int, int] | None = None, origin: tuple[int, int] = (0, 0),
                  frame_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """Shift the source sampling maps (output px -> source px) in place so the content inside
    each mask moves by its displacement: sampling from (u - dx) shows the pixel moved by +dx.
    Returns a per-output-pixel brightness gain when any region changes brightness, else None.
    step: field grid in output px (None: per effect, FIELD_STEP; 1: exact per pixel).
    The maps may be a window of the frame starting at origin (rows, cols) of a frame of
    frame_shape, aligned as field_align() says; results per pixel are the same as full-frame.
    Work is limited to the part of the screen where each region can be (memory traffic, not
    arithmetic, is what limits the speed); per pixel the results are the same as full-frame."""
    gain = None
    frame_shape = frame_shape or map_x.shape
    oy, ox = origin
    for r in regions:
        box, screen_per_src = _screen_box(r, map_x, map_y)
        if box is None:
            continue
        if step is None:
            st = adaptive_step(r, field_step(r.effect), screen_per_src)
        else:
            st = (step, step) if isinstance(step, int) else tuple(step)
        gbox = (box[0] + oy, box[1] + oy, box[2] + ox, box[3] + ox)
        smooth = SMOOTH_STEP.get(r.effect) if step is None else None
        irr = None
        if max(st) > 1:
            R, (fdx, fdy, fb) = _coarse_eval(r, lambda u, v: r.fields(u, v, t), 3, map_x, map_y, st, gbox,
                                             origin, frame_shape)
        elif smooth:
            R, (irr,) = _coarse_eval(r, lambda u, v: (r.calm_irregular(u, v, t),), 1, map_x, map_y, smooth,
                                     gbox, origin, frame_shape)
        else:
            R = (slice(box[0], box[1]), slice(box[2], box[3]))
        mx, my = map_x[R], map_y[R]                     # views: writes land in the full maps
        w = cv2.remap(r.feather, np.ascontiguousarray(mx), np.ascontiguousarray(my), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        idx = w > 1e-3
        if not idx.any():
            continue
        u, v = mx[idx].astype(np.float64), my[idx].astype(np.float64)
        if max(st) <= 1:
            dx, dy, b = r.fields(u, v, t, None if irr is None else irr[idx].astype(np.float64))
        else:
            dx, dy = fdx[idx].astype(np.float64), fdy[idx].astype(np.float64)
            b = fb[idx].astype(np.float64) if fb is not None else None
        ww = w[idx].astype(np.float64)
        if b is not None:
            if gain is None:
                gain = np.ones(map_x.shape, np.float32)
            g = gain[R]
            g[idx] *= (1.0 + ww * (b - 1.0)).astype(np.float32)
        mx[idx] = (u - ww * dx).astype(np.float32)
        my[idx] = (v - ww * dy).astype(np.float32)
    return gain
