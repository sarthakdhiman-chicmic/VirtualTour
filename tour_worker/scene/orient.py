"""Which way is up in a splat scene, and where its floor and ceiling are.

A trained scene comes in whatever axes the capture used: z up, y up, COLMAP's y down, or tilted.
With no scene.json, the worker finds up from the geometry:

  axis   floors and ceilings are flat, so along the true up axis the opaque splats bunch into two
         sharp peaks at the ends of the range. Each principal axis and each file axis is tried; the
         one with the strongest pair of end peaks wins (exact to ~1.5 deg on the sample, tilted 20 deg).
  sign   furniture stands on the floor: between the two planes, more mass sits in the lower half.
         That cue is weak (1.18x on the sample apartment), so when it is not clearly lopsided the
         file's convention decides (a z axis: +z up; a y axis: -y up, as COLMAP / 3DGS write it)
         and the result says up is uncertain. An explicit up (the plan's scene.up_axis, from the
         API) always wins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tour_worker.scene import SceneInvalid
from tour_worker.scene.splat_io import SplatModel

SAMPLE = 400_000
OPAQUE = 0.5
BINS = 100
END_BINS = 20                # where the floor / ceiling peaks are looked for, at each end
SIGN_RATIO = 1.1             # lower / upper mass (or its inverse) at least this: the sign is trusted
MIN_PEAK_SCORE = 1.5         # weaker end peaks: no clear floor and ceiling (an exterior, a single wall)


@dataclass
class SceneFrame:
    up: np.ndarray                     # unit vector, scene axes
    floor: float                       # height of the floor along up (scene units)
    ceiling: float                     # height of the ceiling along up
    confident: bool                    # the up direction (sign included) is trusted
    source: str                        # "given" | "geometry" | "convention"
    warnings: list = field(default_factory=list)

    @property
    def room_height(self) -> float:
        return self.ceiling - self.floor

    def height(self, points: np.ndarray) -> np.ndarray:
        """Height above the floor of (N, 3) points."""
        return points @ self.up - self.floor


def _sample(model: SplatModel, seed: int = 0) -> np.ndarray:
    keep = model.opacities > OPAQUE
    pts = model.means[keep] if keep.sum() >= 1000 else model.means
    if len(pts) < 100:
        raise SceneInvalid("the scene has too few opaque splats to find its floor")
    if len(pts) > SAMPLE:
        pts = pts[np.random.default_rng(seed).choice(len(pts), SAMPLE, replace=False)]
    return pts.astype(np.float64)


def _profile(h: np.ndarray):
    lo, hi = np.percentile(h, [0.5, 99.5])
    hist, edges = np.histogram(h, bins=BINS, range=(lo, hi))
    hist = hist / max(1, hist.sum())
    i_lo = int(np.argmax(hist[:END_BINS]))
    i_hi = BINS - END_BINS + int(np.argmax(hist[-END_BINS:]))
    score = min(hist[i_lo], hist[i_hi]) / (np.median(hist) + 1e-9)
    centres = (edges[:-1] + edges[1:]) / 2
    return score, hist, i_lo, i_hi, centres


def _refine_axis(c: np.ndarray, axis: np.ndarray, rounds: int = 4) -> np.ndarray:
    """Tighten the up axis: the normal of planes fitted to the floor and ceiling points (the
    smallest principal direction of each slab), averaged; repeated as the slabs sharpen."""
    for _ in range(rounds):
        _, hist, i_lo, i_hi, centres = _profile(c @ axis)
        h = c @ axis
        tol = 1.5 * (centres[1] - centres[0])
        normals = []
        for i in (i_lo, i_hi):
            slab = c[np.abs(h - centres[i]) < tol]
            if len(slab) < 200:
                continue
            for _trim in range(3):              # refit without the points off the plane (wall feet)
                mean = np.median(slab, 0)
                _, vecs = np.linalg.eigh(np.cov((slab - mean).T))
                n = vecs[:, 0]
                r = np.abs((slab - mean) @ n)
                keep = r <= max(np.percentile(r, 60), 1e-9)
                if keep.sum() < 200:
                    break
                slab = slab[keep]
            normals.append(n if n @ axis >= 0 else -n)
        if not normals:
            break
        new = np.mean(normals, 0)
        new /= np.linalg.norm(new)
        if np.degrees(np.arccos(np.clip(new @ axis, -1, 1))) > 25:     # a slab that is not a plane
            break
        axis = new
    return axis


def estimate_frame(model: SplatModel, up_hint=None) -> SceneFrame:
    """The scene's up, floor and ceiling. up_hint: an explicit up (it wins; only the planes are found)."""
    pts = _sample(model)
    centre = np.median(pts, 0)
    c = pts - centre
    warnings: list = []
    if up_hint is not None:
        up = np.asarray(up_hint, np.float64)
        up = up / np.linalg.norm(up)
        score, hist, i_lo, i_hi, centres = _profile(c @ up)
        source, confident = "given", True
    else:
        _, vecs = np.linalg.eigh(np.cov(c.T))
        best = None
        for a in [vecs[:, i] for i in range(3)] + list(np.eye(3)):
            prof = _profile(c @ a)
            if best is None or prof[0] > best[0][0] + 1e-9:
                best = (prof, a)
        (score, hist, i_lo, i_hi, centres), axis = best
        axis = _refine_axis(c, axis / np.linalg.norm(axis))
        score, hist, i_lo, i_hi, centres = _profile(c @ axis)
        # mass in the lower and upper halves between the planes, the same 6 % margin off each plane
        h = c @ axis
        lo_h, hi_h = centres[i_lo], centres[i_hi]
        span, mid_h = hi_h - lo_h, 0.5 * (lo_h + hi_h)
        lower = float(((h > lo_h + 0.06 * span) & (h < mid_h)).mean())
        upper = float(((h > mid_h) & (h < hi_h - 0.06 * span)).mean())
        ratio = (lower + 1e-9) / (upper + 1e-9)
        if ratio >= SIGN_RATIO or ratio <= 1 / SIGN_RATIO:
            up, source, confident = (axis if ratio >= 1 else -axis), "geometry", True
        else:
            # the file's convention: the file axis nearest the found axis, + for z, - for y
            k = int(np.argmax(np.abs(axis)))
            sign = np.sign(axis[k]) or 1.0
            want = 1.0 if k == 2 else -1.0 if k == 1 else 1.0
            up = axis * sign * want
            source, confident = "convention", False
            warnings.append("scene_up_uncertain")
        if score < MIN_PEAK_SCORE:
            confident = False
            warnings.append("scene_floor_unclear")
        # heights were measured along `axis`; flip the profile if up points the other way
        if np.dot(up, axis) < 0:
            hist, centres = hist[::-1], -centres[::-1]
            i_lo, i_hi = BINS - 1 - i_hi, BINS - 1 - i_lo
    base = float(centre @ up)
    return SceneFrame(up=up, floor=base + float(centres[i_lo]), ceiling=base + float(centres[i_hi]),
                      confident=confident, source=source, warnings=warnings)
