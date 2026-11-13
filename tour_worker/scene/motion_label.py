"""Label a splat scene's moving things (curtains, plants) from segmented views, and give each
labelled Gaussian its motion (tour_worker/scene/motion3d.py). Run once per scene, offline
(tools/scene/label_motion.py).

  views   frames of the scene with known cameras (e.g. a rendered tour: each frame's camera comes
          from the plan) and their Mask2Former masks (the tour_segment job)
  vote    every Gaussian is projected into every view; when its centre is visible there (a depth
          buffer of the opaque Gaussians) it is "seen", and when it also falls in a curtain / plant
          mask it is a "hit". A Gaussian is labelled when hits >= MIN_HITS and hits / seen >= MIN_SHARE.
  grow    Gaussians never seen (the back of a curtain, the inside of a plant) take the label of a
          labelled neighbour within one GROW_VOXEL - so the whole object moves, not its front skin.
  objects labelled Gaussians are grouped into objects (connected OBJECT_VOXEL cells); tiny groups
          are dropped (a stray hit on a wall).
  motion  curtain: a sway along its normal, largest at the hem and zero at the rod (weight
          ((top - h) / height) ** CURTAIN_POWER), a slow wave across its width, and a smaller
          sideways drift. plant: a whole-plant sway that grows toward the tips (weight
          ((h - base) / height) ** PLANT_POWER) plus a leaf flutter whose phase and direction are
          shared by every Gaussian in a FLUTTER_CELL - leaves move as pieces, they do not fizz.

Everything is deterministic: the same scene and views give the same file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from tour_worker.scene.motion3d import TERMS, MotionField
from tour_worker.scene.splat_io import SplatModel

CLASSES = {"curtain": 1, "potted_plant": 2}         # blinds, trees, water: not moved indoors (yet)
OPAQUE = 0.5                                         # Gaussians that hide what is behind them
ZBUF_DOWN = 4                                        # depth buffer at 1/4 of the mask resolution
DEPTH_TOL = (0.03, 0.04)                             # visible: depth <= zbuf * (1 + 0.04) + 0.03 m
MIN_HITS, MIN_SHARE = 2, 0.5
GROW_VOXEL = 0.03
OBJECT_VOXEL = 0.05
MIN_OBJECT = 200                                     # Gaussians

CURTAIN_AMP, CURTAIN_POWER = 0.030, 1.6              # metres at the hem
CURTAIN_FREQ = (0.21, 0.34)                          # Hz: the sway, the sideways drift
CURTAIN_WAVE_M = (1.0, 0.7)                          # wavelength across the width
CURTAIN_DRIFT = 0.35                                 # sideways drift, share of the sway
PLANT_AMP, PLANT_POWER, PLANT_FREQ = 0.015, 1.8, 0.42
FLUTTER_AMP, FLUTTER_FREQ, FLUTTER_CELL = 0.004, 1.25, 0.04


@dataclass
class View:
    viewmat: np.ndarray            # 4x4 world -> camera (OpenCV axes)
    K: np.ndarray                  # 3x3 at (width, height)
    width: int
    height: int
    masks: dict                    # class name -> (h, w) bool mask (any resolution: scaled to it)


def _hash01(c: np.ndarray, salt: float) -> np.ndarray:
    """A deterministic pseudo-random value in [0, 1) per integer cell (N, 3)."""
    x = np.sin(c[:, 0] * 12.9898 + c[:, 1] * 78.233 + c[:, 2] * 37.719 + salt) * 43758.5453
    return (x - np.floor(x)).astype(np.float64)


def vote(model: SplatModel, views: list[View]) -> tuple[np.ndarray, np.ndarray]:
    """(seen (N,), hits (N, len(CLASSES)+1)) counts over the views."""
    pts = model.means.astype(np.float64)
    n = len(pts)
    seen = np.zeros(n, np.int32)
    hits = np.zeros((n, len(CLASSES) + 1), np.int32)
    opaque = model.opacities >= OPAQUE
    for v in views:
        R, t = v.viewmat[:3, :3], v.viewmat[:3, 3]
        cam = pts @ R.T + t
        z = cam[:, 2]
        front = z > 0.05
        u = np.full(n, -1.0); w = np.full(n, -1.0)
        u[front] = v.K[0, 0] * cam[front, 0] / z[front] + v.K[0, 2]
        w[front] = v.K[1, 1] * cam[front, 1] / z[front] + v.K[1, 2]
        inside = front & (u >= 0) & (u < v.width) & (w >= 0) & (w < v.height)
        zw, zh = max(1, v.width // ZBUF_DOWN), max(1, v.height // ZBUF_DOWN)
        zu = (u * zw / v.width).astype(np.int64); zv = (w * zh / v.height).astype(np.int64)
        zbuf = np.full(zh * zw, np.inf)
        occ = inside & opaque
        np.minimum.at(zbuf, zv[occ] * zw + zu[occ], z[occ])
        zb = np.full(n, np.inf)
        zb[inside] = zbuf[zv[inside] * zw + zu[inside]]
        visible = inside & (z <= zb * (1 + DEPTH_TOL[1]) + DEPTH_TOL[0])
        seen += visible
        for name, cid in CLASSES.items():
            m = v.masks.get(name)
            if m is None:
                continue
            mh, mw = m.shape
            mu = np.clip((u * mw / v.width).astype(np.int64), 0, mw - 1)
            mv = np.clip((w * mh / v.height).astype(np.int64), 0, mh - 1)
            hit = visible.copy()
            hit[visible] = m[mv[visible], mu[visible]]
            hits[:, cid] += hit
    return seen, hits


def label(model: SplatModel, views: list[View]) -> np.ndarray:
    """(N,) int8: 0 still, else a CLASSES id."""
    seen, hits = vote(model, views)
    lab = np.zeros(len(model), np.int8)
    best = np.argmax(hits[:, 1:], axis=1) + 1
    h = hits[np.arange(len(model)), best]
    ok = (h >= MIN_HITS) & (h >= MIN_SHARE * np.maximum(seen, 1))
    lab[ok] = best[ok]
    return grow(model, lab, seen)


def grow(model: SplatModel, lab: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Labelled classes grown into Gaussians no view saw (the far side of a curtain, a plant's
    inside): an unseen Gaussian within one GROW_VOXEL of a labelled one takes its label."""
    lab = lab.copy()
    pts = model.means.astype(np.float64)
    cells = np.floor(pts / GROW_VOXEL).astype(np.int64)
    for cid in CLASSES.values():
        src = cells[lab == cid]
        if not len(src):
            continue
        lo = src.min(0) - 1
        dims = src.max(0) - lo + 2
        grid = np.zeros(tuple(dims), bool)
        grid[tuple((src - lo).T)] = True
        grid = ndi.binary_dilation(grid, structure=np.ones((3, 3, 3), bool))
        cand = (lab == 0) & (seen == 0)
        c = cells[cand] - lo
        inb = ((c >= 0) & (c < dims)).all(1)
        grow = np.zeros(cand.sum(), bool)
        grow[inb] = grid[tuple(c[inb].T)]
        idx = np.flatnonzero(cand)[grow]
        lab[idx] = cid
    return lab


def objects(model: SplatModel, lab: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """[(class id, indices)] per object, ordered (class, then position) so ids are stable."""
    out = []
    pts = model.means.astype(np.float64)
    for cid in sorted(CLASSES.values()):
        idx = np.flatnonzero(lab == cid)
        if not len(idx):
            continue
        cells = np.floor(pts[idx] / OBJECT_VOXEL).astype(np.int64)
        lo = cells.min(0)
        grid = np.zeros(tuple(cells.max(0) - lo + 1), bool)
        grid[tuple((cells - lo).T)] = True
        comp, k = ndi.label(grid, structure=np.ones((3, 3, 3), bool))
        cid_of = comp[tuple((cells - lo).T)]
        groups = []
        for j in range(1, k + 1):
            members = idx[cid_of == j]
            if len(members) >= MIN_OBJECT:
                groups.append(members)
        groups.sort(key=lambda m: tuple(np.round(pts[m].mean(0), 3)))
        out += [(cid, m) for m in groups]
    return out


def _horizontal_basis(up: np.ndarray):
    up = up / np.linalg.norm(up)
    a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - up * (a @ up); e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def build_field(model: SplatModel, lab: np.ndarray, up, scene_sha256: str) -> MotionField:
    """The motion of every labelled Gaussian (objects() groups; unlabelled ones stay still)."""
    up = np.asarray(up if up is not None else (0, 0, 1), np.float64); up /= np.linalg.norm(up)
    e1, e2 = _horizontal_basis(up)
    pts = model.means.astype(np.float64)
    idx_all, dirs_all, amp_all, freq_all, ph_all, counts = [], [], [], [], [], {}
    for k, (cid, m) in enumerate(objects(model, lab)):
        p = pts[m]
        h = p @ up
        rng = np.random.default_rng(1000 * cid + k)
        dirs = np.zeros((len(m), TERMS, 3)); amp = np.zeros((len(m), TERMS))
        freq = np.zeros((len(m), TERMS)); ph = np.zeros((len(m), TERMS))
        if cid == CLASSES["curtain"]:
            top, bot = np.percentile(h, [98, 2])
            height = max(top - bot, 0.3)
            wgt = np.clip((top - h) / height, 0, 1) ** CURTAIN_POWER
            xy = np.c_[p @ e1, p @ e2]
            xy0 = xy - xy.mean(0)
            _, vecs = np.linalg.eigh(np.cov(xy0.T))
            major, minor = vecs[:, 1], vecs[:, 0]
            width_dir = major[0] * e1 + major[1] * e2
            normal = minor[0] * e1 + minor[1] * e2
            s = xy0 @ major
            base = rng.uniform(0, 2 * math.pi)
            dirs[:, 0] = normal; dirs[:, 1] = width_dir
            amp[:, 0] = CURTAIN_AMP * wgt; amp[:, 1] = CURTAIN_AMP * CURTAIN_DRIFT * wgt
            freq[:, 0], freq[:, 1] = CURTAIN_FREQ
            ph[:, 0] = base + 2 * math.pi * s / CURTAIN_WAVE_M[0]
            ph[:, 1] = base + 1.3 + 2 * math.pi * s / CURTAIN_WAVE_M[1]
            name = "curtain"
        else:
            bot, top = np.percentile(h, [3, 97])
            height = max(top - bot, 0.15)
            wgt = np.clip((h - bot) / height, 0, 1) ** PLANT_POWER
            ang = rng.uniform(0, 2 * math.pi)
            dirs[:, 0] = math.cos(ang) * e1 + math.sin(ang) * e2
            amp[:, 0] = PLANT_AMP * wgt
            freq[:, 0] = PLANT_FREQ
            ph[:, 0] = rng.uniform(0, 2 * math.pi) + 0.8 * wgt          # tips lag the stem a little
            cells = np.floor(p / FLUTTER_CELL).astype(np.int64)
            a1, a2 = 2 * math.pi * _hash01(cells, 1.0 + k), np.arccos(1 - 2 * _hash01(cells, 7.0 + k))
            d = np.c_[np.sin(a2) * np.cos(a1), np.sin(a2) * np.sin(a1), np.cos(a2)]
            dirs[:, 1] = d
            amp[:, 1] = FLUTTER_AMP * wgt
            freq[:, 1] = FLUTTER_FREQ
            ph[:, 1] = 2 * math.pi * _hash01(cells, 13.0 + k)
            name = "potted_plant"
        keep = amp.max(1) > 1e-5                     # the rod / pot stay still: no need to list them
        idx_all.append(m[keep]); dirs_all.append(dirs[keep]); amp_all.append(amp[keep])
        freq_all.append(freq[keep]); ph_all.append(ph[keep])
        counts[name] = counts.get(name, 0) + int(keep.sum())
    if idx_all:
        idx = np.concatenate(idx_all); order = np.argsort(idx)
        cat = lambda xs: np.concatenate(xs)[order].astype(np.float32)   # noqa: E731
        return MotionField(idx[order].astype(np.int64), cat(dirs_all), cat(amp_all), cat(freq_all), cat(ph_all),
                           scene_sha256, len(model), counts)
    z = np.zeros((0, TERMS), np.float32)
    return MotionField(np.zeros(0, np.int64), np.zeros((0, TERMS, 3), np.float32), z, z, z, scene_sha256,
                       len(model), {})


# ---- motion from the scene's labels (scene/semantics.py) ----------------------------------------------
FLAME_MIN = 300                       # bright warm Gaussians in a fireplace: it is lit
FLAME_AMP_UP, FLAME_AMP_SIDE = 0.035, 0.020
FLAME_FREQ_UP, FLAME_FREQ_SIDE = (1.6, 2.6), (0.9, 1.4)
FLAME_CELL = 0.04


def _bright_warm(model: SplatModel) -> np.ndarray:
    from tour_worker.scene.splat_io import SH_C0
    raw = 0.5 + SH_C0 * model.sh[:, 0]
    return (raw.max(1) > 0.9) & ((raw[:, 0] >= 1.15 * raw[:, 2]) | (raw.min(1) > 0.92))


def flame_field(model: SplatModel, local: np.ndarray, fireplaces: list[dict], up_scene, R, scene_sha256: str):
    """Flames in the lit fireplaces: the bright, warm Gaussians inside each fireplace's box (the
    largest connected cluster), each 4 cm patch licking upward (a wave travelling up the flame) and
    swaying along the firebox; zero at the flame's base, most at its tips. None when no fire is lit."""
    up = np.asarray(up_scene, np.float64)
    hot = _bright_warm(model)
    idx_all, dirs_all, amp_all, freq_all, ph_all = [], [], [], [], []
    for k, f in enumerate(fireplaces):
        lo, hi = np.asarray(f["min"]) - 0.1, np.asarray(f["max"]) + 0.1
        inside = np.flatnonzero(hot & np.all((local >= lo) & (local <= hi), axis=1))
        if len(inside) < FLAME_MIN:
            continue
        cells = np.floor(local[inside] / 0.05).astype(np.int64)
        c0 = cells.min(0)
        grid = np.zeros(tuple(cells.max(0) - c0 + 1), bool)
        grid[tuple((cells - c0).T)] = True
        comp, n = ndi.label(grid, structure=np.ones((3, 3, 3), bool))
        cid = comp[tuple((cells - c0).T)]
        big = np.bincount(cid)[1:].argmax() + 1
        idx = inside[cid == big]
        if len(idx) < FLAME_MIN:
            continue
        z = local[idx, 2]
        base, tip = np.percentile(z, [10, 98])
        h = np.clip((z - base) / max(tip - base, 0.1), 0, 1)
        a = math.radians(f["footprint"]["axis_deg"])
        side = np.asarray(R, np.float64).T @ np.array([math.cos(a), math.sin(a), 0.0])   # along the firebox
        cells = np.floor(local[idx][:, :2] / FLAME_CELL).astype(np.int64)
        c3 = np.c_[cells, np.zeros(len(cells), np.int64)]
        r = [_hash01(c3, 31.0 + k + s_) for s_ in range(4)]
        amp = np.stack([FLAME_AMP_UP * h ** 1.3, FLAME_AMP_SIDE * h ** 1.5], 1)
        freq = np.stack([FLAME_FREQ_UP[0] + (FLAME_FREQ_UP[1] - FLAME_FREQ_UP[0]) * r[0],
                         FLAME_FREQ_SIDE[0] + (FLAME_FREQ_SIDE[1] - FLAME_FREQ_SIDE[0]) * r[1]], 1)
        ph = np.stack([2 * math.pi * r[2] - 2 * math.pi * 1.2 * h, 2 * math.pi * r[3] - math.pi * h], 1) % (2 * math.pi)
        dirs = np.stack([np.tile(up, (len(idx), 1)), np.tile(side, (len(idx), 1))], 1)
        idx_all.append(idx); dirs_all.append(dirs); amp_all.append(amp); freq_all.append(freq); ph_all.append(ph)
    if not idx_all:
        return None
    idx = np.concatenate(idx_all)
    order = np.argsort(idx)
    cat = lambda xs: np.concatenate(xs)[order].astype(np.float32)   # noqa: E731
    return MotionField(idx[order].astype(np.int64), cat(dirs_all), cat(amp_all), cat(freq_all), cat(ph_all),
                       scene_sha256, len(model), {"flame": int(len(idx))})


def merge_fields(fields: list, scene_sha256: str, splats: int) -> MotionField | None:
    """One motion field from several (curtains / plants, flames); a Gaussian in two keeps the first."""
    fields = [f for f in fields if f is not None and len(f)]
    if not fields:
        return None
    idx = np.concatenate([f.idx for f in fields])
    _, first = np.unique(idx, return_index=True)
    keep = np.sort(first)
    cat = lambda name: np.concatenate([getattr(f, name) for f in fields])[keep]   # noqa: E731
    order = np.argsort(idx[keep])
    classes: dict = {}
    for f in fields:
        for k, v in f.classes.items():
            classes[k] = classes.get(k, 0) + v
    return MotionField(idx[keep][order].astype(np.int64), cat("dirs")[order], cat("amp")[order], cat("freq")[order],
                       cat("phase")[order], scene_sha256, splats, classes)


def motion_from_labels(model: SplatModel, scene_labels, seen: np.ndarray, local: np.ndarray, up_scene, R,
                       scene_sha256: str) -> MotionField | None:
    """The scene's motion from its labels: curtains sway, plants stir (grown into the Gaussians no
    view saw), lit fireplaces flicker."""
    lab = np.zeros(len(model), np.int8)
    lab[scene_labels.mask("curtain")] = CLASSES["curtain"]
    lab[scene_labels.mask("plant")] = CLASSES["potted_plant"]
    lab = grow(model, lab, seen)
    sway = build_field(model, lab, up_scene, scene_sha256)
    fires = [o for o in scene_labels.objects if o["class"] == "fireplace"]
    flames = flame_field(model, local, fires, up_scene, R, scene_sha256)
    return merge_fields([sway, flames], scene_sha256, len(model))
