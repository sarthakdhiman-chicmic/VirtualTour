"""Mask helpers shared by the motion effects: loading and feathering a region's mask, sampling a
field at points, the leaf measure, and the "hold this still" guards around solid objects."""
from __future__ import annotations

import os

import cv2
import numpy as np

FEATHER_FRAC = 0.04            # feather width as a fraction of the region's shorter side
FEATHER_PX = (4.0, 16.0)
# a mask must belong to its detection: most of it inside the (slightly grown) bbox, and filling a
# fair part of it. Stored detect masks can be out of step with the detection list (seen: a curtain
# detection whose mask file was a side table), and moving the wrong object is worse than no motion.
BOX_MARGIN_FRAC = 0.03
MIN_INSIDE_SHARE = 0.8
MIN_BOX_FILL = 0.15


def mask_relpath(shot_id: str, region_id: str) -> str:
    """Where a region's mask lives under the job's images dir."""
    return os.path.join("masks", f"{shot_id}_{region_id}.png")


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def load_feather(mask_path: str, src_w: int, src_h: int, bbox_px) -> tuple[np.ndarray | None, str]:
    """The mask scaled to the source photo, checked against its bbox, clipped to it and feathered
    inward: (float32 (src_h, src_w) in [0, 1], "") or (None, reason)."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None, "unreadable"
    if m.shape != (src_h, src_w):
        # the mask is the detected image's size; it must be the same photo (same aspect)
        if abs(m.shape[1] / m.shape[0] - src_w / src_h) > 0.02:
            return None, "aspect"
        m = cv2.resize(m, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    binary = (m >= 128).astype(np.uint8)
    total = int(binary.sum())
    if not total:
        return None, "empty"
    bw, bh = bbox_px[2] - bbox_px[0], bbox_px[3] - bbox_px[1]
    mx, my = BOX_MARGIN_FRAC * src_w, BOX_MARGIN_FRAC * src_h
    x0, y0 = max(0, int(bbox_px[0] - mx)), max(0, int(bbox_px[1] - my))
    x1, y1 = min(src_w, int(np.ceil(bbox_px[2] + mx))), min(src_h, int(np.ceil(bbox_px[3] + my)))
    inside = int(binary[y0:y1, x0:x1].sum())
    if inside < MIN_INSIDE_SHARE * total or inside < MIN_BOX_FILL * max(1.0, bw * bh):
        return None, "mismatch"
    clipped = np.zeros_like(binary)
    clipped[y0:y1, x0:x1] = binary[y0:y1, x0:x1]
    binary = clipped
    feather = min(max(FEATHER_FRAC * min(bw, bh), FEATHER_PX[0]), FEATHER_PX[1])
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # the mask's outermost pixel ring stays at exactly 0, so bilinear sampling just outside the
    # mask (between pixel centres) never picks up a non-zero weight
    return _smoothstep((dist - 1.0) / feather).astype(np.float32), ""


def sample_bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """img at the points (u, v), bilinear, 0 outside. cv2.remap caps output sides at 32767, so the
    points are laid out as rows of 4096."""
    n = u.size
    if n == 0:
        return np.zeros(u.shape, np.float32)
    w = 4096
    rows = -(-n // w)
    pad = rows * w - n
    mu = np.pad(u.astype(np.float32).ravel(), (0, pad)).reshape(rows, w)
    mv = np.pad(v.astype(np.float32).ravel(), (0, pad)).reshape(rows, w)
    out = cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.ravel()[:n].reshape(u.shape)


def leafiness(crop: np.ndarray, soften: bool = True) -> np.ndarray:
    """0..1 from the photo: foliage hue (yellow-green to green-cyan) with some saturation and
    light, scaled by fine texture (leaves are busy; trunks and sky are smooth), softened 2 px
    (unsoftened for finding thin solid objects: softening lets green bleed over a lantern rod)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, sat, val = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    w = (np.clip(1 - np.abs(h - 55) / 40, 0, 1) ** 0.5            # OpenCV hue: green = 60
         * np.clip((sat - 0.12) / 0.15, 0, 1) * np.clip((val - 0.06) / 0.08, 0, 1))
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F)), (0, 0), 3)
    tex_w = np.clip(tex / (np.percentile(tex, 70) + 1e-6), 0, 1)
    w = (w * (0.4 + 0.6 * tex_w)).astype(np.float32)
    return np.clip(1.5 * (cv2.GaussianBlur(w, (0, 0), 2.0) if soften else w), 0, 1)


def big_parts(mask: np.ndarray, min_extent: float, by_height: bool = False) -> np.ndarray:
    """The connected parts of a binary mask at least min_extent px across (or tall)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    size = (st[1:, cv2.CC_STAT_HEIGHT] if by_height
            else np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]))
    keep[1:] = size >= min_extent
    return keep[lab]


def guard(solid: np.ndarray, reach: float) -> np.ndarray:
    """0 on and near solid pixels, rising to 1 at `reach` px: a pixel there that moves by up to
    its own distance never samples (drags) the solid object."""
    if not solid.any():
        return np.ones(solid.shape, np.float32)
    dist = cv2.distanceTransform((~solid).astype(np.uint8), cv2.DIST_L2, 5)
    x = np.clip((dist - 1.0) / max(reach, 1.0), 0, 1)
    return (x * x * (3 - 2 * x)).astype(np.float32)


def crop_box(feather: np.ndarray, margin: int):
    """(y0, y1, x0, x1) around the mask's non-zero pixels plus margin, or None when empty."""
    ys, xs = np.nonzero(feather > 0)
    if len(ys) == 0:
        return None
    H, W = feather.shape
    return (max(0, ys.min() - margin), min(H, ys.max() + 1 + margin),
            max(0, xs.min() - margin), min(W, xs.max() + 1 + margin))


def touching(parts: np.ndarray, edge: np.ndarray, within: float) -> np.ndarray:
    """The connected parts of `parts` that come within `within` px of `edge`."""
    if not parts.any() or not edge.any():
        return np.zeros_like(parts) if not edge.any() else parts
    near = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5) <= within
    n, lab = cv2.connectedComponents(parts.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[near & parts])] = True
    keep[0] = False
    return keep[lab]
