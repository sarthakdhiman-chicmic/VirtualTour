"""fire_flicker: lit fires in fireplaces (detected by tour_worker.vision.fire_detect in the
tour_segment job). The effect moves the photo's own flames: upward turbulence, anchored at the
base (logs, grate), flames flickering in brightness, and a soft warm glow around the fire pulsing on
the hearth and nearby surfaces. The flames move only inside the fire's own box (fire_box), and
structure inside it is held still (fire_structure).

The region's feather is the glow (1 on the flames, fading out around them); r.flame (set by the
loader) is the flames alone. Every effect function takes the Region as `r`."""
from __future__ import annotations

import math

import cv2
import numpy as np

from tour_worker.motion.masks import sample_bilinear
from tour_worker.vision.fire_detect import CAVITY_V, FLAME_HUE, FLAME_SAT, FLAME_VAL, flame_zone

FIRE_GLOW = 0.08            # the warm light's flicker on the hearth and walls around a fire
FIRE_PLUME = 0.45           # flames may stretch above their tips by this share of their height
# 2026-10-21 (user: "too much flame motion will warp the mantles"): the flames move only inside the
# fire's own box - the dark cavity connected to them, at most FIREBOX_SIDE of their width beside
# them and FIRE_PLUME of their height above - and their strength follows their own size
FIREBOX_SIDE = 0.25
STRUCT_EDGE = 0.25          # gradient (x 255) above which an edge is hard: structure, not flame
STRUCT_FAINT_EDGE = 0.06    # a faint edge counts as structure when it runs straight for ...
STRUCT_LONG = 0.35          # ... this share of the fire box's width (or height)
STRUCT_LIT_V = 0.28         # in a fire's box, non-flame pixels at least this bright are held ...
STRUCT_ABOVE = 0.15         # ... above the fire (rows above its top + this share of its height)
FIRE_AMP_OF_HEIGHT = 0.15   # worker cap: peak flame displacement <= this x the flames' height


def flame_px(r) -> float:
    return max(3.0, 0.18 * getattr(r, "flame_h", r.bh))


def feature_px(r) -> float:
    return flame_px(r)                                # flame tongues


def fields(r, u, v, t, irregular=None):
    """Fire (reviews 2026-10-21): the v1 licking tongues read best; "only one kind of motion"
    was the fault. Layers: tongues licking up and travelling up the flames; the fire surging
    (growing / shrinking) over a second or two; a slight lean with the draft; a fast small
    flutter at the flame edges; brightness flicker rising through the flames, a quick
    sparkle, the fire's pulse, and a soft glow on its surroundings. Structure inside the
    fire's box (walls, glass frame, grate, glass reflections) is held still (tour_worker.motion.loader)."""
    fw = sample_bilinear(r.flame, u, v)
    plume = getattr(r, "plume", None)
    pw = sample_bilinear(plume, u, v) if plume is not None else fw
    fh, fb = r.flame_h, r.flame_bottom
    s = np.clip((fb - v) / max(fh, 1.0), 0.0, 1.0) ** 0.8          # still at the base (logs)
    lam = flame_px(r)
    rise = 0.9 * fh * t                                             # tongues travel upward
    A = r.amp
    surge = 0.75 + 0.35 * float(r.noise_b(1.7, 0.2, 0.6 * t))    # the fire grows / shrinks
    lean = 0.15 * math.sin(2 * math.pi * 0.31 * t + r.phase)     # the draft
    n1 = r.noise_a(u / lam, (v + rise) / lam, 1.3 * t)
    n2 = r.noise_b(u / lam + 7.0, (v + rise) / lam, 1.7 * t)
    ff = 0.5 * lam                                                  # fast flutter, finer
    n3 = r.noise_a(u / ff + 11.0, (v + 1.4 * rise) / ff, 4.0 * t)
    n4 = r.noise_b(u / ff + 23.0, (v + 1.4 * rise) / ff, 4.0 * t)
    dx = A * (0.4 * n1 + lean + 0.15 * n4) * s * pw
    dy = -A * (0.5 * surge * (1.0 + n2) + 0.2 * n3) * s * pw         # lick up, surge, flutter
    whole = float(r.noise_b(0.3, 0.7, 2.4 * t))                  # the fire's overall flicker
    local = r.noise_a(u / lam + 3.0, (v + 1.6 * rise) / lam, 3.0 * t)
    sparkle = r.noise_b(u / ff + 5.0, (v + 2.0 * rise) / ff, 6.0 * t)
    b = (1.0 + fw * (0.28 * local + 0.10 * whole + 0.08 * sparkle + 0.08 * (surge - 0.75))
         + (1.0 - fw) * FIRE_GLOW * whole)
    return dx, dy, b


def flame_plume(flame: np.ndarray, height: float) -> np.ndarray:
    """The flames smeared upward by `height` px, fading with height: where the tongues may lick
    above the tips the photo caught (0..1)."""
    out = (flame > 0).astype(np.float32)
    base = out.copy()
    n = 12
    for i in range(1, n + 1):
        sh = int(round(i * height / n))
        if sh <= 0 or sh >= base.shape[0]:
            continue
        up = np.zeros_like(base)
        up[:-sh] = base[sh:]
        out = np.maximum(out, (1.0 - i / (n + 1)) * up)
    return cv2.GaussianBlur(out, (0, 0), 2.0).clip(0, 1).astype(np.float32)


def fire_box(src: np.ndarray, flame: np.ndarray) -> np.ndarray:
    """Where a fire may move (0..1): the flames plus the dark cavity connected to them, within a
    box FIREBOX_SIDE of the flames' width either side and FIRE_PLUME of their height above (down
    to their base). The mantel, surround and hearth stay outside."""
    f = flame > 0
    ys, xs = np.nonzero(f)
    H, W = f.shape
    fh, fw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    y0, y1 = max(0, int(ys.min() - FIRE_PLUME * fh)), min(H, int(ys.max()) + 2)
    x0, x1 = max(0, int(xs.min() - FIREBOX_SIDE * fw)), min(W, int(xs.max() + FIREBOX_SIDE * fw) + 1)
    v = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255
    cand = ((v < CAVITY_V) | f[y0:y1, x0:x1]).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[f[y0:y1, x0:x1]])] = True
    keep[0] = False
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = keep[lab]
    return out


def fire_structure(src: np.ndarray, flame: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Inside a fire's box, what is not fire (bool): everything that is neither flame nor dark
    empty space - firebox walls and back panels (herringbone, brick), logs, glass reflections,
    the frame - plus hard straight edges even when dark (a grate, andirons, frame bars).
    Review 2026-10-21: side views warped the walls; behind glass the reflections slid with the
    flames. The flames themselves (and a few px around them) never count."""
    out = np.zeros(flame.shape, bool)
    ys, xs = np.nonzero(box > 0.5)
    if not len(ys):
        return out
    pad = 6
    y0, y1 = max(0, ys.min() - pad), min(flame.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(flame.shape[1], xs.max() + 1 + pad)
    crop = src[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat, val = hsv[..., 1] / 255, hsv[..., 2] / 255
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    edge = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    hard = edge > STRUCT_EDGE * 255
    # long straight-ish runs only: walls, frames, bars (flames make short broken edges)
    lines = np.zeros_like(hard, np.uint8)
    k = max(9, int(0.15 * max(1, min(y1 - y0, x1 - x0))))
    for kern in (np.ones((1, k), np.uint8), np.ones((k, 1), np.uint8)):
        lines |= cv2.morphologyEx(hard.astype(np.uint8), cv2.MORPH_OPEN, kern)
    # faint but LONG straight runs too: a screen rod or wire, a door frame's bar - dark on the
    # dark cavity, so low contrast (review 2026-10-21: a wire across a brick fireplace's
    # opening warped). Flames never make long straight lines, so length is the test.
    # Only ABOVE the fire: among the logs, long faint edges are the logs themselves.
    faint = (edge > STRUCT_FAINT_EDGE * 255).astype(np.uint8)
    kh = max(15, int(STRUCT_LONG * (x1 - x0)))
    kv = max(15, int(STRUCT_LONG * (y1 - y0)))
    faint_lines = (cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((1, kh), np.uint8))
                   | cv2.morphologyEx(faint, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8)))
    # only flames and dark empty space may move: anything lit that is not flame - a back wall's
    # herringbone or brick, logs, glass reflections (grey, often dim), the frame - is held
    lit = val >= STRUCT_LIT_V
    # the flame PIXELS (flame-coloured and bright inside the fire's zone - the zone itself is
    # closed over the gaps between tongues, where the back wall shows through)
    hue = hsv[..., 0]
    tongues = ((flame[y0:y1, x0:x1] > 0) & (hue >= FLAME_HUE[0]) & (hue <= FLAME_HUE[1])
               & (sat > FLAME_SAT) & (val > FLAME_VAL))
    near_flame = cv2.dilate(tongues.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    # lit non-flame pixels are held only ABOVE the fire (the back wall, glass reflections over
    # the flames): logs and what shows between the tongues may move with the fire (2026-10-21:
    # holding them too left some fires only flickering in light)
    fz = flame[y0:y1, x0:x1] > 0
    fy = np.nonzero(fz.any(axis=1))[0]
    above = np.zeros_like(lit)
    if len(fy):
        above[: max(0, int(fy.min() + STRUCT_ABOVE * (fy.max() - fy.min() + 1)))] = True
    st = ((lines > 0) | ((lit | (faint_lines > 0)) & above & ~fz)) & ~near_flame
    st = cv2.dilate(st.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out[y0:y1, x0:x1] = st & (box[y0:y1, x0:x1] > 0.5)
    return out


def fire_glow(flame: np.ndarray) -> np.ndarray:
    """1 on the flames, a soft warm halo around them (about the fire's own size), 0 beyond."""
    ys, xs = np.nonzero(flame > 0)
    size = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    sigma = max(4.0, 0.6 * size)
    halo = cv2.GaussianBlur((flame > 0).astype(np.float32), (0, 0), sigma)
    halo = np.clip(halo / max(float(halo.max()), 1e-6) * 2.0, 0, 1)
    halo[halo < 0.01] = 0.0                    # no faint tail across the whole photo
    return np.maximum(flame, halo).astype(np.float32)


def fire_zone_file(path: str, src_w: int, src_h: int):
    """Flame tongues -> their filled zone (tour_worker/vision/fire_detect.flame_zone), written next to
    the mask; returns (zone path, its bbox normalised). Masks from before the zone existed work."""
    m = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (src_w, src_h), interpolation=cv2.INTER_NEAREST)
    z = flame_zone(m)
    zp = path[:-4] + ".zone.png"
    cv2.imwrite(zp, z)
    ys, xs = np.nonzero(z)
    if not len(ys):
        return path, (0.0, 0.0, 1.0, 1.0)
    return zp, (xs.min() / src_w, ys.min() / src_h, (xs.max() + 1) / src_w, (ys.max() + 1) / src_h)
