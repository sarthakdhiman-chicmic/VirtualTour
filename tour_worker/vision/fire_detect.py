"""Lit fires in fireplaces (2026-10-16: lit fireplaces only, product decision).

Detection runs in the tour_segment job, inside the fireplaces OneFormer (ADE20K) finds: flame
pixels are saturated orange-yellow and clearly brighter than their own surroundings (so a warm
stone or brick surround, which is also inside the fireplace mask, does not count). Colour alone
was tried first and failed both ways (pale gas flames missed; a petrified-wood table, warm
pendants and stone chimneys taken for fire): the fireplace mask is what makes it reliable.

The effect (tour_worker/motion/region.py, "fire_flicker") moves the photo's own flames: upward
turbulence, anchored at the base (logs, grate), flames flickering in brightness, and a soft warm
glow around the fire pulsing on the hearth and nearby surfaces.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_FLAME_AREA = 0.0002     # of the photo
FLAME_HUE = (3, 34)         # OpenCV hue: red-orange .. yellow
FLAME_SAT = 0.35
FLAME_VAL = 0.5
LOCAL_LIFT = 0.12           # brighter than the median of its own 0.06-photo-width neighbourhood
# per flame patch (measured on the test set 2026-10-16): real flames sit on dark logs / soot and are
# saturated; warm stone chimneys and fire-lit brick, also inside a fireplace mask, are neither
# (flames: 8-29% very dark pixels around, saturation >= 0.47; stone / brick: 0-6%, 0.40-0.50)
PATCH_DARK_AROUND = 0.07
PATCH_SAT = 0.45
# 2026-10-21 (review of 11 "lit" fireplaces from the S3 photos: 5 were not lit - an orange throw on
# the table in front, an orange marble surround, the hearth floor, unlit logs twice). A flame is
# light: its brightest part is near white (PATCH_CORE_V at the patch's 95th percentile); sunlit
# logs, wood and marble stay below. And a fire burns in the FIREBOX - the dark cavity of the
# fireplace - not on its surround, hearth or what stands in front: most of a patch must lie
# inside the firebox (FIREBOX_SHARE).
PATCH_CORE_V = 0.85
CAVITY_V = 0.32             # firebox: darker than this ...
CAVITY_MIN_SHARE = 0.04     # ... in parts at least this share of the fireplace mask
FIREBOX_SHARE = 0.5
# firelight on the opening's lintel / brass trim reads as flame colour: a FLAT patch (height under
# REFLECT_FLAT x its width) in the upper REFLECT_TOP of the firebox is a reflection (a linear
# fireplace's row of tongues is flat too, but burns low in the box); patches farther than
# GROUP_REACH x the main flame's size from it are glints elsewhere on the fireplace
REFLECT_FLAT = 0.2
REFLECT_TOP = 0.42
GROUP_REACH = 0.5


def firebox(bgr: np.ndarray, fireplace: np.ndarray) -> np.ndarray:
    """The fireplace's dark cavity (bool, bgr's size): the dark parts of the fireplace mask that
    are a fair share of it, each closed over by its convex hull (the flames, logs and glowing
    insert inside it are light). Empty when the mask has no dark cavity."""
    out = np.zeros(bgr.shape[:2], bool)
    if not fireplace.any():
        return out
    ys, xs = np.nonzero(fireplace)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    v = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255
    dark = ((v < CAVITY_V) & fireplace[y0:y1, x0:x1]).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    area = max(1, int(fireplace.sum()))
    box = np.zeros(dark.shape, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < CAVITY_MIN_SHARE * area:
            continue
        pts = cv2.findNonZero((lab == i).astype(np.uint8))
        cv2.fillConvexPoly(box, cv2.convexHull(pts), 1)
    out[y0:y1, x0:x1] = box > 0
    return out


def flame_mask(bgr: np.ndarray, fireplace: np.ndarray, cavity: np.ndarray | None = None) -> np.ndarray:
    """0/255 mask of flame pixels inside a fireplace mask (same size as bgr). cavity: the firebox
    (firebox()); when given, patches must lie mostly inside it."""
    if not fireplace.any():
        return np.zeros(bgr.shape[:2], np.uint8)
    ys, xs = np.nonzero(fireplace)
    pad = 8
    y0, y1 = max(0, ys.min() - pad), min(bgr.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(bgr.shape[1], xs.max() + 1 + pad)
    crop = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    k = max(15, int(0.06 * bgr.shape[1]) | 1)
    local = cv2.medianBlur((v * 255).astype(np.uint8), min(k, 255) | 1).astype(np.float32) / 255
    flame = ((h >= FLAME_HUE[0]) & (h <= FLAME_HUE[1]) & (s > FLAME_SAT) & (v > FLAME_VAL)
             & (v - local > LOCAL_LIFT))
    inside = cv2.dilate(fireplace[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    flame &= inside
    flame = cv2.morphologyEx(flame.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dark = v < 0.2
    kr = max(9, int(0.007 * bgr.shape[1]) | 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(flame, connectivity=8)
    keep = np.zeros(n, bool)
    core = np.zeros(n, bool)
    for i in range(1, n):
        m = lab == i
        ring = cv2.dilate(m.astype(np.uint8), np.ones((kr, kr), np.uint8)).astype(bool) & ~m
        # with a firebox known, it decides "inside the fire's cavity"; the dark-ring test (its
        # older stand-in) failed fires behind glass, where the glass lifts the blacks (Downing's
        # gas fires, 2026-10-21)
        in_box = (ring.any() and dark[ring].mean() >= PATCH_DARK_AROUND) if cavity is None \
            else cavity[y0:y1, x0:x1][m].mean() >= FIREBOX_SHARE
        keep[i] = in_box and s[m].mean() >= PATCH_SAT
        core[i] = float(np.percentile(v[m], 95)) >= PATCH_CORE_V
        if cavity is not None and keep[i]:
            cy = np.nonzero(cavity[y0:y1, x0:x1].any(axis=1))[0]
            if len(cy):
                x, y, w, hh = st[i, :4]
                rel = (y + hh / 2 - cy.min()) / max(1, cy.max() - cy.min())
                if hh < REFLECT_FLAT * w and rel < REFLECT_TOP:
                    keep[i] = False                        # firelight on the lintel / trim
    # a fire is lit when one of its patches glows (a near-white core); its dimmer tongues count
    # with it. No glowing patch: unlit logs, wood, marble - nothing is kept.
    if not (keep & core).any():
        keep[:] = False
    else:
        # the fire: its glowing patches and, chained outward, every kept patch within
        # GROUP_REACH x a member's size (a row of separate tongues stays one fire); anything
        # farther is a glint elsewhere on the fireplace
        member = keep & core
        changed = True
        while changed:
            changed = False
            for i in np.nonzero(keep & ~member)[0]:
                x, y, w, hh = st[i, :4]
                for m_ in np.nonzero(member)[0]:
                    mx, my, mw, mh = st[m_, :4]
                    gap = max(0, mx - (x + w), x - (mx + mw), my - (y + hh), y - (my + mh))
                    if gap <= GROUP_REACH * max(mw, mh):
                        member[i] = changed = True
                        break
        keep = member
    out = np.zeros(bgr.shape[:2], np.uint8)
    out[y0:y1, x0:x1] = keep[lab].astype(np.uint8) * 255
    return out


def flame_zone(mask: np.ndarray) -> np.ndarray:
    """The area the flame tongues occupy, closed into one shape (0/255). Tongues fill under 15% of
    their box, which the region loader rejects as a mismatched mask, and the turbulence should
    move through the gaps between tongues too."""
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return m * 255
    ys, xs = np.nonzero(m)
    size = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    k = max(5, int(0.25 * size) | 1)
    z = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    z = cv2.dilate(z, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(z)
    if n > 2:                                              # one fire: keep its biggest part
        z = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return z * 255


def fire_regions(bgr: np.ndarray, regions: list[dict]) -> list[dict]:
    """Replace "fireplace" regions by "fire" regions where a lit fire is inside (none if unlit).
    bgr is the image at the regions' `space`. Returns the new region list."""
    out, n = [], 0
    H, W = bgr.shape[:2]
    for r in regions:
        if r["class_name"] != "fireplace":
            out.append(r)
            continue
        fm = flame_mask(bgr, r["mask"] > 0, firebox(bgr, r["mask"] > 0))
        if (fm > 0).sum() < MIN_FLAME_AREA * H * W:
            continue                                       # unlit (or no visible flames)
        fm = flame_zone(fm)
        ys, xs = np.nonzero(fm)
        out.append({"object_id": f"fire_{n:03d}", "class_name": "fire",
                    "confidence": r["confidence"],
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    "mask": fm, "space": r["space"]})
        n += 1
    return out


# COCO (no fireplace class, review 2026-10-19): flames are only looked for where COCO says the spot
# is a wall / stone / brick / tile surface, an oven-like box, or nothing it knows - a firebox. A
# flame-coloured patch on a table, lamp, couch, plant, chair ... is furniture, never a fire.
FIRE_SURFACES = ("wall", "oven", "fireplace")
FIRE_GROUP = 0.03          # flame patches closer than 3% of the photo width are one fire


def fire_surface(seg: np.ndarray, segments_info: list[dict], id2label: dict) -> np.ndarray:
    """Where a fire could be (bool): unlabelled pixels and surfaces in FIRE_SURFACES."""
    known, ok = [], []
    for s in segments_info:
        label = id2label.get(s["label_id"], id2label.get(str(s["label_id"]), ""))
        known.append(s["id"])
        if label.startswith(FIRE_SURFACES):
            ok.append(s["id"])
    return np.isin(seg, ok) | ~np.isin(seg, known)


def find_fires(bgr: np.ndarray, seg: np.ndarray, segments_info: list[dict], id2label: dict) -> list[dict]:
    """Lit fires anywhere in the photo (COCO path): "fire" regions, one per fireplace."""
    H, W = bgr.shape[:2]
    fm = flame_mask(bgr, fire_surface(seg, segments_info, id2label)) > 0
    if fm.sum() < MIN_FLAME_AREA * H * W:
        return []
    k = max(3, int(FIRE_GROUP * W) | 1)
    n, lab = cv2.connectedComponents(cv2.dilate(fm.astype(np.uint8), np.ones((k, k), np.uint8)))
    out = []
    for i in range(1, n):
        part = fm & (lab == i)
        if part.sum() < MIN_FLAME_AREA * H * W:
            continue
        z = flame_zone(part.astype(np.uint8) * 255)
        ys, xs = np.nonzero(z)
        out.append({"object_id": f"fire_{len(out):03d}", "class_name": "fire", "confidence": 1.0,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    "mask": z, "space": [W, H]})
    return out

