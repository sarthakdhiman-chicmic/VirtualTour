"""Rigid-plane depth parallax (client review 2026-11-20, item 10) - prototype.

A photo becomes a stack of rigid planar layers, so a small camera move shows near things moving
against far ones while every straight structure stays straight:

  depth     Depth Anything V2 Small's relative inverse depth -> metric-like depth, the room assumed
            to span NEAR_M..FAR_M (the network gives no scale); the photo's field of view HFOV_DEG
  planes    structure (OneFormer: wall, floor, ceiling, window, door, cabinet, counter, mirror,
            blind) is split into planes by RANSAC on its back-projected points; every object
            (sofa, table, chair, ...) is one rigid plane fitted to its own points; pixels with no
            label join the plane nearest in depth. A plane seen from a moved camera is an exact
            homography: its lines stay straight, it never bends
  layers    planes ordered far to near; each is drawn with its own homography
  fill      what a nearer layer uncovers is the layer behind it, edge-extended (its nearest own
            pixel, no invented detail) into the part of the frame nearer layers covered
  travel    capped so the largest disparity between near and far stays within CAP_PX at the
            working size (the uncovered band never wider than the fill)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

HFOV_DEG = 75.0
NEAR_M, FAR_M = 1.5, 8.0
STRUCTURE = {"wall", "floor", "ceiling", "windowpane", "window", "door", "cabinet", "counter", "mirror",
             "blind", "shelf", "stairs", "sky", "building", "house", "roof", "rug"}
ONE_SURFACE = {"floor", "ceiling", "rug"}       # one plane over all their regions
# what may hold several planes (a room's walls meet at corners; a window wall can turn); a floor, a
# ceiling, a rug or an object is ONE flat plane (depth noise split the floor in two)
MULTI_PLANE = {"wall", "windowpane", "window", "cabinet", "building"}
MIN_PLANE_PX = 400              # (at the segmentation size) a smaller region joins its neighbour
RANSAC_ITERS = 150
PLANE_TOL = 0.04                # inlier distance, as a share of the point's depth
MAX_PLANES_PER_REGION = 3
CAP_PX = 8.0                    # the most a near layer may move against the far one (working px)
FILL_PX = 24                    # edge extension band behind nearer layers
EDGE_R, EDGE_EPS = 4, 1e-3      # guided-filter radius (px) and smoothing that snap layer edges to the photo
FRINGE = 0.02                   # a layer weight above this: the photo's own pixel, not the fill
COVERED = 0.98                  # nearer layers this opaque hide the fill band completely
OWN_T = 0.15                    # a layer with this share of a pixel (soft mask) owns it, if nearest ...
DEPTH_GATE = 0.15               # ... and the pixel's depth is within this share of the layer's there
MERGE_DEPTH = 0.08              # touching objects this close in depth are one layer ...
MERGE_SIZE = 0.2                # ... when the smaller is at most this share of the larger (an item on furniture)
CARVE_FROM = {"wall", "door"}          # (not windows: the view through them has real depth of its own)
CARVE_NEARER = 0.25             # a blob this much nearer than its wall / door / window plane is an object in front
CARVE_MIN_PX = 600              # (at 960 px wide) smaller blobs are depth noise
CARVE_WIN = 121                 # px (at 1920 wide): the neighbourhood whose farthest depth is "the wall behind"
CARVE_FILL, CARVE_ASPECT = 0.35, 4.0   # an object fills a third of its box and is no strip
SURE = 0.85                     # a soft share this high is the layer's own: not depth-checked ...
SEE_THROUGH = 0.35              # ... unless (objects) the pixel is this much farther than the object: seen through it
REST_DEPTH = 0.30               # an item resting on furniture joins it within this depth difference
REST_PX, REST_SHARE = 8, 0.5    # resting: half its bottom edge within 8 px above the furniture ...
CONTACT_DEPTH = 0.12            # ... and at the same depth there (within this share)
AA_SIGMA = 0.0                  # px: the owner's edge anti-alias (0: none - the bilinear warp smooths a moving edge)
EDGE_BAND = 3                   # px (at 1920 wide): an object's rim, where each pixel goes to the side it looks like ...
EDGE_WIN = 9                    # ... judged against the colours within this many px on either side (2026-11-25: the
                                # chair's mask took the black window frame along as a dark outline)
ONE_PLANE_SHARE = 0.75          # a window / wall region this much on one plane is that one plane (2026-11-25: a
ONE_PLANE_TOL = 2.5             # window in three pieces tore its mullion); tolerance x PLANE_TOL
# soft seams (2026-11-25: one wall / ceiling / tapestry cut into pieces by the segmentation tore along
# the pieces' ragged edges, at every strength): where two layers meet but the PHOTO's depth runs on
# across the seam, both bend to one shared depth there, so the seam cannot open
SEAM_R = 8                      # px (at 1920 wide): the photo's depth compared this far either side ...
SOFT_JUMP = 0.06                # ... and the seam is soft below this relative jump (at SEAM_R and 2 SEAM_R)
SEAM_W = 48                     # px (at 1920 wide): the band over which a layer bends to the seam
SEAM_CELL = 8                   # px (at 1920 wide): the bend is stored on a grid this coarse (it is smooth)


@dataclass
class Plane:
    n: np.ndarray               # unit normal (camera coords), facing the camera
    c: float                    # n . X = c  (c > 0)
    mask: np.ndarray            # bool, working size
    label: str
    depth: float                # median depth, for ordering
    zmap: np.ndarray | None = None   # the plane's depth at every pixel (set by build_planes)

    def homography(self, K: np.ndarray, Kinv: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Source pixel -> pixel seen by a camera translated by t (metres, camera coords, no turn)."""
        return K @ (np.eye(3) - np.outer(t, self.n) / self.c) @ Kinv


def intrinsics(w: int, h: int, hfov_deg: float = HFOV_DEG) -> np.ndarray:
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def metric_depth(inv: np.ndarray, near: float = NEAR_M, far: float = FAR_M) -> np.ndarray:
    """Relative inverse depth (higher = nearer) -> depth in metres-like units, robustly normalised."""
    lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
    d = np.clip((inv - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return 1.0 / (1.0 / far + d * (1.0 / near - 1.0 / far))


def _fit_plane(P: np.ndarray, rng: np.random.Generator):
    """RANSAC plane n.X = c on points P (N, 3); (n, c, inliers) or None."""
    if len(P) < 30:
        return None
    best, best_in = None, None
    tol = PLANE_TOL * np.linalg.norm(P, axis=1)
    for _ in range(RANSAC_ITERS):
        a, b, c3 = P[rng.integers(0, len(P), 3)]
        n = np.cross(b - a, c3 - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        c = float(n @ a)
        inl = np.abs(P @ n - c) < tol
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (n, c), inl
    if best is None or best_in.sum() < 30:
        return None
    # refine on the inliers (least squares, SVD)
    Q = P[best_in]
    cen = Q.mean(0)
    n = np.linalg.eigh(np.cov((Q - cen).T))[1][:, 0]      # the least-spread direction (3x3)
    c = float(n @ cen)
    if c < 0:
        n, c = -n, -c
    inl = np.abs(P @ n - c) < tol
    return n, c, inl


def build_planes(img_shape, depth_inv: np.ndarray, seg: np.ndarray, labels: dict[int, str],
                 seed: int = 0) -> tuple[list[Plane], np.ndarray, np.ndarray]:
    """The photo's rigid layers (far to near), K and the plane index map (working size = image)."""
    h, w = img_shape[:2]
    rng = np.random.default_rng(seed)
    z = metric_depth(cv2.resize(depth_inv, (w, h), interpolation=cv2.INTER_LINEAR))
    segw = cv2.resize(seg.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
    K = intrinsics(w, h)
    Kinv = np.linalg.inv(K)
    ys, xs = np.mgrid[0:h, 0:w]
    rays = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    X = rays * z[..., None]
    step = 4                                              # fit on a grid of every 4th pixel
    sub = np.zeros((h, w), bool)
    sub[::step, ::step] = True
    planes: list[Plane] = []
    # one physical surface may come as several regions (OneFormer split p14's floor in two, which then
    # fitted two planes at different angles that sheared against each other - the floor "breathed"):
    # a floor, a ceiling, a rug is fitted ONCE over all its regions
    regions: list[tuple[str, np.ndarray]] = []
    merged: dict[str, np.ndarray] = {}
    for sid, lab in labels.items():
        r = segw == int(sid)
        if lab in ONE_SURFACE:
            merged[lab] = merged[lab] | r if lab in merged else r
        elif lab not in STRUCTURE:
            # an object region in pieces is several objects (OneFormer's one "light" took the ceiling
            # pendants and a floor lamp: the lamp got the pendants' depth - 2026-11-25): each piece
            # its own layer at its own depth
            n_cc, cc = cv2.connectedComponents(r.astype(np.uint8), connectivity=8)
            for k in range(1, n_cc):
                regions.append((lab, cc == k))
        else:
            regions.append((lab, r))
    regions += list(merged.items())
    for lab, region in regions:
        if region.sum() < MIN_PLANE_PX * (w / 960) ** 2:
            continue
        rest = region.copy()
        k_max = MAX_PLANES_PER_REGION if lab in MULTI_PLANE else 1
        if k_max > 1:
            # a region that is mostly one flat surface is fitted once, over all of it: its pieces would
            # move apart (only a corner window / wall really needs two)
            pts_idx = rest & sub
            fit = _fit_plane(X[pts_idx], rng) if pts_idx.sum() >= 60 else None
            if fit is not None:
                n, c, _ = fit
                dist = np.abs((X[pts_idx] * n).sum(-1) - c) / np.maximum(z[pts_idx], 1e-6)
                if (dist < PLANE_TOL * ONE_PLANE_TOL).mean() >= ONE_PLANE_SHARE:
                    planes.append(Plane(n=n, c=c, mask=rest, label=lab, depth=float(np.median(z[rest]))))
                    continue
        for _ in range(k_max):
            pts_idx = rest & sub
            P = X[pts_idx]
            if len(P) < 60:
                break
            fit = _fit_plane(P, rng)
            if fit is None:
                break
            n, c, _ = fit
            if lab in STRUCTURE:
                dist = np.abs((X * n).sum(-1) - c) / np.maximum(z, 1e-6)
                own = rest & (dist < PLANE_TOL * 1.5)
            else:
                # an object moves as one upright card facing the camera at its own depth: it shifts and
                # scales, never shears (a side table's plane, fitted mostly to its top, came out tilted
                # and bent the table's legs at every strength - 2026-11-25)
                own = rest.copy()
                n, c = np.array([0.0, 0.0, 1.0]), float(np.median(z[own]))
            if own.sum() < MIN_PLANE_PX * (w / 960) ** 2:
                break
            planes.append(Plane(n=n, c=c, mask=own, label=lab, depth=float(np.median(z[own]))))
            rest &= ~own
            if rest.sum() < MIN_PLANE_PX * (w / 960) ** 2:
                break
    planes += _carve_unlabelled_objects(planes, X, z, rays, w)
    # every pixel belongs to exactly one plane: an unassigned one (no label, or a sliver) joins the
    # plane of its nearest assigned pixel in the image - never a plane elsewhere that merely has a
    # similar depth (that tore a sofa into wall-moving patches)
    idx = np.full((h, w), -1, np.int32)
    for i, p in enumerate(planes):
        idx[p.mask & (idx < 0)] = i
    if (idx < 0).any() and planes:
        assigned = idx >= 0
        _, lab = cv2.distanceTransformWithLabels((~assigned).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        order_ = np.flatnonzero(assigned.ravel())
        lut = np.zeros(lab.max() + 1, np.int32)
        lut[lab.ravel()[order_]] = idx.ravel()[order_]
        idx = lut[lab]
    for i, p in enumerate(planes):
        p.mask = idx == i
    planes = _merge_touching(planes, z)
    order = sorted(range(len(planes)), key=lambda i: -planes[i].depth)      # far first
    planes = [planes[i] for i in order]
    idx = np.full((h, w), -1, np.int32)
    for i, p in enumerate(planes):
        idx[p.mask] = i
    for p in planes:                     # each plane's own depth at every pixel (ownership is depth-checked)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = rays @ p.n
            p.zmap = np.where(denom > 1e-6, p.c / denom, np.inf).astype(np.float32)
    PHOTO_DEPTH[0] = z.astype(np.float32)
    return planes, K, idx


PHOTO_DEPTH: list = [None]          # the photo's depth (build_planes), for depth-checked ownership


DECOR = {"object", "vase", "bowl", "book", "bottle", "cup", "wine glass", "plant", "potted plant", "clock", "remote",
         "laptop", "keyboard", "mouse", "cell phone", "scissors", "teddy bear", "toaster", "microwave"}
SURFACES = {"table", "counter", "cabinet", "shelf", "desk", "dining table"}


def _kinds(label: str) -> set[str]:
    return set(label.split("+"))


def _carve_unlabelled_objects(planes: list[Plane], X, z, rays, w: int) -> list[Plane]:
    """Things the segmentation missed (p16's white vase against a white wall was labelled "wall" and
    "door", and a wall plane was even fitted through it, so it moved with the wall): inside a wall /
    door / window region, or among pixels no plane took, a blob clearly NEARER than the farthest
    surface around it (CARVE_NEARER below the local far depth, over CARVE_WIN px) is something
    standing in front - cut out as its own object (an upright card at its own depth, "object")."""
    if not planes:
        return []
    taken = np.zeros(z.shape, bool)
    host = np.zeros(z.shape, bool)
    for p in planes:
        taken |= p.mask
        if p.label in CARVE_FROM:
            host |= p.mask
    host |= ~taken
    k = int(CARVE_WIN * w / 1920) | 1
    far = cv2.dilate(np.where(host, z, 0).astype(np.float32), np.ones((k, k), np.uint8))   # local far surface
    nearer = host & (z < (1 - CARVE_NEARER) * far)
    nearer = cv2.morphologyEx(nearer.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(nearer, connectivity=8)
    min_px = CARVE_MIN_PX * (w / 960) ** 2
    out = []
    for i in range(1, n_cc):
        area, bw_, bh_ = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_px:
            continue
        # only an object's shape - compact, not a strip or an arch: a window mullion is nearer than the view
        # through it and a doorway's frame nearer than the room beyond, but both are structure (p14)
        if area < CARVE_FILL * bw_ * bh_ or max(bw_, bh_) > CARVE_ASPECT * min(bw_, bh_):
            continue
        m = cc == i
        d = float(np.median(z[m]))
        for p in planes:                                  # the hosts give the blob up
            p.mask &= ~m
        out.append(Plane(n=np.array([0.0, 0.0, 1.0]), c=d, mask=m, label="object", depth=d))
    return out


def _rests_on(small: np.ndarray, big: np.ndarray, z: np.ndarray | None) -> bool:
    """small stands on big: at least REST_SHARE of small's bottom edge (each column's lowest pixel)
    has big within REST_PX below it, AT THE SAME DEPTH there (within CONTACT_DEPTH). Something hidden
    behind big also seems to "sit" on its top edge in the picture (a console behind a sofa), but it is
    farther away at that line."""
    cols = np.flatnonzero(small.any(0))
    if not len(cols):
        return False
    h = small.shape[0]
    bottom = h - 1 - np.argmax(small[::-1, cols], axis=0)
    hits, zs, zb = 0, [], []
    for c, y in zip(cols, bottom):
        below = np.flatnonzero(big[y:min(h, y + REST_PX + 1), c])
        if len(below):
            hits += 1
            if z is not None:
                zs.append(z[y, c])
                zb.append(z[y + below[0], c])
    if hits < REST_SHARE * len(cols):
        return False
    if z is None:
        return True
    a, b = float(np.median(zs)), float(np.median(zb))
    return abs(a - b) / min(a, b) < CONTACT_DEPTH


def _merge_touching(planes: list[Plane], z: np.ndarray) -> list[Plane]:
    """A small item resting on furniture (a vase on a console, books on a table: its base meets the
    furniture at the same depth) is one layer with it: they move together and cannot tear apart
    (2026-11-25: p16's vase tore off its console). Structure planes are left as they are."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    objs = [p for p in planes if p.label not in STRUCTURE]
    rest = [p for p in planes if p.label in STRUCTURE]
    changed = True
    while changed:
        changed = False
        for a in range(len(objs)):
            grown = cv2.dilate(objs[a].mask.astype(np.uint8), k) > 0
            for b in range(a + 1, len(objs)):
                pa, pb = objs[a], objs[b]
                if not (grown & pb.mask).any():
                    continue
                # only a small item onto what it stands on (a vase on a console): two pieces of furniture
                # touching in the picture are often metres apart (a lamp behind a sofa)
                sa, sb = int(pa.mask.sum()), int(pb.mask.sum())
                if min(sa, sb) > MERGE_SIZE * max(sa, sb):
                    continue
                small, big = (pa, pb) if sa < sb else (pb, pa)
                # ONLY a decor item resting on a surface - by what they are and where the item's base is,
                # not by depth: touching in the picture says little (a vase "touching" a plant reflected in
                # the mirror took its depth), and depth is unreliable for exactly these things (p16's white
                # vase against a mirror read 5-6.5 on a console at 3.8: it slid off it - 2026-11-25)
                if not (_kinds(small.label) <= DECOR and _kinds(big.label) & SURFACES):
                    continue
                if not _rests_on(small.mask, big.mask, None):
                    continue
                m = pa.mask | pb.mask
                d = float(np.median(z[m]))
                objs[a] = Plane(n=np.array([0.0, 0.0, 1.0]), c=d, mask=m, label=f"{pa.label}+{pb.label}", depth=d)
                del objs[b]
                changed = True
                break
            if changed:
                break
    return rest + objs


def _remap(idx, remap):
    out = np.full_like(idx, -1)
    for old, new in remap.items():
        out[idx == old] = new
    return out


def max_parallax_px(planes: list[Plane], K: np.ndarray, t: np.ndarray, idx: np.ndarray) -> float:
    """The largest on-screen shift between layers for translation t: f |t_xy| (1/z_near - 1/z_far),
    measured on the layers' own depths (10th / 90th percentile)."""
    if not planes:
        return 0.0
    zs = [p.depth for p in planes if p.mask.any()]
    f = K[0, 0]
    lat = float(np.hypot(t[0], t[1]))
    return f * lat * (1.0 / min(zs) - 1.0 / max(zs)) + f * abs(float(t[2])) * 0.0


def travel_for(planes, K, idx, direction: np.ndarray, cap_px: float = CAP_PX) -> np.ndarray:
    """The camera translation along `direction` (unit, camera coords) whose parallax is cap_px."""
    d = np.asarray(direction, float)
    d /= max(np.linalg.norm(d), 1e-9)
    one = max_parallax_px(planes, K, d, idx)
    return d * (cap_px / one if one > 1e-9 else 0.0)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter (grey guide, box means): src's edges snapped to the guide's."""
    ksz = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, cv2.CV_32F, ksz)              # noqa: E731
    mi, mp = mean(guide), mean(src)
    cov = mean(guide * src) - mi * mp
    var = mean(guide * guide) - mi * mi
    a = cov / (var + eps)
    b_ = mp - a * mi
    return mean(a) * guide + mean(b_)


def soft_masks(img: np.ndarray, planes: list[Plane]) -> list[np.ndarray]:
    """Each layer's mask refined at full resolution: the segmentation's (half-resolution, stepped)
    boundaries snapped to the photo's own edges (guided filter, EDGE_R px) and feathered; normalised
    so the layers' weights sum to one at every pixel."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    soft = [np.clip(guided_filter(grey, p.mask.astype(np.float32), EDGE_R, EDGE_EPS), 0.0, 1.0) for p in planes]
    total = np.maximum(sum(soft), 1e-6)
    return [m / total for m in soft]


def push_pull_fill(colour: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """colour where weight is 0 filled by the smooth spread of the weighted colours around it
    (push-pull: weighted averages down a pyramid, pulled back up into the holes) - a soft
    continuation of the layer's edge, never invented detail."""
    c = colour.astype(np.float32) * weight[..., None]
    w = weight.astype(np.float32)
    levels = [(c, w)]
    while min(w.shape) > 8:
        c, w = cv2.pyrDown(c), cv2.pyrDown(w)
        levels.append((c, w))
    c, w = levels[-1]
    filled = c / np.maximum(w, 1e-6)[..., None]
    for c, w in reversed(levels[:-1]):
        up = cv2.resize(filled, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR)
        own = c / np.maximum(w, 1e-6)[..., None]
        k = np.clip(w, 0.0, 1.0)[..., None]
        filled = own * k + up * (1 - k)
    return filled


def extended_layers(img: np.ndarray, planes: list[Plane], owner_out: list | None = None):
    """Per layer (far to near): its colours and alpha. Every pixel belongs WHOLLY to the nearest
    layer with a share of it over OWN_T (the edge-snapped soft masks): a moving object takes its whole
    edge with it, and nothing of it is left on the layer behind (2026-11-25: the split edge pixels
    stayed on the wall as a halo). The owner's edge gets a sub-pixel anti-alias (AA_SIGMA). Behind
    nearer layers each layer has FILL_PX of fill - its own colours continued smoothly (push-pull),
    never invented - under what nearer layers cover, so it is never seen at rest."""
    h, w = img.shape[:2]
    soft = soft_masks(img, planes)
    z = PHOTO_DEPTH[0]
    owner = np.full((h, w), -1, np.int32)
    for i, m in enumerate(soft):                             # far to near: a nearer share wins ...
        claim = m > OWN_T
        if z is not None and planes[i].zmap is not None:
            # only on the contested edge (soft share below SURE): a layer's own interior is never
            # dropped for its depth (a lamp's stem has noisy depth)
            # ... only where the pixel's own depth is the layer's: sky between branches, the wall at a
            # vase's edge stay with the layer behind (the client's "separated at depth edges")
            with np.errstate(invalid="ignore"):
                rel = (z - planes[i].zmap) / np.maximum(planes[i].zmap, 1e-6)   # + : the pixel is behind the layer
            sure = m >= SURE
            if planes[i].label not in STRUCTURE:
                # seen through the object (a glass top or panel shows what is behind it): the pixel is
                # the background's, even inside the object (2026-11-25: the glass parallaxed with its console)
                sure &= rel < SEE_THROUGH
            claim &= sure | (np.abs(rel) < DEPTH_GATE)
        owner[claim] = i
    if (owner < 0).any():                                    # no layer fits the depth: the largest share
        best = np.argmax(np.stack(soft), axis=0).astype(np.int32)
        owner = np.where(owner < 0, best, owner)
    owner = _rim_by_colour(img, planes, owner)
    if owner_out is not None:
        owner_out.append(owner)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FILL_PX + 1,) * 2)
    # what is behind each pixel: the farther layer whose own pixels are NEAREST in the image (a hidden
    # pixel behind the sofa continues the rug next to it, not a console that is merely next in depth -
    # 2026-11-25: a console's fill drew a dark crescent behind a sofa's end)
    behind = np.full((h, w), -1, np.int32)
    for o in np.unique(owner):
        if o <= 0:
            continue
        far = owner < o
        if not far.any():
            continue
        _, lab = cv2.distanceTransformWithLabels((~far).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        order_ = np.flatnonzero(far.ravel())
        lut = np.zeros(lab.max() + 1, np.int32)
        lut[lab.ravel()[order_]] = owner.ravel()[order_]
        mine = owner == o
        behind[mine] = lut[lab[mine]]
    out = [None] * len(planes)
    nearer = np.zeros((h, w), bool)
    for i in range(len(planes) - 1, -1, -1):                 # near to far, to know what is nearer
        own = owner == i
        hard = own.astype(np.float32)
        aa = cv2.GaussianBlur(hard, (0, 0), AA_SIGMA) if (i > 0 and AA_SIGMA > 0) else hard
        band = (cv2.dilate(own.astype(np.uint8), kernel) > 0) & (behind == i)
        # opaque on its own pixels and on the hidden fill under nearer layers; over farther layers only
        # its anti-aliased edge
        alpha = np.where(own | (band & nearer), 1.0, np.where(nearer, 0.0, aa))
        fill = push_pull_fill(img, hard)
        if (band & nearer).any():
            fill = mirror_fill(img, own, band & nearer, fill)
        colour = np.where(own[..., None], img.astype(np.float32), fill)
        out[i] = (np.clip(colour, 0, 255).astype(np.uint8), alpha.astype(np.float32))
        nearer |= own
    return out


def render(layers, planes: list[Plane], K: np.ndarray, t: np.ndarray, crop: np.ndarray,
           out_size: tuple[int, int]) -> np.ndarray:
    """One frame: every layer warped by its plane's homography for translation t, then the 2D framing
    `crop` (output px -> working px, a 3x3), composited far to near. One resample per layer."""
    Kinv = np.linalg.inv(K)
    ow, oh = out_size
    acc = np.zeros((oh, ow, 3), np.float32)
    for (colour, alpha), p in zip(layers, planes, strict=True):
        H = p.homography(K, Kinv, t)                         # source -> moved view
        m = np.linalg.inv(H) @ crop                          # output -> source
        c = cv2.warpPerspective(colour, m, (ow, oh), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        a = cv2.warpPerspective(alpha, m, (ow, oh), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
        acc = acc * (1 - a) + c * a
    return np.clip(acc + 0.5, 0, 255).astype(np.uint8)


def parallax_px(planes: list[Plane], K: np.ndarray, t: np.ndarray, samples: int = 300, seed: int = 0) -> float:
    """How far layers move against each other on screen for camera translation t (any direction,
    forward included): each layer's mean displacement over sample pixels of its own; the largest
    difference from the farthest layer's, in working px."""
    if not planes:
        return 0.0
    rng = np.random.default_rng(seed)
    Kinv = np.linalg.inv(K)
    disp = []
    for p in planes:
        ys, xs = np.nonzero(p.mask)
        if not len(xs):
            disp.append(None)
            continue
        k = rng.choice(len(xs), min(samples, len(xs)), replace=False)
        pts = np.stack([xs[k], ys[k], np.ones(len(k))]).astype(float)
        q = p.homography(K, Kinv, t) @ pts
        q = q[:2] / q[2]
        disp.append(np.c_[q[0] - pts[0], q[1] - pts[1]])
    far = next(d for d in disp if d is not None)               # planes are far to near
    ref = far.mean(0)
    return max(float(np.linalg.norm(d - ref, axis=1).mean()) for d in disp if d is not None)


def travel_toward(planes, K, direction: np.ndarray, cap_px: float = CAP_PX) -> np.ndarray:
    """The translation along `direction` (camera coords: x right, y down, z forward) whose parallax
    (parallax_px) is cap_px."""
    d = np.asarray(direction, float)
    d /= max(np.linalg.norm(d), 1e-9)
    one = parallax_px(planes, K, d * 0.01) / 0.01
    return d * (cap_px / one if one > 1e-9 else 0.0)


def _rim_by_colour(img: np.ndarray, planes: list[Plane], owner: np.ndarray) -> np.ndarray:
    """An object's rim (EDGE_BAND px inside its edge) checked pixel by pixel: a pixel that looks like
    what is around the object (the mean colour of the other layers' pixels within EDGE_WIN) more than
    like the object's own inside (its pixels within EDGE_WIN, rim excluded) goes to the layer next to
    it. The segmentation's edge is a pixel or two off; a sliding object must not take the wall's or a
    window frame's pixels with it."""
    h, w = owner.shape
    k = w / 1920
    band_px, win = max(1, round(EDGE_BAND * k)), max(3, round(EDGE_WIN * k)) | 1
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1,) * 2)
    out = owner.copy()
    for i, p in enumerate(planes):
        if p.label in STRUCTURE:
            continue
        own = owner == i
        if not own.any():
            continue
        ys, xs = np.nonzero(own)
        y0, y1 = max(0, ys.min() - win), min(h, ys.max() + win + 1)
        x0, x1 = max(0, xs.min() - win), min(w, xs.max() + win + 1)
        o = own[y0:y1, x0:x1].astype(np.uint8)
        inner = cv2.erode(o, ker) > 0
        rim = (o > 0) & ~inner
        outside = (cv2.dilate(o, ker) == 0)
        if not rim.any() or not outside.any():
            continue
        L_ = lab[y0:y1, x0:x1]
        box = lambda a: cv2.boxFilter(a, cv2.CV_32F, (win, win), normalize=False)   # noqa: E731
        ni, no = box(inner.astype(np.float32)), box(outside.astype(np.float32))
        mi = np.stack([box(L_[..., c] * inner) for c in range(3)], -1) / np.maximum(ni, 1)[..., None]
        mo = np.stack([box(L_[..., c] * outside) for c in range(3)], -1) / np.maximum(no, 1)[..., None]
        di, do = np.linalg.norm(L_ - mi, axis=-1), np.linalg.norm(L_ - mo, axis=-1)
        give = rim & (ni > 0) & (no > 0) & (do < di)
        if not give.any():
            continue
        # to the layer of the nearest pixel outside the object
        sub = owner[y0:y1, x0:x1]
        _, lbl = cv2.distanceTransformWithLabels(o, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        order_ = np.flatnonzero((o == 0).ravel())
        lut = np.zeros(lbl.max() + 1, np.int32)
        lut[lbl.ravel()[order_]] = sub.ravel()[order_]
        sub_out = out[y0:y1, x0:x1]
        sub_out[give] = lut[lbl[give]]
    return out


def mirror_fill(img: np.ndarray, own: np.ndarray, hidden: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    """The hidden fill continued as the layer's own texture mirrored across its edge (a rug's weave, a
    floor's boards keep going) instead of a smooth smear: each hidden pixel takes the pixel as far on the
    other side of its nearest own pixel; where that is not the layer's own, the smooth fill."""
    h, w = own.shape
    out = smooth.copy()
    _, lbl = cv2.distanceTransformWithLabels((~own).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    order_ = np.flatnonzero(own.ravel())
    if not len(order_):
        return out
    lut = np.zeros(lbl.max() + 1, np.int64)
    lut[lbl.ravel()[order_]] = order_
    hy, hx = np.nonzero(hidden)
    near = lut[lbl[hy, hx]]
    ny, nx = near // w, near % w
    my, mx = 2 * ny - hy, 2 * nx - hx
    ok = (my >= 0) & (my < h) & (mx >= 0) & (mx < w)
    ok[ok] &= own[my[ok], mx[ok]]
    out[hy[ok], hx[ok]] = img[my[ok], mx[ok]]
    return out


def plane_rho(p: Plane, Kinv: np.ndarray) -> np.ndarray:
    """The plane's inverse depth along each pixel's ray is affine in the pixel: a . (x, y, 1)."""
    return (Kinv.T @ p.n) / p.c


def seam_bend(planes: list[Plane], owner: np.ndarray, z: np.ndarray, K: np.ndarray):
    """Soft seams: where two layers meet but the photo's depth runs on across the seam (within SOFT_JUMP
    at SEAM_R and 2 SEAM_R px either side), both layers bend to the same inverse depth there - the
    two planes' mean at the seam, spread smoothly along it - fading back to their own plane over SEAM_W.
    A real depth edge (a sofa against the wall) stays a cut.

    Returns (cell px F, per layer the inverse-depth correction on the F-px grid, or None where the layer
    has no soft seam); cell (gy, gx) is centred on pixel ((gx + .5) F - .5, (gy + .5) F - .5)."""
    h, w = owner.shape
    k = w / 1920
    R, F, W = max(2, round(SEAM_R * k)), max(2, round(SEAM_CELL * k)), SEAM_W * k
    gh, gw = -(-h // F), -(-w // F)
    Kinv = np.linalg.inv(K)
    coef = np.array([plane_rho(p, Kinv) for p in planes])
    num, den = np.zeros((gh, gw)), np.zeros((gh, gw))
    seeds = np.zeros((len(planes), gh, gw), bool)
    pairs = []
    jsum, jcnt = np.zeros((gh, gw)), np.zeros((gh, gw))
    for dy, dx in ((0, 1), (1, 0)):
        a, b = owner[:h - dy, :w - dx], owner[dy:, dx:]
        ys, xs = np.nonzero(a != b)
        if not len(ys):
            continue
        jump = np.zeros(len(ys))
        for r in (R, 2 * R):
            za = z[np.clip(ys - r * dy, 0, h - 1), np.clip(xs - r * dx, 0, w - 1)]
            zb = z[np.clip(ys + dy + r * dy, 0, h - 1), np.clip(xs + dx + r * dx, 0, w - 1)]
            jump = np.maximum(jump, np.abs(za - zb) / np.maximum(np.minimum(za, zb), 1e-6))
        np.add.at(jsum, (ys // F, xs // F), np.minimum(jump, 1.0))
        np.add.at(jcnt, (ys // F, xs // F), 1.0)
        pairs.append((dy, dx, ys, xs, a[ys, xs], b[ys, xs]))
    # soft or hard decided over a stretch of seam, not pixel by pixel: a seam whose depth jump hovered at
    # the threshold flipped along its length and waved (p42's mirror frame, 2026-11-25)
    js = cv2.GaussianBlur(jsum, (0, 0), 1.5)
    jc = cv2.GaussianBlur(jcnt, (0, 0), 1.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        soft_cell = np.where(jc > 1e-6, js / jc, 1.0) < SOFT_JUMP
    for dy, dx, ys, xs, ia, ib in pairs:
        soft = soft_cell[ys // F, xs // F]
        ys, xs, ia, ib = ys[soft], xs[soft], ia[soft], ib[soft]
        ra = coef[ia, 0] * xs + coef[ia, 1] * ys + coef[ia, 2]
        rb = coef[ib, 0] * (xs + dx) + coef[ib, 1] * (ys + dy) + coef[ib, 2]
        gy, gx = ys // F, xs // F
        np.add.at(num, (gy, gx), (ra + rb) / 2)
        np.add.at(den, (gy, gx), 1.0)
        seeds[ia, gy, gx] = True
        seeds[ib, (ys + dy) // F, (xs + dx) // F] = True
    out: list[np.ndarray | None] = [None] * len(planes)
    if not den.any():
        return F, out
    # everything is smoothed over sig cells (W / 2): a bend that followed the ragged seam pixel by pixel
    # waved the straight lines beside it (p42's mirror frames, 2026-11-25)
    sig = max(1.0, W / F / 2)
    bn, bd = cv2.GaussianBlur(num, (0, 0), sig), cv2.GaussianBlur(den, (0, 0), sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        target = np.where(bd > 1e-6, bn / bd, 0.0)             # the shared depth at and near soft seams
    gys, gxs = np.mgrid[0:gh, 0:gw]
    cx, cy = (gxs + 0.5) * F - 0.5, (gys + 0.5) * F - 0.5
    line = 1.0 / (math.sqrt(2 * math.pi) * sig)                 # a straight seam's blurred density on it
    for i in range(len(planes)):
        if not seeds[i].any():
            continue
        rho = coef[i, 0] * cx + coef[i, 1] * cy + coef[i, 2]
        sd = seeds[i].astype(np.float64)
        n_, d_ = cv2.GaussianBlur(sd * (target - rho), (0, 0), sig), cv2.GaussianBlur(sd, (0, 0), sig)
        with np.errstate(invalid="ignore", divide="ignore"):
            # the seam's offset carried smoothly off it, full strength on the seam (gaps in it too),
            # fading over ~2 sig
            delta = np.where(d_ > 1e-6, n_ / d_, 0.0) * np.clip(2 * d_ / line, 0.0, 1.0)
        delta = delta.astype(np.float32)
        if np.abs(delta).max() > 0:
            out[i] = delta
    return F, out
