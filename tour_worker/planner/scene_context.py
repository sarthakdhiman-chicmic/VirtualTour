"""What a photo shows, for the planner (2026-10-19, user: "the planner needs to know all the right
input from the image context itself" - no per-tour rules).

Built from the OneFormer scene summary the worker's tour_segment job returns for every photo
(every class: area share, box, centre, in photo-normalised coordinates) plus the saliency signals
of scoring/motion.py. Classes are sorted into FEATURES - the things a viewer looks at (a door, a
chandelier, a fireplace, an island, a bed, a pool) - and SURFACES - walls, floors, ceilings,
ground, sky - that carry no interest of their own. From the features:

  interest_x / interest_y  where the interest is (weighted centre; the camera aims there)
  top_y / bottom_y         the vertical span of the interest (a move ends at the interest's
                           height, never sinking onto a floor or paving below it)
  richness 0..1            how much there is to see (distinct features, how spread out): a hero
                           shot gets its long screen time only when the photo has the content
  h_extent / v_extent      how far the interest spreads across / up the frame (tracking and
                           panning moves suit wide spreads; pedestal / tilt moves tall ones)
  dominance 0..1           share of the interest in the single strongest feature (a push-in
                           earns its place only on one clear subject)
  tall_subject             a tall, narrow feature (a door, a staircase, a tall window): a
                           pedestal rise reads well
  high_interest            features up high (pendants, chandeliers, beams-level windows)
  open_floor               visible floor / ground share: depth in front of the camera, which an
                           orbiting gimbal move needs to feel right

Class names differ between the COCO and ADE20K OneFormers ("door-stuff" / "door",
"light" / "chandelier"); they are matched by keyword, and an unknown class counts as a
mid-weight thing unless it looks like a surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# keyword -> interest weight (the first keyword contained in the class name wins)
FEATURE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("fireplace", 1.0), ("fire", 1.0), ("chandelier", 1.0), ("pendant", 1.0), ("light", 0.9),
    ("lamp", 0.7), ("door", 1.0), ("stair", 1.0), ("window", 0.8), ("blind", 0.5),
    ("swimming", 1.0), ("pool", 1.0), ("water", 0.9), ("river", 0.8), ("sea", 0.8),
    ("lake", 0.8), ("house", 1.0), ("building", 0.9), ("roof", 0.6), ("bed", 1.0),
    ("couch", 0.9), ("sofa", 0.9), ("dining", 0.9), ("table", 0.8), ("counter", 0.9),
    ("island", 0.9), ("kitchen", 0.9), ("oven", 0.7), ("stove", 0.7), ("range", 0.7),
    ("sink", 0.6), ("bathtub", 0.9), ("tub", 0.9), ("shower", 0.7), ("toilet", 0.3),
    ("mirror", 0.6), ("tv", 0.4), ("television", 0.4), ("cabinet", 0.5), ("shelf", 0.4),
    ("chair", 0.5), ("bench", 0.5), ("armchair", 0.6), ("curtain", 0.6), ("plant", 0.6),
    ("flower", 0.6), ("vase", 0.5), ("tree", 0.4), ("umbrella", 0.5), ("fence", 0.15),
    ("railing", 0.2), ("book", 0.3), ("painting", 0.6), ("picture", 0.6), ("pillow", 0.3),
    ("cushion", 0.3), ("rug", 0.0), ("carpet", 0.0), ("person", 0.0),
)
SURFACE_KEYS = (
    "wall", "floor", "ceiling", "ground", "pavement", "gravel", "grass", "dirt", "road",
    "sky", "sand", "earth", "field", "path", "sidewalk", "mountain", "hill", "platform",
    "stone", "brick", "tile", "wood", "cloth", "paper", "mud", "snow",
)
FLOOR_KEYS = ("floor", "rug", "carpet", "ground", "pavement", "gravel", "grass", "road",
              "sidewalk", "path", "dirt", "sand")
UNKNOWN_THING_WEIGHT = 0.4
MIN_AREA = 0.002          # smaller classes are noise (a few pixels of "bottle")
AREA_CAP = 0.15           # a huge feature is not 100x as interesting as a small one
BIG_FEATURE_BOX = 0.01


def class_weight(name: str) -> float:
    n = name.lower()
    for key, w in FEATURE_WEIGHTS:
        if key in n:
            return w
    if any(k in n for k in SURFACE_KEYS):
        return 0.0
    return UNKNOWN_THING_WEIGHT


@dataclass
class Feature:
    name: str
    weight: float           # class weight x sqrt(area) (area capped)
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]


@dataclass
class SceneContext:
    interest_x: float = 0.5
    interest_y: float = 0.5
    top_y: float = 0.3
    bottom_y: float = 0.7
    richness: float = 0.5
    h_extent: float = 0.4
    v_extent: float = 0.3
    dominance: float = 0.3
    tall_subject: bool = False
    high_interest: bool = False
    open_floor: float = 0.0
    semantic: bool = False          # built from a scene summary (else saliency only)
    features: tuple[Feature, ...] = ()

    @property
    def rises_to_interest(self) -> bool:
        """The camera should come up to the interest: a tall subject (a door, a staircase), or
        the interest sitting above a foreground of floor / steps / paving (an entry walk-up)."""
        return self.tall_subject or (self.open_floor >= 0.6 and self.interest_y <= 0.55)

    def interest_grid(self, rows: int, cols: int) -> list[list[float]]:
        """The features painted onto a rows x cols grid (cell = weight-density of the features
        covering it), max cell 1: blended with the saliency grid to judge camera windows."""
        g = [[0.0] * cols for _ in range(rows)]
        for f in self.features:
            x0, y0, x1, y1 = f.bbox
            area = max(1e-4, (x1 - x0) * (y1 - y0))
            dens = f.weight / math.sqrt(area)
            for i in range(rows):
                cy = (i + 0.5) / rows
                if not y0 <= cy <= y1:
                    continue
                for j in range(cols):
                    if x0 <= (j + 0.5) / cols <= x1:
                        g[i][j] += dens
        m = max((v for r in g for v in r), default=0.0)
        return [[v / m for v in r] for r in g] if m > 0 else g


def _wpct(vals: list[float], ws: list[float], q: float) -> float:
    pairs = sorted(zip(vals, ws, strict=True))
    tot = sum(ws)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * tot:
            return v
    return pairs[-1][0]


def from_scene(summary: dict | None, peak_x: float = 0.5, peak_y: float = 0.5,
               spread: float = 0.4) -> SceneContext:
    """SceneContext from a tour_segment scene summary ({"classes": {name: {area, count, bbox,
    centroid}}}); saliency values only (peak, spread) when there is no summary."""
    classes = (summary or {}).get("classes") or {}
    feats, floor = [], 0.0
    for name, c in classes.items():
        area = float(c.get("area") or 0.0)
        if any(k in name.lower() for k in FLOOR_KEYS):
            floor += area
        w = class_weight(name)
        if w <= 0 or area < MIN_AREA or not c.get("bbox"):
            continue
        bx = tuple(float(v) for v in c["bbox"])
        mid = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
        cen = tuple(float(v) for v in (c.get("centroid") or mid))
        feats.append(Feature(name, w * math.sqrt(min(area, AREA_CAP)), bx, cen))
    if not feats:
        return SceneContext(interest_x=peak_x, interest_y=peak_y,
                            richness=max(0.0, min(1.0, (spread - 0.35) / 0.12)),
                            h_extent=spread, open_floor=min(1.0, floor / 0.25))
    ws = [f.weight for f in feats]
    tot = sum(ws)
    xs = [f.centroid[0] for f in feats]
    ys = [f.centroid[1] for f in feats]
    ix = sum(w * x for w, x in zip(ws, xs, strict=True)) / tot
    iy = sum(w * y for w, y in zip(ws, ys, strict=True)) / tot
    # the saliency peak keeps a say: a feature list misses what OneFormer has no class for
    ix, iy = 0.75 * ix + 0.25 * peak_x, 0.75 * iy + 0.25 * peak_y
    top = _wpct([f.bbox[1] for f in feats], ws, 0.2)
    bottom = _wpct([f.bbox[3] for f in feats], ws, 0.8)
    # distinct notable features of a size a viewer takes in (box >= 1% of the photo): a plain
    # facade has 2, a staged kitchen 7, the richest rooms 9-10 (Turkey mix, 2026-10-19)
    big = {f.name for f in feats if class_weight(f.name) >= 0.5
           and (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) >= BIG_FEATURE_BOX}
    h_ext = _wpct(xs, ws, 0.9) - _wpct(xs, ws, 0.1)
    v_ext = _wpct(ys, ws, 0.9) - _wpct(ys, ws, 0.1)
    richness = max(0.0, min(1.0, 0.7 * (len(big) - 2) / 7 + 0.3 * min(1.0, h_ext / 0.6)))
    tall = any(
        class_weight(f.name) >= 0.8 and (f.bbox[3] - f.bbox[1]) >= 0.35
        and (f.bbox[3] - f.bbox[1]) >= 1.3 * (f.bbox[2] - f.bbox[0])
        for f in feats
    )
    high = any(class_weight(f.name) >= 0.7 and f.centroid[1] < 0.3 for f in feats)
    feats.sort(key=lambda f: -f.weight)
    return SceneContext(
        interest_x=round(ix, 4), interest_y=round(iy, 4), top_y=round(top, 4),
        bottom_y=round(bottom, 4), richness=round(richness, 4), h_extent=round(h_ext, 4),
        v_extent=round(v_ext, 4), dominance=round(max(ws) / tot, 4), tall_subject=tall,
        high_interest=high, open_floor=round(min(1.0, floor / 0.25), 4), semantic=True,
        features=tuple(feats),
    )


def blend_grid(
    saliency: list[list[float]], ctx: SceneContext, w_sem: float = 0.5
) -> list[list[float]]:
    """Saliency grid blended with the features' grid (same shape), max cell 1."""
    if not saliency or not ctx.semantic:
        return saliency
    rows, cols = len(saliency), len(saliency[0])
    sem = ctx.interest_grid(rows, cols)
    g = [[(1 - w_sem) * saliency[i][j] + w_sem * sem[i][j] for j in range(cols)]
         for i in range(rows)]
    m = max(v for r in g for v in r) or 1.0
    return [[round(v / m, 4) for v in r] for r in g]
