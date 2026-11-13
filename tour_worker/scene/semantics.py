"""What every splat of a scene is (bed, sofa, window, fireplace, countertop, vase ...): Mask2Former's
classes voted onto the Gaussians from rendered survey views. Made once per scene (the scene_label
job, at upload) and cached like the motion file; the 3D planner reads it to know what a room holds
(planner3d), and the motion file comes from the same labels (curtains, plants, lit fireplaces).

  views    per room, a few open spots on its floor x VIEW_HEADINGS headings, eye height, the
           horizon a little below centre - every surface of a room is seen from two sides
  segment  each view through Mask2Former (the platform's one segmentation model, vision/segment.py):
           a class per pixel
  vote     every Gaussian projected into every view; "seen" when its centre is visible there (a
           depth buffer of the opaque Gaussians), a "hit" for the class under it. Labelled with its
           top class when hits >= MIN_HITS and hits >= MIN_SHARE x seen (torch: on the GPU when the
           renderer is)
  objects  a class's labelled Gaussians grouped into objects (connected OBJECT_VOXEL cells, at least
           MIN_OBJECT Gaussians): a bed, a sofa, each window - with their box, footprint and room

The file (.npz) is bound to its scene like the motion file: scene_sha256 and the splat count must
match or it is not used.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

FORMAT_VERSION = 1
# Mask2Former ADE20K classes kept (as id2label spells them, stripped). Everything else is "other".
VOCAB = (
    # pieces a shot composes on
    "bed", "sofa", "armchair", "table", "coffee table", "desk", "kitchen island", "counter", "countertop",
    "bathtub", "pool table", "chest of drawers", "buffet", "ottoman", "bench", "chair", "swivel chair", "stool",
    # features a view looks at
    "windowpane", "door", "screen door", "fireplace", "television receiver", "painting", "poster", "mirror",
    "bookcase", "shelf", "wardrobe", "cabinet", "refrigerator", "oven", "stove", "sink", "dishwasher",
    "microwave", "hood", "stairs", "stairway", "step", "bannister", "railing", "column", "curtain", "blind",
    "chandelier", "lamp", "sconce", "light", "fan", "rug",
    # details
    "vase", "flower", "plant", "sculpture", "clock", "cushion", "pillow", "book", "bottle", "tray", "plate",
    "glass", "towel", "basket", "pot", "box", "toilet", "shower", "computer", "monitor", "radiator",
    # surroundings
    "wall", "floor", "ceiling", "sky", "tree", "palm", "grass", "water", "swimming pool", "house", "building",
    "fence",
)
INDEX = {n: k + 1 for k, n in enumerate(VOCAB)}           # 0: unlabelled
VIEW_HEADINGS = 8
VIEW_HFOV, VIEW_W, VIEW_H = 90.0, 1280, 720
VIEW_PITCH = 10.0                                         # degrees below level
SPOTS_PER_ROOM, SPOT_MIN_GAP = 2, 1.2
OPAQUE = 0.5
ZBUF_DOWN = 4
DEPTH_TOL = (0.03, 0.04)
MIN_HITS, MIN_SHARE = 2, 0.4
OBJECT_VOXEL, MIN_OBJECT = 0.05, 150
SKIP_OBJECTS = {"wall", "floor", "ceiling", "sky", "tree", "palm", "grass", "house", "building", "fence"}


@dataclass
class SceneLabels:
    labels: np.ndarray            # (N,) uint8: 0 unlabelled, else VOCAB index + 1
    scene_sha256: str
    splats: int
    objects: list = field(default_factory=list)   # dicts (find_objects)
    vocab: tuple = VOCAB

    def name(self, k: int) -> str | None:
        return self.vocab[k - 1] if k > 0 else None

    def mask(self, *names: str) -> np.ndarray:
        ids = [k + 1 for k, n in enumerate(self.vocab) if n in names]
        return np.isin(self.labels, ids)

    def save(self, path: str) -> None:
        np.savez_compressed(path, labels=self.labels.astype(np.uint8),
                            meta=np.array([FORMAT_VERSION, self.splats], np.int64),
                            scene_sha256=np.array(self.scene_sha256), vocab=np.array(list(self.vocab)),
                            objects=np.array(json.dumps(self.objects)))


class LabelsInvalid(ValueError):
    pass


def load_labels(path: str) -> SceneLabels:
    try:
        z = np.load(path, allow_pickle=False)
        version, splats = (int(v) for v in z["meta"])
        lab = SceneLabels(z["labels"].astype(np.uint8), str(z["scene_sha256"]), splats,
                          json.loads(str(z["objects"])), tuple(str(v) for v in z["vocab"]))
    except Exception as e:                         # noqa: BLE001 - any read problem is a bad file
        raise LabelsInvalid(f"labels file unreadable ({type(e).__name__}: {e})") from None
    if version != FORMAT_VERSION or lab.labels.shape != (splats,) or lab.labels.max(initial=0) > len(lab.vocab):
        raise LabelsInvalid("labels file malformed")
    return lab


def check_binding(lab: SceneLabels, scene_sha256: str, splats: int) -> str | None:
    if lab.scene_sha256 != scene_sha256:
        return "the labels file was made for another scene"
    if lab.splats != splats:
        return f"the labels file expects {lab.splats} Gaussians, the scene has {splats}"
    return None


# ---- views ---------------------------------------------------------------------------------------
@dataclass
class SurveyView:
    room: str
    eye: np.ndarray               # planner axes (z above the floor)
    target: np.ndarray


def survey_views(space, rooms: list[str], eye_h: float = 1.5) -> list[SurveyView]:
    """Per room, up to SPOTS_PER_ROOM open spots at least SPOT_MIN_GAP apart (the most open first),
    VIEW_HEADINGS headings each, looking VIEW_PITCH below level."""
    out = []
    drop = math.tan(math.radians(VIEW_PITCH))
    for room in rooms:
        if room not in space.room_keys:
            continue
        mask = space.space_mask(room, 0.25) & (space.room_of == space.room_keys.index(room))
        pts = space.spread_points(mask, n=SPOTS_PER_ROOM, min_gap=SPOT_MIN_GAP)
        for p in pts:
            for k in range(VIEW_HEADINGS):
                a = 2 * math.pi * k / VIEW_HEADINGS
                d = np.array([math.cos(a), math.sin(a)])
                out.append(SurveyView(room, np.array([p[0], p[1], eye_h]),
                                      np.array([p[0] + 2.5 * d[0], p[1] + 2.5 * d[1], eye_h - 2.5 * drop])))
    return out


def class_map(segmenter, bgr: np.ndarray) -> np.ndarray:
    """Mask2Former's panoptic map -> (h, w) uint8 of INDEX classes (0: anything not kept)."""
    _, seg, info = segmenter.panoptic(bgr)
    out = np.zeros(seg.shape, np.uint8)
    id2label = segmenter.id2label
    for s in info:
        name = str(id2label.get(s["label_id"], id2label.get(str(s["label_id"]), ""))).strip()
        k = INDEX.get(name)
        if k:
            out[seg == s["id"]] = k
    return out


# ---- the vote ------------------------------------------------------------------------------------
class Voter:
    """Accumulates seen / per-class hits over views (torch; on `device`)."""

    def __init__(self, means: np.ndarray, opacities: np.ndarray, device: str = "cpu"):
        import torch
        self.torch, self.device = torch, device
        self.pts = torch.from_numpy(means.astype(np.float32)).to(device)
        self.opaque = torch.from_numpy(opacities >= OPAQUE).to(device)
        n = len(means)
        self.seen = torch.zeros(n, dtype=torch.int16, device=device)
        self.hits = torch.zeros((n, len(VOCAB) + 1), dtype=torch.int16, device=device)

    def add(self, viewmat: np.ndarray, K: np.ndarray, width: int, height: int, classes: np.ndarray) -> None:
        torch = self.torch
        R = torch.from_numpy(viewmat[:3, :3].astype(np.float32)).to(self.device)
        t = torch.from_numpy(viewmat[:3, 3].astype(np.float32)).to(self.device)
        cam = self.pts @ R.T + t
        z = cam[:, 2]
        front = z > 0.05
        zs = torch.where(front, z, torch.ones_like(z))
        u = K[0, 0] * cam[:, 0] / zs + K[0, 2]
        v = K[1, 1] * cam[:, 1] / zs + K[1, 2]
        inside = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        zw, zh = max(1, width // ZBUF_DOWN), max(1, height // ZBUF_DOWN)
        cell = (v * zh / height).long().clamp(0, zh - 1) * zw + (u * zw / width).long().clamp(0, zw - 1)
        zbuf = torch.full((zh * zw,), float("inf"), device=self.device)
        occ = inside & self.opaque
        zbuf.scatter_reduce_(0, cell[occ], z[occ], reduce="amin")
        visible = inside & (z <= zbuf[cell] * (1 + DEPTH_TOL[1]) + DEPTH_TOL[0])
        idx = torch.nonzero(visible).squeeze(1)
        self.seen[idx] += 1
        cm = torch.from_numpy(classes).to(self.device)
        mh, mw = classes.shape
        mu = (u[idx] * mw / width).long().clamp(0, mw - 1)
        mv = (v[idx] * mh / height).long().clamp(0, mh - 1)
        c = cm[mv, mu].long()
        keep = c > 0
        self.hits[idx[keep], c[keep]] += 1

    def labels(self) -> np.ndarray:
        torch = self.torch
        h = self.hits.clone()
        h[:, 0] = 0
        best = torch.argmax(h, dim=1)
        top = h.gather(1, best[:, None]).squeeze(1).to(torch.int32)
        ok = (top >= MIN_HITS) & (top.float() >= MIN_SHARE * self.seen.clamp(min=1).float())
        return torch.where(ok, best, torch.zeros_like(best)).to(torch.uint8).cpu().numpy()

    @property
    def seen_np(self) -> np.ndarray:
        return self.seen.cpu().numpy()


# ---- objects -------------------------------------------------------------------------------------
def find_objects(local: np.ndarray, labels: np.ndarray, space=None) -> list[dict]:
    """Each kept class's labelled Gaussians as objects (planner axes, z above the floor): class,
    centre, box, footprint (a rotated rectangle), top, splat count and room. Stable order."""
    import cv2
    out = []
    for name in VOCAB:
        if name in SKIP_OBJECTS:
            continue
        idx = np.flatnonzero(labels == INDEX[name])
        if len(idx) < MIN_OBJECT:
            continue
        p = local[idx].astype(np.float64)
        cells = np.floor(p / OBJECT_VOXEL).astype(np.int64)
        lo = cells.min(0)
        grid = np.zeros(tuple(cells.max(0) - lo + 1), bool)
        grid[tuple((cells - lo).T)] = True
        comp, k = ndi.label(grid, structure=np.ones((3, 3, 3), bool))
        cid = comp[tuple((cells - lo).T)]
        for j in range(1, k + 1):
            m = cid == j
            if m.sum() < MIN_OBJECT:
                continue
            q = p[m]
            (cx, cy), (w, h), ang = cv2.minAreaRect(q[:, :2].astype(np.float32))
            if w < h:
                w, h, ang = h, w, ang + 90
            centre = np.median(q, 0)
            out.append({"class": name, "centre": [round(float(x), 3) for x in centre],
                        "min": [round(float(x), 3) for x in np.percentile(q, 2, 0)],
                        "max": [round(float(x), 3) for x in np.percentile(q, 98, 0)],
                        "footprint": {"centre": [round(float(cx), 3), round(float(cy), 3)], "long": round(float(w), 3),
                                      "short": round(float(h), 3), "axis_deg": round(float(ang) % 180, 1)},
                        "top": round(float(np.percentile(q[:, 2], 95)), 3), "splats": int(m.sum()),
                        "room": space.room_near(centre[:2]) if space is not None else None})
    out.sort(key=lambda o: (o["class"], o["centre"][0], o["centre"][1], o["centre"][2]))
    return out


def label_scene(model, space, renderer, segmenter, rooms: list[str], scene_sha256: str,
                progress=None, device: str | None = None) -> tuple[SceneLabels, dict, np.ndarray]:
    """Survey, segment, vote: the scene's labels and objects, stats, and how many views saw each
    Gaussian (the motion step grows labels into the unseen ones). renderer: render/splat_shot.make_renderer's
    (its render_view(viewmat, K, w, h) -> BGR uint8); progress(done, total) per view."""
    from tour_worker.scene.camera import intrinsics, look_at
    views = survey_views(space, rooms)
    dev = device or str(getattr(getattr(renderer, "means", None), "device", "cpu"))
    voter = Voter(model.means, model.opacities, dev)
    K = intrinsics(VIEW_HFOV, VIEW_W, VIEW_H)
    up = space.up_scene
    for k, v in enumerate(views):
        V = look_at(space.to_scene(v.eye), space.to_scene(v.target), up=up)
        bgr = render_view(renderer, V, K, VIEW_W, VIEW_H)
        voter.add(V, K, VIEW_W, VIEW_H, class_map(segmenter, bgr))
        if progress:
            progress(k + 1, len(views))
    labels = voter.labels()
    objects = find_objects(space.local, labels, space)
    lab = SceneLabels(labels, scene_sha256, len(model), objects)
    counts = {VOCAB[c - 1]: int(n) for c, n in zip(*np.unique(labels[labels > 0], return_counts=True))}
    stats = {"views": len(views), "rooms": len({v.room for v in views}),
             "labelled": int((labels > 0).sum()), "seen": int((voter.seen_np > 0).sum()),
             "classes": dict(sorted(counts.items(), key=lambda t: -t[1])), "objects": len(objects)}
    return lab, stats, voter.seen_np


def render_view(renderer, viewmat, K, w: int, h: int) -> np.ndarray:
    """A BGR uint8 frame from either renderer (GPU gsplat or the CPU reference)."""
    if hasattr(renderer, "render_bgr8"):
        from tour_worker.render.splat_shot import GPU_LOCK
        with GPU_LOCK:
            return renderer.render_bgr8(viewmat, K, w, h, w, h)
    img = renderer.render(viewmat, K, w, h)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1].copy()
