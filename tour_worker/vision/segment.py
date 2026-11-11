"""Mask2Former (ADE20K panoptic) object masks for perceptual motion.

Replaced OneFormer on 2026-11-11 (client review: one panoptic model, one licence chain; the M2F
benchmark: one pass instead of two, ~8.7x faster on CPU, better foliage / water / plane masks).
Model `facebook/mask2former-swin-large-ade-panoptic`, pinned below. Its Hugging Face card says
licence "other" (the Mask2Former code is MIT, Meta); it is trained on ADE20K, whose images are for
non-commercial research / education (MIT CSAIL terms): the same legal question as before.

One pass gives every motion class, fireplaces included. Two raw masks are refined
(tour_worker.vision.refine): pools / water keep the liquid surface only (coping and basin walls
stay still), and a fireplace returns a "fire" region with its flame core only when it is lit.

Output per photo: the same shape as a detect result, so the planner (tour_worker/planner) treats both alike:
  {object_id, class_name, confidence, bbox: [x1, y1, x2, y2], mask: 0/255 uint8, space: [w, h]}
bbox and mask are in `space` pixels: the photo scaled to at most MAX_SIDE on its long side.
"Stuff" classes (curtain, tree, water) come back as one merged region per class; they are split
into connected parts so each curtain panel or tree moves on its own.
"""
from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np

# Label (Mask2Former ADE20K id2label, exactly as spelled there) -> our class name (the planner's
# perceptual_effects keys). "fireplace" is not a motion class itself: its lit flames are.
MODELS = {
    "m2f_ade20k": {
        "id": "facebook/mask2former-swin-large-ade-panoptic",
        "revision": "ae53ed7ab37f8ea74f23d0e4f8c3d83c184a0caa",
        "weights": "model.safetensors",
        "sha256": "959905d66e497b8246308f00c4a72dd9e59d3f41e048c2bd0a0b0d5677dfc249",
        "dir": "mask2former_swin_large_ade_panoptic",
        "class_map": {"curtain": "curtain", "blind": "blinds", "plant": "potted_plant",
                      "tree": "tree", "palm": "tree", "water": "water", "sea": "water",
                      "river": "water", "lake": "water", "swimming pool": "water",
                      "fireplace": "fireplace", "sky": "sky"},
    },
}
DEFAULT_MODEL = "m2f_ade20k"


def model_id(model: str = DEFAULT_MODEL) -> str:
    """"<hf id>@<revision[:10]>", as reported in tour_segment results."""
    m = MODELS[model]
    return f"{m['id']}@{m['revision'][:10]}"


MAX_SIDE = 2048            # long side the masks are produced at (memory; the planner reads `space`)
MIN_PART_AREA = 0.003      # parts smaller than 0.3% of the photo are dropped
MIN_SCORE = 0.5


def ensure_model(model_dir: str, model: str = DEFAULT_MODEL) -> str:
    """Download the pinned model if missing and check the weights' hash."""
    spec = MODELS[model]
    weights = os.path.join(model_dir, spec["weights"])
    if not os.path.exists(weights):
        from huggingface_hub import snapshot_download
        snapshot_download(spec["id"], revision=spec["revision"], local_dir=model_dir,
                          allow_patterns=["*.json", spec["weights"]])
    h = hashlib.sha256()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != spec["sha256"]:
        raise RuntimeError("segmentation weights checksum mismatch")
    return model_dir


class Segmenter:
    """Loads Mask2Former once; call it with a BGR image. model: a MODELS key; model_dir: its folder
    (Settings.segment_path())."""

    def __init__(self, model_dir: str, model: str = DEFAULT_MODEL, threads: int | None = None):
        import torch
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
        if threads:
            torch.set_num_threads(threads)
        self.model_name, self.class_map = model, MODELS[model]["class_map"]
        d = ensure_model(model_dir, model)
        self.torch = torch
        self.processor = Mask2FormerImageProcessor.from_pretrained(d)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(d).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    def __call__(self, bgr: np.ndarray) -> list[dict]:
        return self.segment(bgr)[0]

    def panoptic(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """(the photo at mask size, the panoptic id map, segments_info) - one Mask2Former pass."""
        h, w = bgr.shape[:2]
        k = min(1.0, MAX_SIDE / max(w, h))
        if k < 1.0:
            bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        sh, sw = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_panoptic_segmentation(out, target_sizes=[(sh, sw)])[0]
        return bgr, res["segmentation"].cpu().numpy(), res["segments_info"]

    def segment(self, bgr: np.ndarray) -> tuple[list[dict], dict]:
        """(motion regions, scene summary) - one pass, then the water and fireplace refinements."""
        bgr, seg, info = self.panoptic(bgr)
        sh, sw = bgr.shape[:2]
        regions = segments_to_regions(seg, info, self.id2label, (sw, sh), self.class_map)
        scene = scene_summary(seg, info, self.id2label, (sw, sh))
        regions = outdoor_plants(regions, scene)
        return refine_regions(bgr, regions), scene


def refine_regions(bgr: np.ndarray, regions: list[dict]) -> list[dict]:
    """Water regions keep their liquid surface only (dropped when none is left); each fireplace
    becomes a "fire" region holding its flame core when it is lit, and is dropped when it is not
    (tour_worker.vision.refine). Other regions pass through."""
    from tour_worker.vision.refine import refine_fireplace_and_flames, refine_water_surface
    out, fires = [], 0
    min_px = MIN_PART_AREA * bgr.shape[0] * bgr.shape[1]
    for r in regions:
        if r["class_name"] == "water":
            surface, _ = refine_water_surface(bgr, r["mask"])
            if int((surface > 0).sum()) < min_px:
                continue
            out.append(dict(r, mask=surface, bbox=_bbox(surface)))
        elif r["class_name"] == "fireplace":
            flame, _, _, lit = refine_fireplace_and_flames(bgr, r["mask"])
            if lit:
                out.append(dict(r, object_id=f"fire_{fires:03d}", class_name="fire", mask=flame,
                                bbox=_bbox(flame)))
                fires += 1
        else:
            out.append(r)
    return out


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def outdoor_plants(regions: list[dict], scene: dict) -> list[dict]:
    """ADE20K's "plant" covers potted plants and shrubs alike. A photo showing sky (1% or more)
    is outdoors: its plants are shrubs and hedges, which stay still (user decision 2026-10-16), so
    they are renamed "plant_outdoor" (no effect) rather than dropped."""
    sky = sum(c["area"] for k, c in scene.get("classes", {}).items() if k.startswith("sky"))
    if sky < 0.01:
        return regions
    return [dict(r, class_name="plant_outdoor") if r["class_name"] == "potted_plant" else r for r in regions]


def segments_to_regions(seg: np.ndarray, segments_info: list[dict], id2label: dict,
                        space: tuple[int, int], class_map: dict | None = None) -> list[dict]:
    """Panoptic map -> our regions: mapped classes only, stuff split into connected parts, small
    parts dropped, ordered by area (largest first) for stable object ids. class_map: the model's
    (MODELS[...]["class_map"]); default the default model's."""
    sw, sh = space
    min_px = MIN_PART_AREA * sw * sh
    parts = []
    sky = None                       # the sky is ONE region (all its pieces: a balcony's openings)
    for s in segments_info:
        label = id2label.get(s["label_id"], id2label.get(str(s["label_id"]), ""))
        cls = (class_map or MODELS[DEFAULT_MODEL]["class_map"]).get(label)
        score = float(s.get("score", 1.0))
        if not cls or score < MIN_SCORE:
            continue
        m = (seg == s["id"]).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if cls == "sky":
            sky = (m, score) if sky is None else (np.maximum(sky[0], m), max(sky[1], score))
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < min_px:
                continue
            parts.append((int(area), cls, score, [int(x), int(y), int(x + bw), int(y + bh)],
                          ((lab == i) * 255).astype(np.uint8)))
    if sky is not None and int(sky[0].sum()) >= SKY_MIN_AREA * sw * sh:
        ys, xs = np.nonzero(sky[0])
        parts.append((int(sky[0].sum()), "sky", sky[1],
                      [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], sky[0] * 255))
    parts.sort(key=lambda p: (-p[0], p[1], p[3]))
    return [{"object_id": f"seg_{i:03d}", "class_name": cls, "confidence": round(score, 4), "bbox": bbox,
             "mask": mask, "space": [sw, sh]} for i, (_, cls, score, bbox, mask) in enumerate(parts)]


SCENE_MIN_AREA = 0.002     # classes covering less than 0.2% of the photo are left out
SKY_MIN_AREA = 0.01        # a sky under 1% of the photo is no region (nothing to see drift)


def scene_summary(seg: np.ndarray, segments_info: list[dict], id2label: dict,
                  space: tuple[int, int]) -> dict:
    """What is in the photo, for choosing its camera move (bed, sofa, kitchen island, window,
    stairs, ceiling, sky, water ...): every Mask2Former class with its share of the photo, the box
    around all its parts and their area-weighted centre, all normalised to the photo (0..1)."""
    sw, sh = space
    acc: dict[str, dict] = {}
    for s in segments_info:
        if float(s.get("score", 1.0)) < MIN_SCORE:
            continue
        label = id2label.get(s["label_id"], id2label.get(str(s["label_id"]), ""))
        ys, xs = np.nonzero(seg == s["id"])
        if not len(ys):
            continue
        a = acc.setdefault(label, {"n": 0, "px": 0, "sx": 0.0, "sy": 0.0,
                                   "box": [sw, sh, 0, 0]})
        a["n"] += 1
        a["px"] += len(ys)
        a["sx"] += float(xs.sum())
        a["sy"] += float(ys.sum())
        b = a["box"]
        a["box"] = [min(b[0], int(xs.min())), min(b[1], int(ys.min())),
                    max(b[2], int(xs.max()) + 1), max(b[3], int(ys.max()) + 1)]
    total = float(sw * sh)
    classes = {}
    for label, a in sorted(acc.items()):
        if a["px"] / total < SCENE_MIN_AREA:
            continue
        b = a["box"]
        classes[label] = {"area": round(a["px"] / total, 4), "count": a["n"],
                          "bbox": [round(b[0] / sw, 4), round(b[1] / sh, 4), round(b[2] / sw, 4),
                                   round(b[3] / sh, 4)],
                          "centroid": [round(a["sx"] / a["px"] / sw, 4), round(a["sy"] / a["px"] / sh, 4)]}
    return {"space": [sw, sh], "classes": classes}

