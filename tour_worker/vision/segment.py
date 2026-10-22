"""OneFormer (COCO panoptic) object masks for perceptual motion.

Replaces the RunPod detect model as the tour's mask source (2026-10-15): one model gives curtains,
window blinds, potted plants, trees and water. Model `shi-labs/oneformer_coco_swin_large`, MIT,
pinned below; trained on COCO panoptic (annotations CC BY 4.0, images under their Flickr licences,
COCO disclaims them: the dataset question needs a legal sign-off before production use).

Output per photo: the same shape as a detect result, so the backend planner treats both alike:
  {object_id, class_name, confidence, bbox: [x1, y1, x2, y2], mask: 0/255 uint8, space: [w, h]}
bbox and mask are in `space` pixels: the photo scaled to at most MAX_SIDE on its long side.
COCO "stuff" classes (curtain, tree, water) come back as one merged region per class; they are split
into connected parts so each curtain panel or tree moves on its own.
"""
from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np

# Two pinned OneFormer checkpoints (MIT weights). ADE20K (2026-10-16) knows fireplaces (lit-fire
# effect), pools and blinds; its dataset images are for non-commercial research / education
# (MIT CSAIL terms) - the product owner accepted that, pending legal (like COCO's Flickr images).
# Label -> our class name (the planner's perceptual_effects keys); "fireplace" is not a motion
# class itself: lit fires are looked for inside it (tour_worker.vision.fire_detect).
MODELS = {
    "coco": {
        "id": "shi-labs/oneformer_coco_swin_large",
        "revision": "3a263017ca5c75adbea145f25f81b118243d4394",
        "sha256": "36d01ceb971e9b3824bf35de022cd3148aad0c928f6365302db492a4cff6bf2b",
        "dir": "oneformer_coco_swin_large",
        "class_map": {"curtain": "curtain", "window-blind": "blinds", "potted plant": "potted_plant",
                      "tree-merged": "tree", "sea": "water", "river": "water", "water-other": "water"},
    },
    "ade20k": {
        "id": "shi-labs/oneformer_ade20k_swin_large",
        "revision": "4a5bac8e64f82681a12db2e151a4c2f4ce6092b2",
        "sha256": "c0b2fe11dfecee6f2f1f315f466946e96f4e94813f3f6d660ff3747b83c28cc9",
        "dir": "oneformer_ade20k_swin_large",
        "class_map": {"curtain": "curtain", "blind, screen": "blinds", "plant": "potted_plant",
                      "tree": "tree", "palm, palm tree": "tree", "water": "water", "sea": "water",
                      "river": "water", "lake": "water", "pool": "water", "fireplace": "fireplace"},
    },
}
# COCO is primary (review 2026-10-19: its water masks are the surface only, submerged steps
# included; ADE20K's "pool" took in coping and walls). Which one runs is the SEGMENT_MODEL setting
# (tour_worker.config). Lit fires with COCO (no fireplace class): colour + label rules found 4/11
# on the known set (missed Downing's gas fires behind glass; took kitchen pendants and a sconce
# for fire) - off (COCO_FIRE_SEARCH) until the fireplace adapter (training/) exists.
DEFAULT_MODEL = "coco"


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
    weights = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(weights):
        from huggingface_hub import snapshot_download
        snapshot_download(spec["id"], revision=spec["revision"], local_dir=model_dir,
                          allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "README.md"])
    h = hashlib.sha256()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != spec["sha256"]:
        raise RuntimeError("oneformer weights checksum mismatch")
    return model_dir


class Segmenter:
    """Loads OneFormer once; call it with a BGR image. model: a MODELS key; model_dir: its folder
    (Settings.oneformer_path()); fire_search: the COCO_FIRE_SEARCH setting."""

    def __init__(self, model_dir: str, model: str = DEFAULT_MODEL, threads: int | None = None,
                 fire_search: bool = False):
        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
        if threads:
            torch.set_num_threads(threads)
        self.model_name, self.class_map, self.fire_search = model, MODELS[model]["class_map"], fire_search
        d = ensure_model(model_dir, model)
        self.torch = torch
        self.processor = OneFormerProcessor.from_pretrained(d)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(d).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    def __call__(self, bgr: np.ndarray) -> list[dict]:
        return self.segment(bgr)[0]

    def segment(self, bgr: np.ndarray) -> tuple[list[dict], dict]:
        """(motion regions, scene summary) - one OneFormer pass."""
        h, w = bgr.shape[:2]
        k = min(1.0, MAX_SIDE / max(w, h))
        if k < 1.0:
            bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        sh, sw = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, task_inputs=["panoptic"], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_panoptic_segmentation(out, target_sizes=[(sh, sw)])[0]
        seg = res["segmentation"].cpu().numpy()
        from tour_worker.vision.fire_detect import find_fires, fire_regions
        regions = segments_to_regions(seg, res["segments_info"], self.id2label, (sw, sh), self.class_map)
        scene = scene_summary(seg, res["segments_info"], self.id2label, (sw, sh))
        regions = outdoor_plants(regions, scene, self.model_name)
        if "fireplace" in self.class_map:              # ADE20K: flames inside its fireplace masks
            return fire_regions(bgr, regions), scene
        if self.fire_search:                          # off: failed validation (4/11, 2026-10-19)
            regions = regions + find_fires(bgr, seg, res["segments_info"], self.id2label)
        return regions, scene


    def fires(self, bgr: np.ndarray) -> list[dict]:
        """The fireplace pass (FIREPLACE_MODEL, 2026-10-22): this model's fireplaces, and inside
        each the lit fire if there is one (tour_worker.vision.fire_detect.fire_regions: the
        firebox, a glowing core, reflections and glints rejected). Only "fire" regions return -
        the rest of this model's classes are the primary model's job."""
        h, w = bgr.shape[:2]
        k = min(1.0, MAX_SIDE / max(w, h))
        if k < 1.0:
            bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        sh, sw = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, task_inputs=["panoptic"], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_panoptic_segmentation(out, target_sizes=[(sh, sw)])[0]
        seg = res["segmentation"].cpu().numpy()
        from tour_worker.vision.fire_detect import fire_regions
        fireplaces = segments_to_regions(seg, res["segments_info"], self.id2label, (sw, sh),
                                         {lab: c for lab, c in self.class_map.items() if c == "fireplace"})
        return [r for r in fire_regions(bgr, fireplaces) if r["class_name"] == "fire"]


def outdoor_plants(regions: list[dict], scene: dict, model: str = DEFAULT_MODEL) -> list[dict]:
    """ADE20K's "plant" covers potted plants and shrubs alike. A photo showing sky (1% or more)
    is outdoors: its plants are shrubs and hedges, which stay still (user decision 2026-10-16), so
    they are renamed "plant_outdoor" (no effect) rather than dropped."""
    if model != "ade20k":                              # COCO's "potted plant" is pots only
        return regions
    sky = sum(c["area"] for k, c in scene.get("classes", {}).items() if k.startswith("sky"))
    if sky < 0.01:
        return regions
    return [dict(r, class_name="plant_outdoor") if r["class_name"] == "potted_plant" else r for r in regions]


def segments_to_regions(seg: np.ndarray, segments_info: list[dict], id2label: dict,
                        space: tuple[int, int], class_map: dict | None = None) -> list[dict]:
    """Panoptic map -> our regions: mapped classes only, stuff split into connected parts, small
    parts dropped, ordered by area (largest first) for stable object ids. class_map: the model's
    (MODELS[...]["class_map"]); default COCO's."""
    sw, sh = space
    min_px = MIN_PART_AREA * sw * sh
    parts = []
    for s in segments_info:
        label = id2label.get(s["label_id"], id2label.get(str(s["label_id"]), ""))
        cls = (class_map or MODELS[DEFAULT_MODEL]["class_map"]).get(label)
        score = float(s.get("score", 1.0))
        if not cls or score < MIN_SCORE:
            continue
        m = (seg == s["id"]).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < min_px:
                continue
            parts.append((int(area), cls, score, [int(x), int(y), int(x + bw), int(y + bh)],
                          ((lab == i) * 255).astype(np.uint8)))
    parts.sort(key=lambda p: (-p[0], p[1], p[3]))
    return [{"object_id": f"seg_{i:03d}", "class_name": cls, "confidence": round(score, 4), "bbox": bbox,
             "mask": mask, "space": [sw, sh]} for i, (_, cls, score, bbox, mask) in enumerate(parts)]


SCENE_MIN_AREA = 0.002     # classes covering less than 0.2% of the photo are left out


def scene_summary(seg: np.ndarray, segments_info: list[dict], id2label: dict,
                  space: tuple[int, int]) -> dict:
    """What is in the photo, for choosing its camera move (bed, couch, dining table, window,
    stairs, ceiling, sky, water ...): every OneFormer class with its share of the photo, the box
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

