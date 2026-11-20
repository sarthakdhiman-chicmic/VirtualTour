"""OneFormer (COCO panoptic) object masks for perceptual motion, and each photo's scene summary.

OneFormer COCO replaced Mask2Former ADE20K on 2026-11-20 (docs/adr/0001-segmentation-oneformer-coco.md):
the lit-fire effect is off, so ADE20K's fireplace class is no longer needed, and the other
pipelines no longer take Mask2Former's masks. Model `shi-labs/oneformer_coco_swin_large`, pinned.

COCO names many classes "<thing>-merged" / "-stuff" / "-other"; the Segmenter renames them once,
at load (COMMON), so scene summaries, motion regions and 3D labels all speak one vocabulary - the
one the planner, the room typing and the 3D planner read: "sofa", "table", "windowpane", "ceiling",
"water", "fence" (COCO's fences take in railings) ...

Water masks keep the liquid surface only (tour_worker.vision.refine: coping and basin walls stay
still).

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

MODELS = {
    "oneformer_coco": {
        "id": "shi-labs/oneformer_coco_swin_large",
        "revision": "3a263017ca5c75adbea145f25f81b118243d4394",
        "weights": "pytorch_model.bin",
        "sha256": "36d01ceb971e9b3824bf35de022cd3148aad0c928f6365302db492a4cff6bf2b",
        "dir": "oneformer_coco_swin_large",
        # common class name -> our motion class (the planner's perceptual_effects keys)
        "class_map": {"curtain": "curtain", "blind": "blinds", "plant": "potted_plant", "tree": "tree",
                      "water": "water", "sea": "water", "river": "water", "sky": "sky"},
    },
}
DEFAULT_MODEL = "oneformer_coco"
# COCO panoptic label -> the common name every reader uses (labels not listed keep their own)
COMMON = {
    "couch": "sofa", "dining table": "table", "table-merged": "table", "tv": "television receiver",
    "laptop": "computer", "window-other": "windowpane", "window-blind": "blind", "potted plant": "plant",
    "ceiling-merged": "ceiling", "sky-other-merged": "sky", "tree-merged": "tree", "grass-merged": "grass",
    "mountain-merged": "mountain", "rock-merged": "rock", "dirt-merged": "earth", "pavement-merged": "sidewalk",
    "fence-merged": "fence", "cabinet-merged": "cabinet", "building-other-merged": "building", "rug-merged": "rug",
    "mirror-stuff": "mirror", "door-stuff": "door", "water-other": "water", "paper-merged": "paper",
    "food-other-merged": "food", "wall-brick": "wall", "wall-stone": "wall", "wall-tile": "wall",
    "wall-wood": "wall", "wall-other-merged": "wall", "floor-wood": "floor", "floor-other-merged": "floor",
}


def common_name(label: str) -> str:
    label = str(label).strip()
    return COMMON.get(label, label)


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
                          allow_patterns=["*.json", "*.txt", spec["weights"]])
    h = hashlib.sha256()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    if h.hexdigest() != spec["sha256"]:
        raise RuntimeError("segmentation weights checksum mismatch")
    return model_dir


class Segmenter:
    """Loads OneFormer once; call it with a BGR image. model: a MODELS key; model_dir: its folder
    (Settings.segment_path()). id2label gives the common names (COMMON)."""

    def __init__(self, model_dir: str, model: str = DEFAULT_MODEL, threads: int | None = None):
        import torch
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
        if threads:
            torch.set_num_threads(threads)
        self.model_name, self.class_map = model, MODELS[model]["class_map"]
        d = ensure_model(model_dir, model)
        self.torch = torch
        self.processor = OneFormerProcessor.from_pretrained(d)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(d).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.id2label = {k: common_name(v) for k, v in self.model.config.id2label.items()}

    def __call__(self, bgr: np.ndarray) -> list[dict]:
        return self.segment(bgr)[0]

    def panoptic(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """(the photo at mask size, the panoptic id map, segments_info) - one OneFormer pass."""
        h, w = bgr.shape[:2]
        k = min(1.0, MAX_SIDE / max(w, h))
        if k < 1.0:
            bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        sh, sw = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, task_inputs=["panoptic"], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        # label_ids_to_fuse: nothing fused (the default) - said explicitly, or transformers logs a
        # line for every image (160 a scene_label run)
        res = self.processor.post_process_panoptic_segmentation(out, target_sizes=[(sh, sw)],
                                                                label_ids_to_fuse=set())[0]
        return bgr, res["segmentation"].cpu().numpy(), res["segments_info"]

    def segment(self, bgr: np.ndarray) -> tuple[list[dict], dict]:
        """(motion regions, scene summary) - one pass, then the water refinement."""
        bgr, seg, info = self.panoptic(bgr)
        sh, sw = bgr.shape[:2]
        regions = segments_to_regions(seg, info, self.id2label, (sw, sh), self.class_map)
        scene = scene_summary(seg, info, self.id2label, (sw, sh))
        return refine_regions(bgr, regions), scene


def refine_regions(bgr: np.ndarray, regions: list[dict]) -> list[dict]:
    """Water regions keep their liquid surface only (dropped when none is left;
    tour_worker.vision.refine). Other regions pass through."""
    from tour_worker.vision.refine import refine_water_surface
    out = []
    min_px = MIN_PART_AREA * bgr.shape[0] * bgr.shape[1]
    for r in regions:
        if r["class_name"] == "water":
            surface, _ = refine_water_surface(bgr, r["mask"])
            if int((surface > 0).sum()) < min_px:
                continue
            out.append(dict(r, mask=surface, bbox=_bbox(surface)))
        else:
            out.append(r)
    return out


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


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
        label = common_name(id2label.get(s["label_id"], id2label.get(str(s["label_id"]), "")))
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
    """What is in the photo, for choosing its camera move (bed, sofa, table, windowpane, stairs,
    ceiling, sky, water ...): every class (common names) with its share of the photo, the box
    around all its parts and their area-weighted centre, all normalised to the photo (0..1)."""
    sw, sh = space
    acc: dict[str, dict] = {}
    for s in segments_info:
        if float(s.get("score", 1.0)) < MIN_SCORE:
            continue
        label = common_name(id2label.get(s["label_id"], id2label.get(str(s["label_id"]), "")))
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

