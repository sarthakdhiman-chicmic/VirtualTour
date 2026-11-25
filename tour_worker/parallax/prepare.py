"""Parallax inputs for a tour's photos (run by the tour_render job before drawing, on the GPU when there).

For every photo a photo shot draws: a working copy WORK_W wide, Depth Anything V2 Small's relative
depth and OneFormer's panoptic segmentation at the working size, turned into the photo's rigid planes
(layers.build_planes) and saved next to the image as <images>/.parallax/<name>.planes.npz. The render
processes build the drawable layers from it at the size they draw the photo (parallax/shot.py).

A photo that cannot be prepared is a warning; its shots render flat."""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from tour_worker.parallax import layers as L

WORK_W = 1920                     # depth, segmentation and the planes (the size the prototype was approved at)
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"     # apache-2.0
DEPTH_REV = "5426e4f0f36572d16453bbda7a8389317b1bef99"
DIRNAME = ".parallax"

_DEPTH: dict = {}


def get_depth(settings):
    """The depth pipeline, loaded once per warm worker (weights cached under MODELS_DIR/hf)."""
    if "pipe" not in _DEPTH:
        import torch
        from transformers import pipeline
        _DEPTH["pipe"] = pipeline("depth-estimation", model=DEPTH_MODEL, revision=DEPTH_REV,
                                  device=0 if torch.cuda.is_available() else -1,
                                  model_kwargs={"cache_dir": os.path.join(settings.models_dir, "hf")})
    return _DEPTH["pipe"]


def planes_path(images: str, name: str) -> str:
    return os.path.join(images, DIRNAME, name + ".planes.npz")


def photo_names(plan) -> list[str]:
    return sorted({s.image_uri.split("/")[-1] for s in plan.shots
                   if s.camera.type not in ("bezier_path", "video_clip") and s.image_uri})


def segment(seg, rgb) -> tuple[np.ndarray, dict[int, str]]:
    """OneFormer panoptic at the image's own size: (segment id map, {id: common label})."""
    import torch
    inputs = seg.processor(images=rgb, task_inputs=["panoptic"], return_tensors="pt").to(seg.device)
    with torch.no_grad():
        out = seg.model(**inputs)
    r = seg.processor.post_process_panoptic_segmentation(out, target_sizes=[rgb.size[::-1]], label_ids_to_fuse=set())[0]
    return (r["segmentation"].cpu().numpy().astype(np.int32),
            {int(s["id"]): seg.id2label[s["label_id"]] for s in r["segments_info"]})


def save_planes(path: str, planes, z: np.ndarray, shape) -> None:
    idx = np.full(shape[:2], -1, np.int16)
    for i, p in enumerate(planes):
        idx[p.mask] = i
    tmp = path + ".tmp.npz"
    np.savez(tmp, idx=idx, z=z.astype(np.float32), n=np.array([p.n for p in planes], np.float64).reshape(-1, 3),
             c=np.array([p.c for p in planes], np.float64), depth=np.array([p.depth for p in planes], np.float64),
             labels=np.array(json.dumps([p.label for p in planes])))
    os.replace(tmp, path)


def prepare(plan, images: str, settings, warnings: list) -> dict:
    """Planes for every photo the plan draws; returns {"photos": n prepared, "seconds": s}."""
    from PIL import Image

    from tour_worker.jobs.segment import get_segmenter
    os.makedirs(os.path.join(images, DIRNAME), exist_ok=True)
    t0, done = time.time(), 0
    depth = seg = None
    for name in photo_names(plan):
        out = planes_path(images, name)
        if os.path.exists(out):
            continue
        try:
            im = cv2.imread(os.path.join(images, name), cv2.IMREAD_COLOR)    # as ShotWarp reads it
            if im is None:
                raise FileNotFoundError(name)
            h, w = im.shape[:2]
            if w > WORK_W:
                im = cv2.resize(im, (WORK_W, round(WORK_W * h / w)), interpolation=cv2.INTER_AREA)
            if depth is None:
                depth, seg = get_depth(settings), get_segmenter(settings)
            rgb = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            inv = np.array(depth(rgb)["predicted_depth"]).astype(np.float32)
            ids, labels = segment(seg, rgb)
            planes, _, _ = L.build_planes(im.shape, inv, ids, labels)
            save_planes(out, planes, L.PHOTO_DEPTH[0], im.shape)
            done += 1
        except Exception as e:  # noqa: BLE001 - parallax is an addition: the photo's shots render flat
            warnings.append(f"enhancement_dropped:{name}:parallax:error_{type(e).__name__}")
    return {"photos": done, "seconds": round(time.time() - t0, 1)}
