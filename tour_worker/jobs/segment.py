"""`tour_segment` job: OneFormer masks for a tour's photos, before tour_plan plans.

Request:  {"job_type": "tour_segment", "job_id": ..., "images": [{"photo_id": ..., "uri": ...}],
           "output_prefix": "s3://bucket/.../segments/<job_id>/"}
Response: {"status": "completed", "results": {photo_id: [region, ...] | None},
           "scenes": {photo_id: scene summary | None}, "model": ..., "warnings": [...]}
A scene summary (tour_worker.vision.segment.scene_summary) lists every class in the photo - bed, couch,
window, stairs, ceiling, sky ... - for the planner's choice of camera move.

Each region is shaped like a detect result plus `space` (tour_worker.vision.segment); its mask is written to
<output_prefix><photo_id>/<object_id>.png and returned as `mask_url`. No images go back to the
backend: tour_plan plans from the JSON, and the render job fetches the masks it needs. A photo that cannot
be fetched or segmented gets None (the tour still renders, without motion on that photo).
Contracts: tour_worker.contract.jobs (TourSegmentRequest / TourSegmentResult). Settings used:
SEGMENT_MODEL, ONEFORMER_DIR / MODELS_DIR, COCO_FIRE_SEARCH, FIREPLACE_MODEL / FIREPLACE_DIR
(tour_worker.config). The fireplace pass (FIREPLACE_MODEL, default ade20k) runs a second OneFormer
for fireplaces only - COCO has no fireplace class - and adds the lit fires found inside them.
"""
from __future__ import annotations

import os
import time

import cv2

from tour_worker.config import Settings, get_settings
from tour_worker.errors import failure
from tour_worker.vision.segment import model_id

_SEGMENTERS: dict = {}                  # loaded once per warm worker (per model / folder)


def get_segmenter(settings: Settings):
    from tour_worker.vision.segment import Segmenter
    key = (settings.segment_model, settings.oneformer_path(), settings.coco_fire_search)
    if key not in _SEGMENTERS:
        _SEGMENTERS[key] = Segmenter(settings.oneformer_path(), settings.segment_model,
                                     fire_search=settings.coco_fire_search)
    return _SEGMENTERS[key]


def get_fireplace_segmenter(settings: Settings):
    """The fireplace pass's model (None when FIREPLACE_MODEL is off, or it is the primary model)."""
    from tour_worker.vision.segment import Segmenter
    if not settings.fireplace_model or settings.fireplace_model == settings.segment_model:
        return None
    key = ("fireplace", settings.fireplace_model, settings.fireplace_path())
    if key not in _SEGMENTERS:
        _SEGMENTERS[key] = Segmenter(settings.fireplace_path(), settings.fireplace_model)
    return _SEGMENTERS[key]


# Bump when the regions or scenes this job returns change for the same models (class maps,
# merges such as the one sky region, mask clean-up): the backend caches results per photo
# under segment_version and reuses them only while it matches.
SEGMENT_VERSION = "1"


def segment_version(settings: Settings) -> str:
    """What a photo's segmentation depends on besides the photo: this job's logic, the models
    (pinned revisions) and the fire settings."""
    fp = settings.fireplace_model if settings.fireplace_model and settings.fireplace_model != settings.segment_model else None
    return (f"seg{SEGMENT_VERSION}|{model_id(settings.segment_model)}|"
            f"{model_id(fp) if fp else 'nofp'}|firesearch{int(bool(settings.coco_fire_search))}")


def run_segment(request: dict, storage, workdir: str, segmenter=None, settings: Settings | None = None,
                fireplace_segmenter=None) -> dict:
    """An image whose `known_version` is this worker's segment_version is not segmented again:
    it is listed in `cached` (the caller already holds its result). The models load only if an
    image needs them."""
    t0 = time.time()
    settings = settings or get_settings()
    images = request.get("images") or []
    prefix = request.get("output_prefix") or ""
    if not images or not prefix:
        return failure("invalid_request", "images and output_prefix are required")
    if not prefix.endswith("/"):
        prefix += "/"
    version = segment_version(settings)
    cached = [str(i.get("photo_id")) for i in images if i.get("known_version") == version]
    todo = [i for i in images if i.get("known_version") != version]
    seg = fp_seg = None
    if todo:
        seg = segmenter or get_segmenter(settings)
        fp_seg = fireplace_segmenter if fireplace_segmenter is not None else (
            None if segmenter is not None else get_fireplace_segmenter(settings))
    results, scenes, warnings = {}, {}, []
    for item in todo:
        pid, uri = str(item.get("photo_id")), item.get("uri")
        local = os.path.join(workdir, f"in_{len(results):03d}")
        try:
            storage.get(uri, local)
            img = cv2.imread(local, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("unreadable image")
            if hasattr(seg, "segment"):
                regions, scenes[pid] = seg.segment(img)
            else:
                regions, scenes[pid] = seg(img), None
            if fp_seg is not None:                       # lit fires, from the fireplace model
                try:
                    regions = [r for r in regions if r.get("class_name") != "fire"] + fp_seg.fires(img)
                except Exception as e:  # the fireplace pass never fails the photo
                    warnings.append(f"fireplace_pass_failed:{pid}:{type(e).__name__}")
        except Exception as e:  # one photo never fails the job
            results[pid] = None
            scenes[pid] = None
            warnings.append(f"segment_failed:{pid}:{type(e).__name__}")
            continue
        out = []
        for r in regions:
            path = os.path.join(workdir, f"{pid}_{r['object_id']}.png")
            cv2.imwrite(path, r.pop("mask"))
            dest = f"{prefix}{pid}/{r['object_id']}.png"
            try:
                storage.put(path, dest)
            except Exception:
                warnings.append(f"mask_upload_failed:{pid}:{r['object_id']}")
                continue
            r["mask_url"] = dest
            out.append(r)
        results[pid] = out
    return {"status": "completed", "results": results, "scenes": scenes, "warnings": warnings,
            "cached": cached, "segment_version": version,
            "model": model_id(settings.segment_model),
            "fireplace_model": model_id(settings.fireplace_model) if fp_seg is not None else None,
            "seconds": round(time.time() - t0, 2)}
