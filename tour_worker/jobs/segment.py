"""`tour_segment` job: OneFormer masks for a tour's photos, before the backend plans.

Request:  {"job_type": "tour_segment", "job_id": ..., "images": [{"photo_id": ..., "uri": ...}],
           "output_prefix": "s3://bucket/.../segments/<job_id>/"}
Response: {"status": "completed", "results": {photo_id: [region, ...] | None},
           "scenes": {photo_id: scene summary | None}, "model": ..., "warnings": [...]}
A scene summary (tour_worker/vision/segment.py scene_summary) lists every class in the photo - bed, couch,
window, stairs, ceiling, sky ... - for the planner's choice of camera move.

Each region is shaped like a detect result plus `space` (tour_worker/vision/segment.py); its mask is written to
<output_prefix><photo_id>/<object_id>.png and returned as `mask_url`. No images go back to the
backend: it plans from the JSON, and the render job fetches the masks it needs. A photo that cannot
be fetched or segmented gets None (the tour still renders, without motion on that photo).
"""
from __future__ import annotations

import os
import time

import cv2

from tour_worker.vision.segment import MODEL_ID, MODEL_REVISION, Segmenter

_SEGMENTER: Segmenter | None = None     # loaded once per warm worker


def get_segmenter() -> Segmenter:
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = Segmenter()
    return _SEGMENTER


def run_segment(request: dict, storage, workdir: str, segmenter=None) -> dict:
    t0 = time.time()
    images = request.get("images") or []
    prefix = request.get("output_prefix") or ""
    if not images or not prefix:
        return {"status": "failed", "error": {"code": "invalid_request", "message": "images and output_prefix are required"}}
    if not prefix.endswith("/"):
        prefix += "/"
    seg = segmenter or get_segmenter()
    results, scenes, warnings = {}, {}, []
    for item in images:
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
            "model": f"{MODEL_ID}@{MODEL_REVISION[:10]}", "seconds": round(time.time() - t0, 2)}
