"""`tour_plan` job: a tour's photos -> photo scoring -> shot plan (the plan tour_render renders).

Request:  {"job_type": "tour_plan", "job_id": ..., "listing_id": ...,
           "photos": [{"photo_id", "uri", "room_type"}], "image_urls"?: {uri: fetch URL},
           "segments"?: {photo_id: [region, ...] | None}, "scenes"?: {photo_id: summary | None},
           "music_track"?: {"track_id", "s3_uri", "sha256", "analysis"},
           "options"?: {"preset", "budget_blocked", "output_width", "output_height", "max_total_s"}}
Response: {"status": "completed", "plan": {...}, "plan_hash": ..., "scoring": {...},
           "warnings": [...], "seconds": ...}

The photos are fetched like tour_render's (image_urls maps a photo's uri to the URL to fetch
instead; the plan keeps the uri, as hashed). Scoring (tour_worker.scoring) and planning
(tour_worker.planner.build_shot_plan) are the backend's former code, unchanged: the same inputs
give the same plan, byte for byte. `segments` and `scenes` are the tour_segment job's `results`
and `scenes`. The plan is checked with contract.plan.parse_plan, the validator tour_render uses,
before it is returned.

A photo that cannot be fetched or decoded is left out with a warning (photo_fetch_failed:<id>,
photo_unreadable:<id>); with no photo left the job fails with image_fetch_failed. Scoring or
planning errors fail it with planning_failed; a plan the validator rejects keeps the validator's
code (invalid_plan, ...).
Contracts: tour_worker.contract.jobs (TourPlanRequest / TourPlanResult).
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from tour_worker.contract.jobs import TourPlanRequest
from tour_worker.contract.plan import parse_plan, plan_failure
from tour_worker.errors import failure


def _fetch_photos(req: TourPlanRequest, storage, workdir: str):
    """(scoring inputs, assets, warnings). assets: photo_id -> {uri, width, height}, the planner's
    asset map (width / height as OpenCV decodes the photo, EXIF orientation applied)."""
    inputs, assets, warnings = [], {}, []
    for i, photo in enumerate(req.photos):
        local = os.path.join(workdir, f"photo_{i:03d}")
        try:
            storage.get(req.image_urls.get(photo.uri, photo.uri), local)
            with open(local, "rb") as f:
                data = f.read()
        except Exception as e:  # noqa: BLE001 - one photo never fails the job
            warnings.append(f"photo_fetch_failed:{photo.photo_id}:{type(e).__name__}")
            continue
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            warnings.append(f"photo_unreadable:{photo.photo_id}")
            continue
        inputs.append((photo.photo_id, data, photo.room_type))
        assets[photo.photo_id] = {"uri": photo.uri, "width": img.shape[1], "height": img.shape[0]}
    return inputs, assets, warnings


def _scoring_summary(report) -> dict:
    return {
        "config_version": report.config.version,
        "photos": len(report.photos),
        "rooms": [{"room_key": r.room_key, "room_type": r.room_type,
                   "narrative_order": r.narrative_order, "hero_photo_id": r.hero_photo_id,
                   "photo_ids": [p.photo_id for p in r.photos]} for r in report.rooms],
        "sequence": list(report.sequence),
        "warnings": list(report.warnings),
    }


def run_plan(request: dict, storage, workdir: str) -> dict:
    """Run one tour_plan request (already shape-checked by the router) in workdir."""
    t0 = time.time()
    try:
        req = TourPlanRequest.model_validate(request)
    except ValueError as e:
        return failure("invalid_request", str(e))

    inputs, assets, warnings = _fetch_photos(req, storage, workdir)
    if not inputs:
        return failure("image_fetch_failed",
                       f"none of the {len(req.photos)} photos could be fetched and decoded")

    from tour_worker.planner.config import PlannerConfig
    from tour_worker.planner.planner import build_shot_plan
    from tour_worker.scoring.pipeline import score_photos

    opts = req.options
    try:
        report = score_photos(inputs)
        cfg = PlannerConfig(output_width=opts.output_width, output_height=opts.output_height,
                            max_total_s=opts.max_total_s)
        plan = build_shot_plan(
            report, req.listing_id, assets, preset=opts.preset,
            budget_blocked=opts.budget_blocked, cfg=cfg,
            detections_by_photo=req.segments,
            music_track=req.music_track.model_dump() if req.music_track else None,
            scenes_by_photo=req.scenes or None,
        )
    except Exception as e:  # noqa: BLE001
        return failure("planning_failed", f"planning_failed: {type(e).__name__}: {e}")

    plan_dict = plan.model_dump(mode="json")
    try:
        parse_plan(plan_dict)                      # the renderer's own validator
    except ValueError as e:
        return plan_failure(e)

    return {"status": "completed", "plan": plan_dict, "plan_hash": plan.provenance.plan_hash,
            "scoring": _scoring_summary(report), "warnings": warnings,
            "seconds": round(time.time() - t0, 2)}
