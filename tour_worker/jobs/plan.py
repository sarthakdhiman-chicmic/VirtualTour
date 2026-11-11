"""`tour_plan` job: a tour's photos -> photo scoring -> shot plan (the plan tour_render renders).

Request:  {"job_type": "tour_plan", "job_id": ..., "listing_id": ...,
           "photos": [{"photo_id", "uri", "room_type", "staged"?}], "image_urls"?: {uri: fetch URL},
           "segments"?: {photo_id: [region, ...] | None}, "scenes"?: {photo_id: summary | None},
           "music_track"?: {"track_id", "s3_uri", "sha256", "analysis"},
           "options"?: {"preset", "budget_blocked", "output_width", "output_height", "max_total_s",
                        "max_bytes", "hevc", "pacing"?: slow|normal|fast,
                        "room_labels"?: {"font", "scale", "placement"}, "mls_copy"}}
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
from tour_worker.planner.transitions import is_dusk
from tour_worker.scoring.rooms import tier_for
from tour_worker.timing import Stopwatch


def _fetch_photos(req: TourPlanRequest, storage, workdir: str):
    """(scoring inputs, assets, warnings, twilight photo ids). assets: photo_id -> {uri, width, height}, the planner's
    asset map (width / height as OpenCV decodes the photo, EXIF orientation applied)."""
    inputs, assets, warnings, dusk = [], {}, [], set()
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
        if tier_for(photo.room_type) == "exterior" and is_dusk(img):   # daylight <-> twilight: a fade
            dusk.add(photo.photo_id)
        assets[photo.photo_id] = {"uri": photo.uri, "width": img.shape[1], "height": img.shape[0]}
    return inputs, assets, warnings, frozenset(dusk)


def _branding(raw: dict | None) -> dict | None:
    """options.branding, checked against the plan contract (contract.plan.Branding); None or empty: none."""
    if not raw:
        return None
    from tour_worker.contract.plan import Branding
    return Branding(**raw).model_dump(exclude_none=True)


def _voiceover(raw: dict | None) -> dict | None:
    """options.voiceover, checked against the plan contract (contract.plan.Voiceover)."""
    if not raw:
        return None
    from tour_worker.contract.plan import Voiceover
    v = Voiceover(**raw)
    if not 0.0 <= v.balance <= 1.0:
        raise ValueError("voiceover.balance must be within 0-1")
    return v.model_dump()


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

    sw = Stopwatch()
    inputs, assets, warnings, dusk = _fetch_photos(req, storage, workdir)
    sw.lap("fetch")
    if not inputs:
        return failure("image_fetch_failed",
                       f"none of the {len(req.photos)} photos could be fetched and decoded")

    from tour_worker.planner.config import PlannerConfig
    from tour_worker.planner.planner import build_shot_plan
    from tour_worker.scoring.pipeline import score_photos

    opts = req.options
    try:
        report = score_photos(inputs)
        sw.lap("scoring")
        if opts.photo_order == "manual":           # drag-to-reorder: the request's order, not the narrative
            from tour_worker.scoring.selector import manual_order
            report = manual_order(report, [p.photo_id for p in req.photos])
        from tour_worker.planner.tour_presets import preset as tour_preset
        pre = tour_preset(opts.preset)
        branding = _branding(opts.branding)
        if pre.get("kicker"):                      # Just Listed: a line above the address
            branding = {**(branding or {}), "kicker": pre["kicker"]}
        end_card = bool(branding and (branding.get("agent") or branding.get("logo_uri")))
        extra = {"final_shot_min_s": PlannerConfig().end_card_hold_s} if end_card else {}
        if pre.get("outdoor_boost"):
            extra["outdoor_boost"] = pre["outdoor_boost"]
        if pre.get("open_high"):
            extra["music_v2_open_high"] = 6.0
        cfg = PlannerConfig(**extra, pacing=opts.pacing or pre["pacing"],
                            output_width=opts.output_width, output_height=opts.output_height,
                            max_total_s=opts.max_total_s, output_max_bytes=opts.max_bytes,
                            output_hevc=opts.hevc, output_mls_copy=opts.mls_copy,
                            output_room_labels=opts.room_labels.model_dump() if opts.room_labels else None,
                            staged_photos=frozenset(p.photo_id for p in req.photos if p.staged),
                            output_branding=branding, voiceover=_voiceover(opts.voiceover), dusk_photos=dusk)
        plan = build_shot_plan(
            report, req.listing_id, assets, preset=opts.preset,
            budget_blocked=opts.budget_blocked, cfg=cfg,
            detections_by_photo=req.segments,
            music_track=req.music_track.model_dump() if req.music_track else None,
            scenes_by_photo=req.scenes or None,
        )
    except Exception as e:  # noqa: BLE001
        return failure("planning_failed", f"planning_failed: {type(e).__name__}: {e}")

    sw.lap("planning")
    plan_dict = plan.model_dump(mode="json")
    try:
        parse_plan(plan_dict)                      # the renderer's own validator
    except ValueError as e:
        return plan_failure(e)

    return {"status": "completed", "plan": plan_dict, "plan_hash": plan.provenance.plan_hash,
            "scoring": _scoring_summary(report), "warnings": warnings,
            "seconds": round(time.time() - t0, 2), "timings": sw.result()}
