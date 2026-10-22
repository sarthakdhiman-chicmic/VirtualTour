"""`tour_render` job: plan -> images (+ masks, music) -> direct render -> checks -> <output_prefix>master.mp4.

Request / result: tour_worker.contract.jobs (TourRenderRequest / TourRenderResult); failures are
{"status": "failed", "error": {"code", "message"}} (tour_worker.errors)."""
from __future__ import annotations

import os
import time

from tour_worker.config import Settings, get_settings
from tour_worker.contract.plan import parse_plan, plan_failure
from tour_worker.errors import failure, from_exception
from tour_worker.io.images import fetch_and_prepare
from tour_worker.render.direct import render_direct
from tour_worker.render.probe import verify


def run_tour(request: dict, storage, workdir: str, settings: Settings | None = None) -> dict:
    """Run one tour_render request in workdir (an empty folder the caller removes)."""
    start_time = time.time()
    settings = settings or get_settings()
    try:
        plan_dict = request.get("plan", {})
        try:
            plan = parse_plan(plan_dict)
        except ValueError as e:
            return plan_failure(e)

        # required (it defaulted to "s3://dummy/out/" before 2026-10: a render nobody could find)
        output_prefix = request.get("output_prefix")
        if not output_prefix:
            return failure("invalid_request", "output_prefix is required")

        image_urls = request.get("image_urls", {})
        warnings = fetch_and_prepare(plan, storage, workdir, image_urls)

        music_path = None
        if plan.music is not None:                 # plan 1.5: fetch the track
            # named after the plan's URI (not the fetch URL, whose query string is a signature)
            music_path = os.path.join(workdir, "music", os.path.basename(plan.music.uri) or "track")
            os.makedirs(os.path.dirname(music_path), exist_ok=True)
            try:
                # like photos and masks: image_urls may map the track's URI to a fetch URL
                storage.get(image_urls.get(plan.music.uri, plan.music.uri), music_path)
            except Exception:
                warnings.append("music_fetch_failed")
                music_path = None
        out_path = render_direct(plan, workdir, warnings=warnings, music_path=music_path,
                                 workers=settings.render_workers)

        verify(out_path, plan)

        master_uri = output_prefix + "master.mp4"
        storage.put(out_path, master_uri)

        return {
            "status": "completed",
            "video_uri": master_uri,
            "duration_s": plan.total_duration_s,
            "width": plan.output.width,
            "height": plan.output.height,
            "fps": plan.output.fps,
            "codec": plan.output.codec,
            "render_seconds": time.time() - start_time,
            "plan_hash": plan.provenance.plan_hash if hasattr(plan, "provenance") else None,
            "warnings": warnings,
            "error": None
        }
    except Exception as e:
        return from_exception(e)
