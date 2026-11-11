"""`tour_render` job: plan -> images (+ masks, music) -> direct render -> checks -> <output_prefix>master.mp4
(+ master_hevc.mp4, master_mls.mp4 when the plan asks for those copies).

Request / result: tour_worker.contract.jobs (TourRenderRequest / TourRenderResult); failures are
{"status": "failed", "error": {"code", "message"}} (tour_worker.errors)."""
from __future__ import annotations

import os
import time

from tour_worker.config import Settings, get_settings
from tour_worker.contract.plan import parse_plan, plan_failure
from tour_worker.errors import failure, from_exception
from tour_worker.io.images import fetch_and_prepare
from tour_worker.render.direct import HEVC_NAME, MLS_NAME, render_direct, wants_hevc, wants_mls
from tour_worker.render.probe import verify
from tour_worker.timing import Stopwatch


def run_tour(request: dict, storage, workdir: str, settings: Settings | None = None, progress=None) -> dict:
    """Run one tour_render request in workdir (an empty folder the caller removes). progress(done,
    total): render progress, as chunks finish (the RunPod handler reports it)."""
    start_time = time.time()
    settings = settings or get_settings()
    sw = Stopwatch()
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
        sw.lap()
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
        voice_path = None
        if plan.voiceover is not None:             # fetched like the music (image_urls may map it)
            voice_path = os.path.join(workdir, "voiceover", os.path.basename(plan.voiceover.uri) or "voice")
            os.makedirs(os.path.dirname(voice_path), exist_ok=True)
            try:
                storage.get(image_urls.get(plan.voiceover.uri, plan.voiceover.uri), voice_path)
            except Exception:
                warnings.append("voiceover_fetch_failed")
                voice_path = None
        sw.lap("fetch")
        out_path = render_direct(plan, workdir, warnings=warnings, music_path=music_path, voice_path=voice_path,
                                 workers=settings.render_workers, timings=sw, progress=progress)

        sw.lap()
        verify(out_path, plan)

        master_uri = output_prefix + "master.mp4"
        hevc_uri = mls_uri = None
        if wants_hevc(plan):                       # checked like the master, before any is stored
            hevc_path = os.path.join(workdir, HEVC_NAME)
            verify(hevc_path, plan, codec="hevc")
        if wants_mls(plan):
            mls_path = os.path.join(workdir, MLS_NAME)
            verify(mls_path, plan)
        sw.lap("verify")
        storage.put(out_path, master_uri)
        if wants_hevc(plan):
            hevc_uri = output_prefix + "master_hevc.mp4"
            storage.put(hevc_path, hevc_uri)
        if wants_mls(plan):
            mls_uri = output_prefix + "master_mls.mp4"
            storage.put(mls_path, mls_uri)
        sw.lap("upload")
        sw.info.update(width=plan.output.width, height=plan.output.height, video_s=plan.total_duration_s)

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
            "hevc_uri": hevc_uri,
            "mls_uri": mls_uri,
            "timings": sw.result(),
            "error": None
        }
    except Exception as e:
        return from_exception(e)
