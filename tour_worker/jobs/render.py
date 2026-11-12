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
from tour_worker.scene import SceneInvalid
from tour_worker.render.direct import HEVC_NAME, MLS_NAME, render_direct, wants_hevc, wants_mls
from tour_worker.render.probe import verify
from tour_worker.timing import Stopwatch


_SCENE: dict = {}          # the last scene's renderer, kept for the next render of the same scene


def scene_renderer(plan, storage, settings, image_urls: dict, sw, workdir: str):
    """The plan's scene, fetched, checked, loaded and on its renderer. The last one is kept in this
    process's memory (a 6 M-splat scene takes ~4 s to load and 1-2 GB to hold), so the next render
    of the same scene neither downloads nor loads it again.

    On disk: with SCENE_CACHE_DIR set, the file is kept there by sha256 (fetched once). Unset (the
    default), it is fetched into this job's workdir and deleted as soon as it is loaded."""
    from tour_worker.render.splat_shot import make_renderer
    from tour_worker.scene.cache import ensure_scene
    from tour_worker.scene.splat_io import load_scene
    sc = plan.scene
    key = (sc.sha256, settings.splat_sh_degree, settings.splat_backend)
    if _SCENE.get("key") == key:
        sw.info.update(scene_cached=True)
        return _SCENE["renderer"]
    _SCENE.clear()
    keep = bool(settings.scene_cache_dir)
    with sw.stage("scene_fetch"):
        path = ensure_scene(sc.uri, sc.sha256, sc.format, storage,
                            settings.scene_cache_dir if keep else os.path.join(workdir, "scene"),
                            image_urls.get(sc.uri))
    try:
        with sw.stage("scene_load"):
            model = load_scene(path, sc.format, max_sh_degree=settings.splat_sh_degree)
            renderer = make_renderer(model, settings.splat_backend, settings.splat_sh_degree)
    finally:
        if not keep and os.path.exists(path):
            os.remove(path)                       # no disk cache: the scene lives in memory only
    sw.info.update(scene_splats=len(model), scene_sh_degree=model.sh_degree, scene_cached=False)
    _SCENE.update(key=key, renderer=renderer)
    return renderer


def scene_motion(plan, renderer, storage, image_urls: dict, sw, workdir: str, warnings: list) -> None:
    """Attach the plan's 3D motion (scene.motion_uri) to the renderer, or take it off when the plan
    has none. A motion file that cannot be used never fails the tour: it renders still, with a
    warning (scene_motion_skipped)."""
    from tour_worker.render.splat_shot import GPU_LOCK
    from tour_worker.scene.cache import ensure_scene
    from tour_worker.scene.motion3d import MotionInvalid, check_binding, load_motion
    sc = plan.scene
    want = sc.motion_sha256 if sc.motion_uri else None
    if not hasattr(renderer, "attach_motion") or renderer.motion_sha256 == want:
        if want is not None:
            sw.info.update(motion_cached=True)
        return
    if want is None:
        with GPU_LOCK:
            renderer.attach_motion(None)
        return
    try:
        with sw.stage("motion_fetch"):
            path = ensure_scene(sc.motion_uri, sc.motion_sha256, "npz", storage, os.path.join(workdir, "motion"),
                                image_urls.get(sc.motion_uri))
        field = load_motion(path)
        why = check_binding(field, sc.sha256, int(renderer.means.shape[0]))
        if why:
            raise MotionInvalid(why)
    except Exception as e:                       # noqa: BLE001 - motion never fails the tour
        warnings.append("scene_motion_skipped")
        sw.info.update(motion_skipped=str(e)[:300])
        with GPU_LOCK:
            renderer.attach_motion(None)
        return
    with GPU_LOCK:
        renderer.attach_motion(field, sc.motion_sha256)
    sw.info.update(motion_splats=len(field), motion_classes=field.classes, motion_cached=False)


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
        splat = None
        if plan.scene is not None:                 # plan 1.6: the 3D scene and its renderer
            try:
                splat = scene_renderer(plan, storage, settings, image_urls, sw, workdir)
            except SceneInvalid as e:
                return failure("scene_invalid", str(e))
            scene_motion(plan, splat, storage, image_urls, sw, workdir, warnings)
        three_d = {"splat": splat, "splat_supersample": settings.splat_supersample} if splat is not None else {}
        out_path = render_direct(plan, workdir, warnings=warnings, music_path=music_path, voice_path=voice_path,
                                 workers=settings.render_workers, timings=sw, progress=progress, **three_d)

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
