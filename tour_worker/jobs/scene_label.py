"""`scene_label` job: a splat scene + its rooms -> what every Gaussian is (scene/semantics.py) and the
scene's motion (scene/motion_label.py), once per scene, at upload.

  fetch + load   the scene (as tour_render: sha256 checked; not kept on disk)
  survey         per room, rendered views (scene/semantics.survey_views) on the job's renderer: the
                 GPU (gsplat) on a GPU worker, the CPU reference otherwise
  segment        each view through Mask2Former (the tour_segment job's model, cached per process)
  vote           classes onto the Gaussians, then objects (a bed, each window ...) with their rooms
  motion         curtains / plants (grown into what no view saw) and lit fireplaces' flames
  store          <output_prefix>scene_labels.npz, and <output_prefix>scene_motion.npz when anything moves

Result: SceneLabelResult (contract/jobs.py). Its labels_uri / sha256 go into tour_plan's scene
(labels_uri), its motion_uri / sha256 into the plan's scene (motion_uri)."""
from __future__ import annotations

import os
import time

from tour_worker import progress as report
from tour_worker.config import Settings, get_settings
from tour_worker.contract.jobs import SceneLabelRequest
from tour_worker.errors import failure
from tour_worker.scene import SceneInvalid
from tour_worker.scene.cache import sha256_file
from tour_worker.timing import Stopwatch


def run_scene_label(request: dict, storage, workdir: str, settings: Settings | None = None, segmenter=None,
                    renderer=None) -> dict:
    """segmenter / renderer: overrides (tests); defaults from settings."""
    from tour_worker.planner3d.plan import room_keys
    from tour_worker.planner3d.space import Space
    from tour_worker.scene.cache import ensure_scene
    from tour_worker.scene.motion_label import motion_from_labels
    from tour_worker.scene.semantics import label_scene
    from tour_worker.scene.splat_io import load_scene
    t0 = time.time()
    settings = settings or get_settings()
    req = SceneLabelRequest.model_validate(request)
    prefix = req.output_prefix if req.output_prefix.endswith("/") else req.output_prefix + "/"
    sc = req.scene
    sw = Stopwatch()
    try:
        with sw.stage("scene_fetch"):
            path = ensure_scene(sc.uri, sc.sha256, sc.format, storage, os.path.join(workdir, "scene"),
                                req.image_urls.get(sc.uri))
        with sw.stage("scene_load"):
            model = load_scene(path, sc.format, max_sh_degree=settings.splat_sh_degree)
        os.remove(path)
    except SceneInvalid as e:
        return failure("scene_invalid", str(e))
    report.note(f"scene: {len(model):,} splats; {len(req.rooms)} room(s)")
    with sw.stage("floor_plan"):
        space = Space(model, sc.up_axis)
        rooms = [r.model_dump() for r in req.rooms]
        keys = room_keys(rooms)
        warnings = space.set_rooms({k: space.to_local(r["point"])[:2] for k, r in zip(keys, rooms)})
    if renderer is None:
        from tour_worker.render.splat_shot import make_renderer
        with sw.stage("renderer"):
            renderer = make_renderer(model, settings.splat_backend, settings.splat_sh_degree)
    if segmenter is None:
        from tour_worker.jobs.segment import get_segmenter
        with sw.stage("model_load"):
            segmenter = get_segmenter(settings)
    with sw.stage("survey_segment_vote"):
        labels, stats, seen = label_scene(model, space, renderer, segmenter, keys, sc.sha256,
                                          progress=lambda d, n: report.progress(d, n, "views"))
    report.note(f"labelled {stats['labelled']:,} of {stats['seen']:,} seen Gaussians; {stats['objects']} objects")
    with sw.stage("motion"):
        motion = motion_from_labels(model, labels, seen, space.local, space.up_scene, space.R, sc.sha256)
    with sw.stage("upload"):
        lpath = os.path.join(workdir, "scene_labels.npz")
        labels.save(lpath)
        out = {"labels_uri": prefix + "scene_labels.npz", "labels_sha256": sha256_file(lpath)}
        storage.put(lpath, out["labels_uri"])
        if motion is not None and len(motion):
            mpath = os.path.join(workdir, "scene_motion.npz")
            motion.save(mpath)
            out.update(motion_uri=prefix + "scene_motion.npz", motion_sha256=sha256_file(mpath))
            storage.put(mpath, out["motion_uri"])
            stats["motion"] = motion.classes
    stats["warnings"] = warnings
    return {"status": "completed", **out, "objects": labels.objects, "stats": stats,
            "seconds": round(time.time() - t0, 2), "timings": sw.result()}
