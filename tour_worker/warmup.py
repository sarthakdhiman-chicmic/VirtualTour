"""Load the heavy libraries when the service starts, not in the first job.

On the RunPod pod the worker's venv lives on the network volume (/workspace, MooseFS), and Python
loads a module's files on first import: the first job after a restart paid minutes for it (Turkey
Trot 4K, 2026-11-16: fetch 252 s vs 19 s on a warm worker - boto3 on first use; then 6 minutes
before the first frame - torch and the CUDA libraries). warm() imports them in a background thread
at startup (the service answers at once; a job that arrives first just waits on the imports it
needs), opens the CUDA context and loads the segmentation model. WORKER_WARMUP=1 turns it on
(tools/runpod/env.sh)."""
from __future__ import annotations

import importlib
import logging
import threading
import time

log = logging.getLogger("tour_worker.warmup")

MODULES = ("numpy", "cv2", "boto3", "botocore", "scipy.ndimage", "PIL.Image", "torch",
           "torch.nn.functional", "tour_worker.render.direct", "tour_worker.render.frames_gpu",
           "tour_worker.render.splat_shot", "tour_worker.scene.semantics", "tour_worker.planner3d.plan",
           "tour_worker.jobs.router", "transformers")


_DONE = threading.Event()
_DONE.set()                      # no warm-up started: nothing to wait for


def wait(timeout: float = 900.0) -> None:
    """Block until the warm-up has finished: a job importing a library while the warm-up thread is
    importing the same one can see it half-loaded (2026-11-16: "cannot import name
    Mask2FormerForUniversalSegmentation from transformers")."""
    _DONE.wait(timeout)


def warm() -> None:
    try:
        _warm()
    finally:
        _DONE.set()


def _warm() -> None:
    t0 = time.time()
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - an optional dependency missing is not a startup failure
            log.info("warm-up: %s not loaded (%s)", name, type(e).__name__)
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")          # the CUDA context and its libraries
    except Exception as e:  # noqa: BLE001
        log.info("warm-up: no CUDA context (%s)", type(e).__name__)
    log.info("warm-up: libraries loaded in %.0f s", time.time() - t0)
    # the segmentation model too (2026-11-23: the first tour_segment after a restart spent 32 s of its
    # 73 s loading OneFormer); kept by jobs.segment.get_segmenter for every later job
    t1 = time.time()
    try:
        from tour_worker.config import get_settings
        from tour_worker.jobs.segment import get_segmenter
        get_segmenter(get_settings())
        log.info("warm-up: segmentation model loaded in %.0f s", time.time() - t1)
    except Exception as e:  # noqa: BLE001 - no weights here (a render-only host): the first job loads it
        log.info("warm-up: segmentation model not loaded (%s)", type(e).__name__)
    t2 = time.time()
    try:                                           # the depth model (parallax, client review item 10)
        from tour_worker.config import get_settings
        from tour_worker.parallax.prepare import get_depth
        get_depth(get_settings())
        log.info("warm-up: depth model loaded in %.0f s", time.time() - t2)
    except Exception as e:  # noqa: BLE001 - the first parallax render loads it
        log.info("warm-up: depth model not loaded (%s)", type(e).__name__)


def start() -> threading.Thread:
    _DONE.clear()
    t = threading.Thread(target=warm, name="warmup", daemon=True)
    t.start()
    return t
