"""Render a tour plan locally: plan + a folder of images -> MP4, through the production
tour_render job (tour_worker.jobs.render.run_tour: plan checks, image preparation, direct render,
playback checks).

  python -m tools.render --plan <plan.json> --images-dir <dir> --out <file.mp4>
      [--music-file <audio>]    plan 1.5 music; default: the plan's music uri (file:// or a path
                                is read from disk; s3:// / https:// from <images-dir> by basename)
      [--workers N]             drawing + encoding processes (default: PREWARP_WORKERS or auto)

Images are found by the basename of each shot's image_uri in --images-dir; motion-region masks
(plan 1.4) as <images-dir>/masks/<shot_id>_<region_id>.png, as a tour workdir has them. A missing
image fails the render (no placeholder). Prints the job result as JSON; exit 1 on failure.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace

from tour_worker.config import get_settings
from tour_worker.io.storage import CompositeStorage, HttpStorage, LocalMapStorage, S3Storage
from tour_worker.jobs.render import run_tour


class _Overrides:
    """A storage that serves some URIs from given files, everything else from `inner`."""

    def __init__(self, inner, files: dict):
        self.inner, self.files = inner, files

    def get(self, uri, dest):
        if uri in self.files:
            shutil.copy(self.files[uri], dest)
        else:
            self.inner.get(uri, dest)

    def put(self, path, uri):
        self.inner.put(path, uri)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", required=True, help="plan JSON (or a request JSON with a 'plan' key)")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True, help="the MP4 to write")
    ap.add_argument("--music-file", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)

    raw = json.load(open(a.plan))
    plan = raw.get("plan", raw)
    settings = get_settings()
    if a.workers:
        settings = replace(settings, render_workers=a.workers)
    out = os.path.abspath(a.out)
    out_dir = os.path.dirname(out)
    os.makedirs(out_dir, exist_ok=True)
    storage = CompositeStorage(s3=S3Storage(), http=HttpStorage(),
                               local=LocalMapStorage(a.images_dir, placeholder=False), local_for_remote=True)
    music = (plan.get("music") or {}).get("uri")
    if a.music_file and music:
        storage = _Overrides(storage, {music: a.music_file})
    with tempfile.TemporaryDirectory(prefix="tour_render_") as wd:
        os.makedirs(os.path.join(wd, "images"))
        masks = os.path.join(a.images_dir, "masks")
        if os.path.isdir(masks):                 # pre-fetched masks: the job skips existing ones
            shutil.copytree(masks, os.path.join(wd, "images", "masks"))
        with tempfile.TemporaryDirectory(prefix="tour_out_", dir=out_dir) as od:
            res = run_tour({"job_type": "tour_render", "plan": plan, "output_prefix": od + "/"},
                           storage, wd, settings=settings)
            if res.get("status") == "completed":
                os.replace(os.path.join(od, "master.mp4"), out)
                res["video_uri"] = out
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
