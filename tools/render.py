"""Render a tour plan through the worker renderer from a local image folder. Images are resolved by
the basename of each shot's image_uri. The direct renderer (tour_worker/render/direct.py: OpenCV +
ffmpeg) is the default; TOUR_RENDERER=remotion uses the old Remotion / Chrome path.

  python -m tools.render --plan <plan.json> --images-dir <dir> --out <mp4>
      [--music-file <audio>]      (plan 1.5 music; default: the plan's music uri if it is file://)
"""
import argparse
import json
import os
import shutil
import tempfile
import time

from tour_worker.contract.plan import parse_plan
from tour_worker.render.probe import verify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("TOUR_RENDER_CONCURRENCY", "6")))
    ap.add_argument("--jpeg-quality", default="95", help='"default" = Remotion default (80)')
    ap.add_argument("--gl", default=None)
    ap.add_argument("--music-file", default=None)
    args = ap.parse_args()
    renderer = os.environ.get("TOUR_RENDERER", "direct")

    raw = json.load(open(args.plan))
    plan = parse_plan(raw.get("plan", raw))

    with tempfile.TemporaryDirectory(prefix="tour_render_") as wd:
        os.makedirs(os.path.join(wd, "images"))
        for s in plan.shots:
            name = s.image_uri.split("/")[-1]
            src = os.path.join(args.images_dir, name)
            if not os.path.exists(src):
                raise SystemExit(f"cannot resolve image for {s.shot_id}: {name}")
            shutil.copy(src, os.path.join(wd, "images", name))
        # plan 1.4 motion-region masks, pre-fetched as <images-dir>/masks/<shot>_<id>.png
        masks = os.path.join(args.images_dir, "masks")
        if os.path.isdir(masks):
            shutil.copytree(masks, os.path.join(wd, "images", "masks"))
        t = time.time()
        if renderer == "remotion":
            from modules.remotion_render import render
            jq = None if args.jpeg_quality == "default" else int(args.jpeg_quality)
            out = render(plan, wd, 1.0, concurrency=args.concurrency, jpeg_quality=jq, gl=args.gl)
        else:
            from tour_worker.render.direct import render_direct
            music = args.music_file
            if music is None and plan.music is not None and plan.music.uri.startswith("file://"):
                music = plan.music.uri[len("file://"):]
            out = render_direct(plan, wd, music_path=music)
        elapsed = time.time() - t
        playback = verify(out, plan)   # production checks, including smooth-playback ones
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        shutil.copy(out, args.out)
    print(json.dumps({"out": args.out, "renderer": renderer, "render_seconds": round(elapsed, 2),
                      "playback": playback}))


if __name__ == "__main__":
    main()
