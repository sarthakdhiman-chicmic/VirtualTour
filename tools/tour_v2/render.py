"""Render a tour plan through the worker renderer (modules/remotion_render.py) from a local image
folder. Images are resolved by the basename of each shot's image_uri.

  python -m tools.tour_v2.render --plan <plan.json> --images-dir <dir> --out <mp4> [--concurrency 6]
"""
import argparse
import json
import os
import shutil
import tempfile
import time

from modules.plan_io import parse_plan
from modules.remotion_render import render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--jpeg-quality", default="95", help='"default" = Remotion default (80)')
    ap.add_argument("--gl", default=None)
    args = ap.parse_args()

    raw = json.load(open(args.plan))
    plan = parse_plan(raw.get("plan", raw))
    with tempfile.TemporaryDirectory(prefix="tour_v2_") as wd:
        os.makedirs(os.path.join(wd, "images"))
        for s in plan.shots:
            name = s.image_uri.split("/")[-1]
            src = os.path.join(args.images_dir, name)
            if not os.path.exists(src):
                raise SystemExit(f"cannot resolve image for {s.shot_id}: {name}")
            shutil.copy(src, os.path.join(wd, "images", name))
        t = time.time()
        jq = None if args.jpeg_quality == "default" else int(args.jpeg_quality)
        out = render(plan, wd, 1.0, concurrency=args.concurrency, jpeg_quality=jq, gl=args.gl)
        elapsed = time.time() - t
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        shutil.copy(out, args.out)
    print(json.dumps({"out": args.out, "render_seconds": round(elapsed, 2), "concurrency": args.concurrency,
                      "jpeg_quality": args.jpeg_quality, "gl": args.gl}))


if __name__ == "__main__":
    main()
