"""Make a scene's 3D motion file (plan 1.6 scene.motion_uri) from segmented frames of a tour.

  python -m tools.scene.label_motion --scene scene.ply --plan plan.json --views views.json --out motion.npz
      [--sh-degree 3]      load the scene as the worker renders it (the Gaussian count must match)

  views.json   [{"time_s": 12.4, "masks": [{"class_name": "curtain", "path": "mask.png"}, ...]}, ...]
               frames of the rendered plan (their cameras are recomputed from the plan, exactly as
               the renderer drew them) and the tour_segment masks of those frames.

Prints what moves (objects, Gaussians per class). Upload the .npz and set scene.motion_uri /
scene.motion_sha256 (its sha256) in the plan. tour_worker/scene/motion_label.py has the method.
"""
import argparse
import json
import sys

import cv2
import numpy as np

from tour_worker.contract.plan import parse_plan
from tour_worker.render.splat_shot import SplatShot
from tour_worker.render.timeline import compute_timeline
from tour_worker.scene.cache import sha256_file
from tour_worker.scene.motion_label import CLASSES, View, build_field, label, objects
from tour_worker.scene.splat_io import load_scene


def cameras(plan_dict: dict, times: list[float]):
    """The renderer's camera (view matrix, K at the output size) at each tour time."""
    plan = parse_plan(plan_dict)
    tl = compute_timeline(plan)
    fps, out = plan.output.fps, []
    for t in times:
        g = int(round(t * fps))
        best = None
        for i, e in enumerate(tl):               # the shot on screen (not one fading out)
            f = g - e["render_from"]
            if 0 <= f < e["render_frames"]:
                best = (i, f)
        if best is None:
            raise SystemExit(f"time {t} s is outside the tour")
        i, f = best
        shot = SplatShot(plan, plan.shots[i], tl[i]["render_frames"], None, supersample=1.0)
        V, K = shot.view(f)
        out.append((V, K, plan.output.width, plan.output.height))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sh-degree", type=int, default=3)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    views_in = json.load(open(a.views))
    cams = cameras(plan, [float(v["time_s"]) for v in views_in])
    views = []
    for v, (V, K, W, H) in zip(views_in, cams):
        masks = {}
        for m in v["masks"]:
            if m["class_name"] not in CLASSES:
                continue
            img = cv2.imread(m["path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise SystemExit(f"cannot read mask {m['path']}")
            prev = masks.get(m["class_name"])
            masks[m["class_name"]] = (img > 127) | (prev if prev is not None else False)
        views.append(View(V, K, W, H, masks))
    sha = sha256_file(a.scene)
    model = load_scene(a.scene, max_sh_degree=a.sh_degree)
    lab = label(model, views)
    field = build_field(model, lab, (plan.get("scene") or {}).get("up_axis"), sha)
    field.save(a.out)
    objs = objects(model, lab)
    print(json.dumps({"scene_sha256": sha, "splats": len(model), "moving": len(field), "classes": field.classes,
                      "objects": [{"class": [k for k, c in CLASSES.items() if c == cid][0], "gaussians": len(m)}
                                  for cid, m in objs],
                      "out": a.out, "out_sha256": sha256_file(a.out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
