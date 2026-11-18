"""What a scene's motion does, measured: the same render with motion and without.

  python -m tools.quality.motion_diff <with_motion.mp4> <without_motion.mp4> <out_dir> [--every 1]

The two renders share their camera frame for frame (same plan, one with scene.motion_uri and one
without), so their difference is exactly what the motion moves - no camera flow to estimate. For
the 3D tours' curtains, plants and flames (scene/motion_label.py) it reports

  - moving share: pixels the motion changes (|diff| > CHANGE, any frame) - where it acts
  - heat.png: the first frame, the changed pixels in red by how often they change - a wall that
    sways, a curtain's halo, motion off its object shows here
  - flicker: inside the changing pixels, the frame-to-frame jumps of the difference against its
    smooth motion (mean |second difference| / mean |first difference|): ~0.1-0.3 for smooth
    motion, toward 1 and above when neighbouring Gaussians jump out of step (white-noise phases)
  - motion p95: how much the moving pixels change (grey levels), so a fix cannot quietly weaken it

Writes <out_dir>/motion_diff.json and heat.png.
"""
import argparse
import json
import os

import cv2
import numpy as np

CHANGE = 6          # grey levels: a pixel the motion changes (encoder noise stays under ~3)


def frames(path: str, every: int):
    cap = cv2.VideoCapture(path)
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % every == 0:
            yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), f
        k += 1
    cap.release()


def measure(with_motion: str, without: str, every: int = 1) -> tuple[dict, np.ndarray]:
    a, b = frames(with_motion, every), frames(without, every)
    count = None
    first = None
    d_prev = d_prev2 = None
    jump, step, n, mags = 0.0, 0.0, 0, []
    for (ga, fa), (gb, _) in zip(a, b):
        d = ga - gb
        if count is None:
            count = np.zeros(d.shape, np.int32)
            first = fa.copy()
        moving = np.abs(d) > CHANGE
        count += moving
        if moving.any():
            mags.append(float(np.percentile(np.abs(d[moving]), 95)))
        if d_prev is not None and d_prev2 is not None:
            m = moving | (np.abs(d_prev) > CHANGE)
            if m.any():
                step += float(np.abs(d - d_prev)[m].mean())
                jump += float(np.abs(d - 2 * d_prev + d_prev2)[m].mean())
                n += 1
        d_prev2, d_prev = d_prev, d
    if count is None:
        raise SystemExit("no frames")
    frames_n = max(1, len(mags))
    heat = first.copy()
    share = count.astype(np.float32) / max(1, int(count.max()))
    red = np.zeros_like(heat)
    red[..., 2] = 255
    w = np.clip(share * 1.5, 0, 0.85)[..., None]
    heat = (heat * (1 - w) + red * w).astype(np.uint8)
    out = {"moving_share": round(float((count > 0).mean()), 5),
           "moving_share_often": round(float((count > 0.25 * frames_n).mean()), 5),
           "flicker": round(jump / step, 3) if step > 0 else None,
           "motion_p95": round(float(np.median(mags)), 2) if mags else 0.0,
           "frames": frames_n}
    return out, heat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("with_motion")
    ap.add_argument("without")
    ap.add_argument("out_dir")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res, heat = measure(a.with_motion, a.without, a.every)
    cv2.imwrite(os.path.join(a.out_dir, "heat.png"), heat)
    json.dump(res, open(os.path.join(a.out_dir, "motion_diff.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
