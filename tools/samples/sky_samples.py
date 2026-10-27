"""Sky motion sample clips (tour_worker/motion/sky.py), for choosing which cloud effects to keep.

  python -m tools.samples.sky_samples <folder of pXX.jpg + pXX_sky.png> <out folder> [--seconds 5] [--strength 1]

Per photo, a 2x2 grid of the same slow push-in, each cell one combination of effects, labelled.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess

import cv2
import numpy as np

from tour_worker.motion.sky import SkyMotion

CELLS = [("drift", ("drift",)), ("billow", ("billow",)), ("drift + billow", ("drift", "billow")),
         ("drift + billow + cloud shadows", ("drift", "billow", "shadows"))]
# --ladder: how fast should the clouds go (share of the photo width per second, top clouds)
LADDER = [("1% / s", ("drift", "billow"), 0.01), ("2% / s", ("drift", "billow"), 0.02),
          ("3% / s", ("drift", "billow"), 0.03), ("2% / s + cloud shadows", ("drift", "billow", "shadows"), 0.02)]
# --shadows: how strong should the cloud shadows be (all at the chosen 2% / s drift)
SHADOWS = [("2% / s, no shadows", ("drift", "billow"), 0.02, 1.0),
           ("cloud shadows - soft", ("drift", "billow", "shadows"), 0.02, 0.6),
           ("cloud shadows - medium", ("drift", "billow", "shadows"), 0.02, 1.0),
           ("cloud shadows - strong", ("drift", "billow", "shadows"), 0.02, 1.5)]
CELL_W, CELL_H, FPS = 960, 540, 30


def camera(img: np.ndarray, t: float, dur: float) -> np.ndarray:
    """A slow 16:9 push-in on the photo's centre (zoom 1.00 -> 1.06), eased."""
    h, w = img.shape[:2]
    u = t / dur
    u = u * u * (3 - 2 * u)
    z = 1.0 + 0.06 * u
    cw, ch = w / z, w / z * 9 / 16
    x0, y0 = (w - cw) / 2, (h - ch) / 2
    m = np.array([[CELL_W / cw, 0, -x0 * CELL_W / cw], [0, CELL_H / ch, -y0 * CELL_H / ch]], np.float32)
    return cv2.warpAffine(img, m, (CELL_W, CELL_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (12 + 15 * len(text), 40), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def render(photo: str, sky_png: str, out: str, seconds: float, strength: float, ladder: bool = False,
           shadows: bool = False) -> None:
    bgr, coarse = cv2.imread(photo), cv2.imread(sky_png, cv2.IMREAD_GRAYSCALE)
    if shadows:
        motions = [(name, SkyMotion(bgr, coarse, effects=fx, strength=k, speed=sp, duration=seconds))
                   for name, fx, sp, k in SHADOWS]
    elif ladder:
        motions = [(name, SkyMotion(bgr, coarse, effects=fx, strength=strength, speed=sp, duration=seconds))
                   for name, fx, sp in LADDER]
    else:
        motions = [(name, SkyMotion(bgr, coarse, effects=fx, strength=strength, duration=seconds))
                   for name, fx in CELLS]
    n = int(round(seconds * FPS))
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{2 * CELL_W}x{2 * CELL_H}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", out],
                            stdin=subprocess.PIPE)
    for f in range(n):
        t = f / FPS
        cells = [label(camera(m.frame(t), t, seconds), name) for name, m in motions]
        grid = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
        proc.stdin.write(grid.tobytes())
    proc.stdin.close()
    proc.wait()
    # the refined sky (alpha) for checking the skyline
    a = motions[0][1].alpha
    cv2.imwrite(out.replace(".mp4", "_sky_alpha.jpg"),
                (bgr * 0.35 + np.dstack([a * 0, a * 120, a * 255]) * 0.65).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--ladder", action="store_true", help="compare drift speeds instead of effects")
    ap.add_argument("--shadows", action="store_true", help="compare cloud shadow strengths")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for photo in sorted(glob.glob(os.path.join(a.src, "p*.jpg"))):
        sky = photo.replace(".jpg", "_sky.png")
        if not os.path.exists(sky):
            continue
        name = os.path.splitext(os.path.basename(photo))[0]
        out = os.path.join(a.out, f"sky_{name}{'_speeds' if a.ladder else '_shadows' if a.shadows else ''}"
                                  f"{'' if a.strength == 1 else f'_x{a.strength:g}'}.mp4")
        render(photo, sky, out, a.seconds, a.strength, a.ladder, a.shadows)
        print("wrote", out)


if __name__ == "__main__":
    main()
