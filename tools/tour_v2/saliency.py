"""Saliency targets per photo (tour v2, Step 2): spectral-residual saliency (Hou & Zhang 2007).

opencv-contrib (cv2.saliency) is not installed and no new dependencies are allowed, so this is a
NumPy/OpenCV port of the StaticSaliencySpectralResidual steps: grayscale, resize to 64x64, FFT,
log amplitude minus its 3x3 box-filtered copy (the spectral residual), inverse FFT with the
original phase, squared magnitude, Gaussian blur, normalise to [0, 1]. The map is then resized to
ANALYSIS_W wide and smoothed again (SMOOTH_SIGMA_FRAC of the width).

Derived per photo (after EXIF orientation):
  peak    centroid (x, y), normalised, of the top 5% of saliency pixels, ignoring a 5% border
  side    "left" or "right": the half with more saliency mass; mass_ratio = larger / smaller
  spread  fraction of pixels above the mean saliency (high = busy wide view, low = one subject)

  python -m tools.tour_v2.saliency --images-dir <dir> --plan fixtures/downing/baseline_plan.json \\
      --out fixtures/downing/saliency.json --sheet out/tour_v2/saliency_sheet.png
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

SR_SIZE = 64
SR_BOX = 3
SR_BLUR_KSIZE, SR_BLUR_SIGMA = 5, 8.0
ANALYSIS_W = 480
SMOOTH_SIGMA_FRAC = 0.02
TOP_FRACTION = 0.05
BORDER_FRACTION = 0.05


def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        return np.asarray(im, dtype=np.float32) / 255.0


def spectral_residual(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (SR_SIZE, SR_SIZE), interpolation=cv2.INTER_AREA)
    spec = np.fft.fft2(small)
    log_amp = np.log(np.abs(spec) + 1e-12)
    residual = log_amp - cv2.blur(log_amp, (SR_BOX, SR_BOX))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spec)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (SR_BLUR_KSIZE, SR_BLUR_KSIZE), SR_BLUR_SIGMA)
    h = round(gray.shape[0] * ANALYSIS_W / gray.shape[1])
    sal = cv2.resize(sal, (ANALYSIS_W, h), interpolation=cv2.INTER_LINEAR)
    sal = cv2.GaussianBlur(sal, (0, 0), SMOOTH_SIGMA_FRAC * ANALYSIS_W)
    sal -= sal.min()
    return sal / max(float(sal.max()), 1e-12)


def targets(sal: np.ndarray) -> dict:
    h, w = sal.shape
    by, bx = round(h * BORDER_FRACTION), round(w * BORDER_FRACTION)
    inner = sal[by:h - by, bx:w - bx]
    thr = np.quantile(inner, 1 - TOP_FRACTION)
    ys, xs = np.nonzero(inner >= thr)
    peak = ((xs.mean() + bx + 0.5) / w, (ys.mean() + by + 0.5) / h)
    left, right = float(sal[:, : w // 2].sum()), float(sal[:, w - w // 2:].sum())
    side = "left" if left > right else "right"
    return {"peak": {"x": round(float(peak[0]), 4), "y": round(float(peak[1]), 4)},
            "side": side, "mass_ratio": round(max(left, right) / max(min(left, right), 1e-12), 4),
            "mass_left": round(left / (left + right), 4),
            "spread": round(float((sal > sal.mean()).mean()), 4),
            "top5_threshold": round(float(thr), 4)}


def overlay(path: str, sal: np.ndarray, t: dict, label: str, tile_w: int = 480) -> np.ndarray:
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    tile_h = round(img.shape[0] * tile_w / img.shape[1])
    img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    heat = cv2.applyColorMap((cv2.resize(sal, (tile_w, tile_h)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    b = round(tile_w * BORDER_FRACTION)
    cv2.rectangle(out, (b, round(tile_h * BORDER_FRACTION)), (tile_w - b, tile_h - round(tile_h * BORDER_FRACTION)),
                  (255, 255, 255), 1)
    cv2.line(out, (tile_w // 2, 0), (tile_w // 2, tile_h), (255, 255, 255), 1)
    px, py = round(t["peak"]["x"] * tile_w), round(t["peak"]["y"] * tile_h)
    cv2.circle(out, (px, py), 10, (255, 255, 255), 2)
    cv2.drawMarker(out, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 14, 2)
    cv2.rectangle(out, (0, 0), (tile_w, 40), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"side {t['side']} x{t['mass_ratio']:.2f}  spread {t['spread']:.3f}  "
                     f"peak ({t['peak']['x']:.2f}, {t['peak']['y']:.2f})",
                (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--plan", required=True, help="to label which shot uses each photo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", required=True)
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    plan = plan.get("plan", plan)
    used = {s["image_uri"].split("/")[-1]: s["shot_id"] for s in plan["shots"]}

    doc = {"method": {"name": "spectral_residual", "port_of": "opencv-contrib StaticSaliencySpectralResidual (NumPy port)",
                      "sr_size": SR_SIZE, "sr_box": SR_BOX, "sr_blur": [SR_BLUR_KSIZE, SR_BLUR_SIGMA],
                      "analysis_width": ANALYSIS_W, "smooth_sigma_frac": SMOOTH_SIGMA_FRAC,
                      "top_fraction": TOP_FRACTION, "border_fraction": BORDER_FRACTION, "exif_transpose": True},
           "photos": {}}
    tiles = []
    for path in sorted(glob.glob(os.path.join(args.images_dir, "imgi_*.jpg"))):
        name = os.path.basename(path)
        sal = spectral_residual(load_gray(path))
        t = targets(sal)
        t["shot_id"] = used.get(name)
        doc["photos"][name] = t
        tiles.append(overlay(path, sal, t, f"{t['shot_id'] or '--'}  {name[:9]}"))
    json.dump(doc, open(args.out, "w"), indent=1)

    cols = 4
    th, tw = max(t.shape[0] for t in tiles), tiles[0].shape[1]
    tiles = [np.pad(t, ((0, th - t.shape[0]), (0, 0), (0, 0))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros((th, tw, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    os.makedirs(os.path.dirname(os.path.abspath(args.sheet)), exist_ok=True)
    cv2.imwrite(args.sheet, sheet)
    for name, t in doc["photos"].items():
        print(f"{t['shot_id'] or '--':<4} {name[:9]} peak ({t['peak']['x']:.3f}, {t['peak']['y']:.3f}) "
              f"side {t['side']:<5} ratio {t['mass_ratio']:.3f} spread {t['spread']:.3f}")


if __name__ == "__main__":
    main()
