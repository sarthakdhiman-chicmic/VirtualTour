"""Fire motion samples for review: every lit fireplace -> a 4 s held clip (fireplace in context),
the planner's strength rule (0.12 x flame height, 1..20 px), then one labelled reel.

  python -m tools.fireplace_samples.render_samples <out dir> <name>=<photo>:<fireplace mask png> ...
"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from modules.camera_path import base_window
from modules.perceptual.effects import mask_relpath
from modules.perceptual.fire import fire_regions
from modules.plan_io import compute_plan_hash, parse_plan
from modules.prewarp import ShotWarp
from modules.timeline import compute_timeline

OW, OH, FPS, DUR = 1920, 1080, 30, 4.0


def clip(name: str, photo: str, fp_png: str, out_dir: str) -> tuple[str, str]:
    img = cv2.imread(photo)
    m = cv2.imread(fp_png, 0)
    W, H = m.shape[1], m.shape[0]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    regs = fire_regions(img, [{"object_id": "fp", "class_name": "fireplace", "confidence": 0.99,
                               "bbox": [0, 0, W, H], "mask": m, "space": [W, H]}])
    if not regs:
        return "", f"{name}: not lit"
    r = regs[0]
    x1, y1, x2, y2 = r["bbox"]
    amp = round(min(max(0.12 * (y2 - y1), 1.0), 20.0), 3)
    ys, xs = np.nonzero(m > 127)
    bw, bh = base_window(W, H, OW, OH)
    z = min(1.8, max(1.0, bw / min(1.0, 2.2 * (xs.max() - xs.min()) / W)))   # fireplace + context
    cx = min(max(xs.mean() / W, bw / z / 2 + 0.005), 1 - bw / z / 2 - 0.005)
    cy = min(max(ys.mean() / H, bh / z / 2 + 0.005), 1 - bh / z / 2 - 0.005)
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(f"{td}/images/masks")
        cv2.imwrite(f"{td}/images/x.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        cv2.imwrite(os.path.join(td, "images", mask_relpath("s01", "fire_000")), r["mask"])
        plan = {"plan_version": "1.4", "listing_id": "fire_samples", "mode": "cinematic", "preset": "p",
                "output": {"width": OW, "height": OH, "fps": FPS, "codec": "h264"}, "total_duration_s": DUR,
                "shots": [{"shot_id": "s01", "order": 1, "asset_id": "a", "image_uri": "x.jpg",
                           "source": {"width": W, "height": H}, "room_key": "r", "room_type": "living_room",
                           "narrative_stage": "living", "role": "hero", "duration_s": DUR,
                           "camera": {"type": "keyframe_path", "move": "hold", "keyframes": [
                               {"t": 0, "cx": cx, "cy": cy, "zoom": z}, {"t": 1, "cx": cx, "cy": cy, "zoom": z}]},
                           "transition_in": {"type": "cut", "duration_s": 0.0},
                           "motion_regions": [{"id": "fire_000", "effect": "fire_flicker", "mask_uri": "s3://x/m.png",
                                               "bbox": [x1 / W, y1 / H, x2 / W, y2 / H], "amplitude_px": amp,
                                               "freq_hz": 1.0, "anchor": "bottom", "seed": 7, "label": "fire"}]}],
                "warnings": [], "provenance": {"scoring_config_version": "0", "planner_config_version": "0"}}
        plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
        pl = parse_plan(plan)
        sw = ShotWarp(pl, pl.shots[0], compute_timeline(pl)[0]["render_frames"], f"{td}/images")
        out = os.path.join(out_dir, f"{name}.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
               "-r", str(FPS), "-i", "-", "-vf", f"drawtext=text='{name}':x=24:y=24:fontsize=40:fontcolor=yellow:box=1:boxcolor=black@0.5",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in range(sw.n):
            p.stdin.write(sw.frame(f).tobytes())
        p.stdin.close()
        p.wait()
    return out, (f"{name}: flame {x2 - x1}x{y2 - y1} px of {W}x{H}, strength {amp} px, "
                 f"on screen p95 {sw.motion_px.get('fire_000', 0.0):.1f} px")


def main(argv):
    out_dir = argv[0]
    os.makedirs(out_dir, exist_ok=True)
    clips = []
    for spec in argv[1:]:
        name, rest = spec.split("=", 1)
        photo, fp = rest.rsplit(":", 1)
        path, note = clip(name, photo, fp, out_dir)
        print(note, flush=True)
        if path:
            clips.append(path)
    listing = os.path.join(out_dir, "reel.txt")
    with open(listing, "w") as f:
        f.writelines(f"file '{os.path.abspath(c)}'\n" for c in clips)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
                    "-c", "copy", os.path.join(out_dir, "fire_samples_reel.mp4")], check=True)
    print("reel:", os.path.join(out_dir, "fire_samples_reel.mp4"))


if __name__ == "__main__":
    main(sys.argv[1:])
