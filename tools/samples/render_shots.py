"""Motion samples from a rendered tour's workdir: chosen shots, camera HELD at the middle of
each shot's move (so only the perceptual motion shows), 5 s each, then one labelled reel.

  python -m tools.samples.render_shots <workdir> <out dir> <shot_id>[=label] ...
"""
import copy
import json
import os
import subprocess
import sys

from tour_worker.render.camera_path import KeyframePath
from tour_worker.contract.plan import compute_plan_hash, parse_plan
from tour_worker.render.frames import ShotWarp
from tour_worker.render.timeline import compute_timeline

DUR = 5.0


def main(argv):
    wd, out_dir = argv[0], argv[1]
    os.makedirs(out_dir, exist_ok=True)
    raw = json.load(open(os.path.join(wd, "plan.json")))
    clips = []
    for spec in argv[2:]:
        sid, _, label = spec.partition("=")
        shot = copy.deepcopy(next(s for s in raw["shots"] if s["shot_id"] == sid))
        cam = shot["camera"]
        if cam["type"] == "keyframe_path":
            cx, cy, z = KeyframePath(cam["keyframes"]).at(0.5)
        else:
            cx, cy, z = 0.5, 0.5, 1.0
        k = {"t": 0, "cx": cx, "cy": cy, "zoom": z}
        shot["camera"] = {"type": "keyframe_path", "move": "hold", "keyframes": [k, dict(k, t=1)]}
        shot["duration_s"] = DUR
        plan = dict(copy.deepcopy(raw), shots=[shot], total_duration_s=DUR, music=None, plan_version="1.4")
        plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
        pl = parse_plan(plan)
        warn = []
        sw = ShotWarp(pl, pl.shots[0], compute_timeline(pl)[0]["render_frames"], os.path.join(wd, "images"), warn)
        name = f"{sid}_{label}" if label else sid
        out = os.path.join(out_dir, f"{name}.mp4")
        ow, oh, fps = pl.output.width, pl.output.height, pl.output.fps
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", f"{ow}x{oh}", "-r", str(fps), "-i", "-", "-vf",
                              f"drawtext=text='{name}':x=24:y=24:fontsize=40:fontcolor=yellow:box=1:boxcolor=black@0.5",
                              "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        for f in range(sw.n):
            p.stdin.write(sw.frame(f).tobytes())
        p.stdin.close()
        p.wait()
        moved = ", ".join(f"{r.spec.label} {sw.motion_px[r.spec.id]:.1f}px" for r in sw.regions)
        print(f"{name}: {moved}  {[w for w in warn if 'dropped' in w]}", flush=True)
        clips.append(out)
    listing = os.path.join(out_dir, "reel.txt")
    with open(listing, "w") as f:
        f.writelines(f"file '{os.path.abspath(c)}'\n" for c in clips)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
                    "-c", "copy", os.path.join(out_dir, "motion_samples_reel.mp4")], check=True)
    print("reel:", os.path.join(out_dir, "motion_samples_reel.mp4"))


if __name__ == "__main__":
    main(sys.argv[1:])
