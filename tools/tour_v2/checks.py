"""Tour v2 checks and deliverables (Step 5).

  python -m tools.tour_v2.checks all --out-dir out/tour_v2 --images-dir <dir> \\
      --reference out/reference/liked_baseline.mp4 --baseline fixtures/downing/baseline_plan.json

Camera numbers come from the plans through the renderer's semantics: ken_burns as
remotion/src/KenBurnsShot.tsx (linear easing, cx/cy linear, zoom log-interpolated, roll 0 here)
and keyframe_path through modules/camera_path.py (the renderer's mirror is parity-tested).
"""
import argparse
import hashlib
import json
import math
import os
import subprocess

import cv2
import numpy as np

from modules.camera_path import KeyframePath, base_window, margin_out_px, source_px_per_output_px
from modules.plan_io import parse_plan
from modules.timeline import compute_timeline

LABELS = {"v0_baseline": "v0  recovered baseline (dissolves)", "v1_cuts": "v1  baseline moves, hard cuts",
          "v2_moves": "v2  multi-step moves, hard cuts"}


# ---------------------------------------------------------------------------------------------
# per-frame cameras
# ---------------------------------------------------------------------------------------------

def shot_cameras(plan_dict: dict) -> list[dict]:
    plan = parse_plan(plan_dict)
    tl = compute_timeline(plan)
    out = []
    for shot, t in zip(plan.shots, tl):
        n = t["render_frames"]
        cam = shot.camera
        if cam.type == "keyframe_path":
            frames = KeyframePath([k.model_dump() for k in cam.keyframes]).frames(n)
        else:
            assert cam.easing == "linear" and not cam.start.roll_deg and not cam.end.roll_deg
            a, b = cam.start, cam.end
            frames = []
            for f in range(n):
                p = f / max(1, n - 1)
                frames.append((a.cx + (b.cx - a.cx) * p, a.cy + (b.cy - a.cy) * p, a.zoom * (b.zoom / a.zoom) ** p))
        out.append({"shot_id": shot.shot_id, "source": (shot.source.width, shot.source.height), "frames": frames,
                    "render_from": t["render_from"], "nominal": [t["nominal_start"], t["nominal_end"]],
                    "fade_frames": t["fade_frames"], "type": cam.type, "move": cam.move or cam.preset,
                    "keyframes": [k.model_dump() for k in cam.keyframes] if cam.keyframes else None,
                    "duration_s": shot.duration_s})
    return out


def grid_speed(c0, c1, src, out_w=1920, out_h=1080, n=9) -> float:
    """Mean on-screen displacement (output px) of an n x n grid of image points between two cameras."""
    sw, sh = src
    bw, _ = base_window(sw, sh, out_w, out_h)
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, n) * out_w, np.linspace(0.05, 0.95, n) * out_h)
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    k0, k1 = bw * sw / c0[2] / out_w, bw * sw / c1[2] / out_w
    src_pts = np.stack([c0[0] * sw + (pts[:, 0] - out_w / 2) * k0, c0[1] * sh + (pts[:, 1] - out_h / 2) * k0], 1)
    back = np.stack([(src_pts[:, 0] - c1[0] * sw) / k1 + out_w / 2, (src_pts[:, 1] - c1[1] * sh) / k1 + out_h / 2], 1)
    return float(np.mean(np.linalg.norm(back - pts, axis=1)))


def shot_rows(shots: list[dict]) -> list[dict]:
    rows = []
    for s in shots:
        fr, (sw, sh) = s["frames"], s["source"]
        margins = [margin_out_px(cx, cy, z, sw, sh, 1920, 1080) for cx, cy, z in fr]
        spp = [source_px_per_output_px(z, sw, sh, 1920, 1080) for _, _, z in fr]
        speeds = [grid_speed(fr[i], fr[i + 1], s["source"]) for i in range(len(fr) - 1)]
        inner = speeds[2:-2]
        i_min = int(np.argmin(inner)) + 2
        zs = [z for _, _, z in fr]
        rows.append({"shot": s["shot_id"], "move": s["move"], "type": s["type"],
                     "frames": [s["render_from"], s["render_from"] + len(fr) - 1], "duration_s": s["duration_s"],
                     "zoom_range": [round(min(zs), 4), round(max(zs), 4)],
                     "min_margin_out_px": round(min(margins), 3), "min_margin_frame": s["render_from"] + int(np.argmin(margins)),
                     "min_spp": round(min(spp), 4), "min_spp_frame": s["render_from"] + int(np.argmin(spp)),
                     "speed_start_px_f": round(speeds[0], 3), "speed_end_px_f": round(speeds[-1], 3),
                     "min_speed_inside_px_f": round(min(inner), 4), "min_speed_frame": s["render_from"] + i_min,
                     "max_speed_px_f": round(max(speeds), 3), "keyframes": s["keyframes"],
                     "speed_profile_every_5": [round(v, 3) for v in speeds[::5]]})
    return rows


# ---------------------------------------------------------------------------------------------
# video measurements
# ---------------------------------------------------------------------------------------------

def ffprobe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt,sample_rate,channels"
                          ":format=duration", "-of", "json", path], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    return {"duration_s": float(d["format"]["duration"]), "frames": int(v["nb_read_frames"]), "fps": v["r_frame_rate"],
            "resolution": f"{v['width']}x{v['height']}", "codec": v["codec_name"], "pix_fmt": v["pix_fmt"],
            "audio": [{"codec": s["codec_name"], "sample_rate": s.get("sample_rate"), "channels": s.get("channels")} for s in a]}


def audio_level(path: str) -> str:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-map", "0:a", "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return " ".join(l.split("] ")[-1] for l in err.splitlines() if "mean_volume" in l or "max_volume" in l)


def framemd5(path: str) -> list[str]:
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-map", "0:v", "-f", "framemd5", "-"],
                         capture_output=True, text=True, check=True).stdout
    return [l.split(",")[-1].strip() for l in out.splitlines() if l and not l.startswith("#")]


def determinism(a: str, b: str) -> dict:
    ma, mb = framemd5(a), framemd5(b)
    diff = [i for i, (x, y) in enumerate(zip(ma, mb)) if x != y]
    return {"frames": [len(ma), len(mb)], "identical": len(ma) == len(mb) and not diff, "n_differing": len(diff),
            "digest": hashlib.sha256("".join(ma).encode()).hexdigest()[:16]}


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Grayscale SSIM, Gaussian window 11x11 sigma 1.5 (Wang et al. 2004)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = g(a), g(b)
    va, vb, cov = g(a * a) - ma * ma, g(b * b) - mb * mb, g(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))).mean())


def read_frames(path: str, wanted: set[int], gray=True) -> dict[int, np.ndarray]:
    cap, got, i = cv2.VideoCapture(path), {}, 0
    last = max(wanted)
    while i <= last:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            got[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if gray else f
        i += 1
    cap.release()
    return got


def ssim_per_second(ref: str, other: str, fps: int = 30, n_frames: int = 2100) -> dict:
    idx = set(range(fps // 2, n_frames, fps))  # middle frame of each second
    fa, fb = read_frames(ref, idx), read_frames(other, idx)
    vals = [(i, ssim(fa[i], fb[i])) for i in sorted(idx) if i in fa and i in fb]
    return {"frames_compared": [i for i, _ in vals], "per_second": [round(v, 4) for _, v in vals],
            "mean": round(float(np.mean([v for _, v in vals])), 4), "min": round(min(v for _, v in vals), 4)}


def shot_identity(video: str, shots: list[dict], images: dict[str, np.ndarray], offset: int = 9) -> list[dict]:
    """Which photo is on screen `offset` frames before and after each nominal boundary (ORB match
    count against the two photos either side)."""
    orb, bf = cv2.ORB_create(2000), cv2.BFMatcher(cv2.NORM_HAMMING)
    desc = {k: orb.detectAndCompute(v, None)[1] for k, v in images.items()}
    bounds = [s["nominal"][0] for s in shots[1:]]
    wanted = {b + d for b in bounds for d in (-offset, offset)}
    fr = read_frames(video, wanted)
    rows = []
    for k, b in enumerate(bounds):
        prev_id, next_id = shots[k]["shot_id"], shots[k + 1]["shot_id"]
        row = {"boundary": b, "expected": [prev_id, next_id]}
        for d, tag in ((-offset, "before"), (offset, "after")):
            _, dsc = orb.detectAndCompute(fr[b + d], None)
            score = {}
            for sid in (prev_id, next_id):
                m = bf.knnMatch(dsc, desc[sid], k=2)
                score[sid] = sum(1 for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)
            row[tag] = max(score, key=score.get)
            row[f"{tag}_matches"] = score
        row["ok"] = row["before"] == prev_id and row["after"] == next_id
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------
# deliverables
# ---------------------------------------------------------------------------------------------

def compare_grid(out_dir: str) -> dict:
    out = os.path.join(out_dir, "compare_grid.mp4")
    inputs, filters = [], []
    for i, v in enumerate(LABELS):
        inputs += ["-i", os.path.join(out_dir, f"{v}.mp4")]
        filters.append(f"[{i}:v]scale=960:540,drawtext=text='{LABELS[v]}':x=12:y=12:fontsize=24:fontcolor=white:"
                       f"box=1:boxcolor=black@0.6:boxborderw=6,drawtext=text='f %{{n}}':x=w-tw-12:y=12:fontsize=24:"
                       f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[v{i}]")
    fc = ";".join(filters) + ";[v0][v1][v2]vstack=inputs=3[out]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[out]",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", out], check=True)
    return ffprobe(out)


def contact_sheet(out_dir: str, shots: list[dict]) -> str:
    tw, th = 480, 270
    wanted = {}
    for s in shots:
        a, b = s["render_from"], s["render_from"] + len(s["frames"]) - 1
        wanted[a] = f"{s['shot_id']} {s['move']} first f{a}"
        wanted[b] = f"{s['shot_id']} {s['move']} last f{b}"
    fr = read_frames(os.path.join(out_dir, "v2_moves.mp4"), set(wanted), gray=False)
    tiles = []
    for f in sorted(wanted):
        t = cv2.resize(fr[f], (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (tw, 22), (0, 0, 0), -1)
        cv2.putText(t, wanted[f], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    sheet = np.vstack([np.hstack([rows[i], rows[i + 1]]) if i + 1 < len(rows) else
                       np.hstack([rows[i], np.zeros_like(rows[i])]) for i in range(0, len(rows), 2)])
    path = os.path.join(out_dir, "v2_contact_sheet.png")
    cv2.imwrite(path, sheet)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()
    o = args.out_dir
    plans = {"v0_baseline": json.load(open(args.baseline)),
             "v1_cuts": json.load(open(os.path.join(o, "v1_cuts.plan.json"))),
             "v2_moves": json.load(open(os.path.join(o, "v2_moves.plan.json")))}
    res = {"renders": {}}
    for name, plan in plans.items():
        shots = shot_cameras(plan)
        video = os.path.join(o, f"{name}.mp4")
        rows = shot_rows(shots)
        res["renders"][name] = {
            "plan_hash": plan["provenance"]["plan_hash"], "ffprobe": ffprobe(video), "audio_level": audio_level(video),
            "min_margin_out_px": min(r["min_margin_out_px"] for r in rows),
            "min_margin_where": min(rows, key=lambda r: r["min_margin_out_px"])["min_margin_frame"],
            "frames_checked": sum(len(s["frames"]) for s in shots),
            "min_spp": min(r["min_spp"] for r in rows), "shots": rows}
        print("done", name, flush=True)
        if name == "v2_moves":
            v2_shots = shots
        if name == "v0_baseline":
            v0_shots = shots
    res["reference"] = {"ffprobe": ffprobe(args.reference), "audio_level": audio_level(args.reference)}
    res["determinism_v2"] = determinism(os.path.join(o, "v2_moves.mp4"), os.path.join(o, "determinism", "v2_moves.run2.mp4"))
    res["v0_vs_reference_ssim"] = ssim_per_second(args.reference, os.path.join(o, "v0_baseline.mp4"))
    images = {}
    for s in plans["v0_baseline"]["shots"]:
        g = cv2.imread(os.path.join(args.images_dir, s["image_uri"].split("/")[-1]), cv2.IMREAD_GRAYSCALE)
        images[s["shot_id"]] = cv2.resize(g, (960, 640), interpolation=cv2.INTER_AREA)
    res["boundaries"] = {"reference": shot_identity(args.reference, v0_shots, images),
                         "v0_baseline": shot_identity(os.path.join(o, "v0_baseline.mp4"), v0_shots, images)}
    res["compare_grid"] = compare_grid(o)
    res["contact_sheet"] = contact_sheet(o, v2_shots)
    json.dump(res, open(os.path.join(o, "checks.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("determinism_v2", "v0_vs_reference_ssim")}, indent=1))


if __name__ == "__main__":
    main()
