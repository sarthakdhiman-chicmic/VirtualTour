"""Recover a ken_burns plan from a rendered tour by registering its frames to the source photos.

No plan file on disk reproduces the liked baseline render, so its cameras are measured: every
sampled frame is matched to its shot's photo (ORB features + RANSAC similarity fit), giving the
window centre and zoom for that frame (optionally refined with ECC, --ecc; measured on the liked
baseline, the ECC fit is tighter but renders score lower SSIM against it, see report.md). Per shot, cx and cy are fitted linear in the renderer's
progress t and ln(zoom) linear in t (renderer: linear easing, log-space zoom), using only frames
outside the dissolves. t follows the renderer's timeline (modules/timeline.py): a shot's camera
runs over its render span [render_from, render_from + render_frames), which includes the fade
overlap.

  python -m tools.tour_v2.recover_baseline --video out/reference/liked_baseline.mp4 \\
      --skeleton <plan with shot order, durations, transitions> --images-dir <dir> \\
      --out fixtures/downing/baseline_plan.json --fits-out out/tour_v2/baseline_fits.json
"""
import argparse
import copy
import json
import math
import os

import cv2
import numpy as np

from modules.plan_io import compute_plan_hash, parse_plan
from modules.timeline import compute_timeline

OUT_W, OUT_H = 1920, 1080
SAMPLE_STEP = 6


def base_window_w(src_w: int, src_h: int) -> float:
    a_src, a_out = src_w / src_h, OUT_W / OUT_H
    return a_out / a_src if a_src >= a_out else 1.0


def register(frame_gray, ref, orb, matcher, ecc=False):
    k, d = orb.detectAndCompute(frame_gray, None)
    ki, di, (h, w) = ref[:3]
    pairs = matcher.knnMatch(d, di, k=2)
    good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.75 * b.distance]
    src = np.float32([k[a.queryIdx].pt for a in good])
    dst = np.float32([ki[a.trainIdx].pt for a in good])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    M, ecc_ok = refine_ecc(frame_gray, ref[3], M) if ecc else (M, False)
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    c = M @ np.array([OUT_W / 2, OUT_H / 2, 1.0])
    zoom = base_window_w(w, h) * w / (scale * OUT_W)
    return {"cx": c[0] / w, "cy": c[1] / h, "zoom": zoom, "rot_deg": rot, "inliers": int(inliers.sum()),
            "ecc": ecc_ok}


def refine_ecc(frame_gray, photo_half, M):
    """Sub-pixel refinement of the frame->photo affine M with ECC (intensity alignment, not
    features), run at half resolution to keep memory low. Returns (M, refined?)."""
    S = np.diag([2.0, 2.0, 1.0])
    M_half = (np.linalg.inv(S) @ np.vstack([M, [0, 0, 1]]) @ S)[:2].astype(np.float32)
    frame_half = cv2.resize(frame_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
        _, M_half = cv2.findTransformECC(frame_half, photo_half, M_half, cv2.MOTION_AFFINE, crit, None, 3)
    except cv2.error:
        return M, False
    return (S @ np.vstack([M_half, [0, 0, 1]]) @ np.linalg.inv(S))[:2], True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--skeleton", required=True, help="plan (or tour_render request) with the shot list")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fits-out", required=True)
    ap.add_argument("--ecc", action="store_true", help="refine each registration with ECC")
    args = ap.parse_args()

    raw = json.load(open(args.skeleton))
    plan_dict = copy.deepcopy(raw.get("plan", raw))
    timeline = compute_timeline(parse_plan(plan_dict))
    shots = plan_dict["shots"]

    # frames to sample per shot: outside both dissolves
    wanted: dict[int, int] = {}
    for k, t in enumerate(timeline):
        lo = t["nominal_start"] + math.ceil(t["fade_frames"] / 2) + 2
        nxt = timeline[k + 1]["fade_frames"] if k + 1 < len(timeline) else 0
        hi = t["nominal_end"] - math.ceil(nxt / 2) - 2
        for f in range(lo, hi + 1, SAMPLE_STEP):
            wanted[f] = k

    orb, matcher = cv2.ORB_create(4000), cv2.BFMatcher(cv2.NORM_HAMMING)
    refs = {}
    for s in shots:
        name = s["image_uri"].split("/")[-1]
        g = cv2.imread(os.path.join(args.images_dir, name), cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(g, None)
        half = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
        refs[s["shot_id"]] = (kp, des, g.shape, half)

    samples: list[list[dict]] = [[] for _ in shots]
    cap, i = cv2.VideoCapture(args.video), 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in wanted:
            k = wanted[i]
            r = register(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), refs[shots[k]["shot_id"]], orb, matcher, args.ecc)
            t = timeline[k]
            r["f"], r["t"] = i, (i - t["render_from"]) / max(1, t["render_frames"] - 1)
            samples[k].append(r)
        i += 1
    cap.release()

    fits = []
    for k, s in enumerate(shots):
        rows = samples[k]
        t = np.array([r["t"] for r in rows])
        cam, fit = {}, {"shot_id": s["shot_id"], "n_samples": len(rows), "frames": [rows[0]["f"], rows[-1]["f"]],
                        "min_inliers": min(r["inliers"] for r in rows),
                        "ecc_refined": sum(r["ecc"] for r in rows),
                        "rot_deg_range": [round(min(r["rot_deg"] for r in rows), 4), round(max(r["rot_deg"] for r in rows), 4)]}
        for ch in ("cx", "cy", "zoom"):
            y = np.array([r[ch] for r in rows])
            yy = np.log(y) if ch == "zoom" else y
            slope, icpt = np.polyfit(t, yy, 1)
            resid = yy - (slope * t + icpt)
            v0, v1 = icpt, icpt + slope
            if ch == "zoom":
                v0, v1 = math.exp(v0), math.exp(v1)
            cam[ch] = (round(float(v0), 6), round(float(v1), 6))
            fit[f"{ch}_max_residual"] = round(float(np.abs(resid).max()), 6)
            # curvature check: a quadratic term would mean the move was eased, not linear
            fit[f"{ch}_quadratic_coef"] = round(float(np.polyfit(t, yy, 2)[0]), 6)
        # the planner places windows exactly on the image edge; a fit can overshoot by a hair, so
        # clamp each end into the image and record the shift
        bw = base_window_w(s["source"]["width"], s["source"]["height"])
        bh = bw * (s["source"]["width"] / s["source"]["height"]) / (OUT_W / OUT_H)
        ends, fit["clamp_shift"] = [], []
        for j in (0, 1):
            z = cam["zoom"][j]
            hw, hh = bw / (2 * z), bh / (2 * z)
            cx = min(max(cam["cx"][j], hw), 1 - hw)
            cy = min(max(cam["cy"][j], hh), 1 - hh)
            fit["clamp_shift"].append([round(cx - cam["cx"][j], 6), round(cy - cam["cy"][j], 6)])
            ends.append({"cx": round(cx, 6), "cy": round(cy, 6), "zoom": z, "roll_deg": 0.0})
        s["camera"] = {"type": "ken_burns", "easing": "linear", "preset": "recovered",
                       "start": ends[0], "end": ends[1]}
        fit["camera"] = s["camera"]
        fits.append(fit)

    plan_dict["provenance"] = {**plan_dict.get("provenance", {}),
                               "recovered_from": os.path.basename(args.video),
                               "recovery": "cameras measured by frame registration (tools/tour_v2/recover_baseline.py)"}
    plan_dict["provenance"]["plan_hash"] = compute_plan_hash(plan_dict)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.fits_out)), exist_ok=True)
    json.dump(plan_dict, open(args.out, "w"), indent=1)
    json.dump({"timeline": timeline, "fits": fits}, open(args.fits_out, "w"), indent=1)
    for f in fits:
        c = f["camera"]
        print(f["shot_id"], f["n_samples"], f["frames"], "inl>=", f["min_inliers"], "rot", f["rot_deg_range"],
              "res", f["cx_max_residual"], f["cy_max_residual"], f["zoom_max_residual"],
              "start", c["start"], "end", c["end"])


if __name__ == "__main__":
    main()
