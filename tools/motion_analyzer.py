import os
import sys
import argparse
import json
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

# Caveat: Optical flow is unreliable on flat regions and compressed video;
# calibrate on TEXTURED synthetic images and treat real-photo numbers as approximate.

def compute_mask(h: int, w: int) -> np.ndarray:
    """
    Mask out:
    - top 7% of rows
    - bottom 7% of rows
    - top-right corner (16% of rows x 38% of columns)
    Returns boolean array where True indicates valid pixel.
    """
    mask = np.ones((h, w), dtype=bool)
    top_cut = int(round(h * 0.07))
    bot_cut = int(round(h * 0.07))
    tr_rows = int(round(h * 0.16))
    tr_cols = int(round(w * 0.38))

    mask[:top_cut, :] = False
    if bot_cut > 0:
        mask[h - bot_cut:, :] = False
    mask[:tr_rows, w - tr_cols:] = False
    return mask

def fit_affine_flow(
    u: np.ndarray,
    v: np.ndarray,
    xn: np.ndarray,
    yn: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """
    Least-squares fit:
    u = a0 + a1*xn + a2*yn
    v = b0 + b1*xn + b2*yn
    """
    # A has columns [1, xn, yn]
    A = np.column_stack([np.ones_like(xn), xn, yn])
    # Solve via lstsq
    a, _, _, _ = np.linalg.lstsq(A, u, rcond=None)
    b, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
    return float(a[0]), float(a[1]), float(a[2]), float(b[0]), float(b[1]), float(b[2])

def analyze_shot_frames(
    frames_gray_resized: List[np.ndarray],
    save_heatmaps_dir: Optional[str] = None,
    shot_idx: int = 0
) -> Dict[str, Any]:
    """
    Analyze optical flow across a shot of resized grayscale frames.
    Frame size is (h, w), where w = 332.
    Frame pairs (t, t+4), skipping first and last 2 frames.
    """
    num_frames = len(frames_gray_resized)
    if num_frames < 9:
        raise ValueError(f"Shot too short for analysis (needs at least 9 frames, got {num_frames})")

    h, w = frames_gray_resized[0].shape
    mask = compute_mask(h, w)
    y_coords, x_coords = np.nonzero(mask)
    xn = (x_coords - (w / 2.0)) / (w / 2.0)
    yn = (y_coords - (h / 2.0)) / (h / 2.0)

    # Pre-define three horizontal bands within valid rows 7%-93%
    top_band_y0 = int(round(h * 0.07))
    top_band_y1 = int(round(h * 0.38))
    mid_band_y0 = int(round(h * 0.38))
    mid_band_y1 = int(round(h * 0.66))
    bot_band_y0 = int(round(h * 0.66))
    bot_band_y1 = int(round(h * 0.93))

    band_masks = {
        "top": mask & (np.arange(h)[:, None] >= top_band_y0) & (np.arange(h)[:, None] < top_band_y1),
        "mid": mask & (np.arange(h)[:, None] >= mid_band_y0) & (np.arange(h)[:, None] < mid_band_y1),
        "bot": mask & (np.arange(h)[:, None] >= bot_band_y0) & (np.arange(h)[:, None] < bot_band_y1),
    }

    band_coords = {}
    for b_name, b_m in band_masks.items():
        by, bx = np.nonzero(b_m)
        band_coords[b_name] = {
            "by": by, "bx": bx,
            "bxn": (bx - (w / 2.0)) / (w / 2.0),
            "byn": (by - (h / 2.0)) / (h / 2.0)
        }

    # Step loop: start at t=2, pairs (t, t+4), while t+4 <= num_frames - 3
    # (skipping first 2 and last 2 frames)
    t = 2
    steps_tx = []
    steps_ty = []
    steps_zoom = []
    steps_speed = []
    steps_residuals = []

    band_zoom_steps = {"top": [], "mid": [], "bot": []}

    pair_idx = 0
    while t + 4 <= num_frames - 3:
        prev_f = frames_gray_resized[t]
        next_f = frames_gray_resized[t + 4]

        flow = cv2.calcOpticalFlowFarneback(
            prev_f, next_f, None,
            pyr_scale=0.5, levels=5, winsize=21,
            iterations=5, poly_n=7, poly_sigma=1.5, flags=0
        )

        u = flow[..., 0]
        v = flow[..., 1]

        u_m = u[y_coords, x_coords]
        v_m = v[y_coords, x_coords]

        a0, a1, a2, b0, b1, b2 = fit_affine_flow(u_m, v_m, xn, yn)

        tx = a0 / w
        ty = b0 / h
        zoom_step = 0.5 * ((a1 / (w / 2.0)) + (b2 / (h / 2.0)))
        step_spd = math.sqrt(tx**2 + ty**2 + zoom_step**2)

        steps_tx.append(tx)
        steps_ty.append(ty)
        steps_zoom.append(zoom_step)
        steps_speed.append(step_spd)

        # Residual ratio
        u_fit = a0 + a1 * xn + a2 * yn
        v_fit = b0 + b1 * xn + b2 * yn
        mean_u = float(np.mean(u_m))
        mean_v = float(np.mean(v_m))
        res_mag = np.sqrt((u_m - u_fit)**2 + (v_m - v_fit)**2)
        dev_mag = np.sqrt((u_m - mean_u)**2 + (v_m - mean_v)**2)
        dev_mean = float(np.mean(dev_mag))
        if dev_mean > 1e-7:
            steps_residuals.append(float(np.mean(res_mag)) / dev_mean)
        else:
            steps_residuals.append(0.0)

        # Band fits
        for b_name in ["top", "mid", "bot"]:
            bc = band_coords[b_name]
            bu_m = u[bc["by"], bc["bx"]]
            bv_m = v[bc["by"], bc["bx"]]
            ba0, ba1, ba2, bb0, bb1, bb2 = fit_affine_flow(bu_m, bv_m, bc["bxn"], bc["byn"])
            b_zoom_step = 0.5 * ((ba1 / (w / 2.0)) + (bb2 / (h / 2.0)))
            band_zoom_steps[b_name].append(b_zoom_step)

        # Heatmap output if requested
        if save_heatmaps_dir:
            os.makedirs(save_heatmaps_dir, exist_ok=True)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            hm_path = os.path.join(save_heatmaps_dir, f"shot_{shot_idx:02d}_pair_{pair_idx:03d}.png")
            cv2.imwrite(hm_path, bgr)

        pair_idx += 1
        t += 4

    num_steps = len(steps_zoom)
    if num_steps == 0:
        raise ValueError("Not enough frame pairs to analyze motion")

    # per shot: zoom_total = prod(1 + zoom_step) - 1, shift_x = sum(tx), shift_y = sum(ty)
    zoom_total = float(np.prod([1.0 + zs for zs in steps_zoom]) - 1.0)
    shift_x = float(sum(steps_tx))
    shift_y = float(sum(steps_ty))

    # Band zoom_total
    band_zoom_totals = {}
    for b_name in ["top", "mid", "bot"]:
        band_zoom_totals[b_name] = float(np.prod([1.0 + zs for zs in band_zoom_steps[b_name]]) - 1.0)
    depth_spread = band_zoom_totals["bot"] - band_zoom_totals["top"]

    # Speed profile
    total_speed = sum(steps_speed)
    cumsum_speed = np.cumsum(steps_speed)

    def cum_frac_at(q: float) -> float:
        if total_speed <= 1e-9:
            return q
        idx = q * num_steps
        i0 = int(math.floor(idx))
        frac = idx - i0
        val = (cumsum_speed[i0 - 1] if i0 > 0 else 0.0) + (steps_speed[i0] * frac if i0 < num_steps else 0.0)
        return float(val / total_speed)

    cum_25 = cum_frac_at(0.25)
    cum_50 = cum_frac_at(0.50)
    cum_75 = cum_frac_at(0.75)

    # Mean speed per quarter
    q_len = num_steps / 4.0
    mean_speed_q0 = float(cum_25 * total_speed / q_len)
    mean_speed_q1 = float((cum_50 - cum_25) * total_speed / q_len)
    mean_speed_q2 = float((cum_75 - cum_50) * total_speed / q_len)
    mean_speed_q3 = float((1.0 - cum_75) * total_speed / q_len)

    mean_residual_ratio = float(np.mean(steps_residuals)) if steps_residuals else 0.0

    return {
        "shot_index": shot_idx,
        "num_frames": num_frames,
        "num_steps": num_steps,
        "zoom_total": round(zoom_total, 6),
        "shift_x": round(shift_x, 6),
        "shift_y": round(shift_y, 6),
        "band_zoom_top": round(band_zoom_totals["top"], 6),
        "band_zoom_mid": round(band_zoom_totals["mid"], 6),
        "band_zoom_bot": round(band_zoom_totals["bot"], 6),
        "depth_spread": round(depth_spread, 6),
        "cum_speed": [round(cum_25, 4), round(cum_50, 4), round(cum_75, 4)],
        "mean_speed_quarters": [
            round(mean_speed_q0, 6),
            round(mean_speed_q1, 6),
            round(mean_speed_q2, 6),
            round(mean_speed_q3, 6)
        ],
        "residual_ratio": round(mean_residual_ratio, 4)
    }

def detect_cuts(frames_gray_resized: List[np.ndarray]) -> List[int]:
    """
    Detect hard cuts: mean absolute frame difference > max(20, 10 * median).
    Returns list of shot boundary frame indices (including 0 and total frames).
    """
    if len(frames_gray_resized) < 2:
        return [0, len(frames_gray_resized)]

    diffs = []
    for i in range(1, len(frames_gray_resized)):
        d = float(np.mean(np.abs(
            frames_gray_resized[i].astype(float) - frames_gray_resized[i - 1].astype(float)
        )))
        diffs.append(d)

    median_diff = float(np.median(diffs))
    threshold = max(20.0, 10.0 * median_diff)

    cuts = [0]
    for i, d in enumerate(diffs):
        if d > threshold:
            cuts.append(i + 1)
    cuts.append(len(frames_gray_resized))
    return sorted(list(set(cuts)))

def analyze_video(
    video_path: str,
    shot_frames: Optional[int] = None,
    cuts_auto: bool = False,
    save_heatmaps_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Read video, resize frames with cv2.INTER_AREA to width 332, split into shots,
    and analyze each shot.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames_resized = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_h, orig_w = frame.shape[:2]
        w = 332
        h = int(round(orig_h * (332.0 / orig_w)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_res = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        frames_resized.append(gray_res)
    cap.release()

    total_frames = len(frames_resized)
    if total_frames == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    # Determine shot boundaries
    if shot_frames is not None:
        boundaries = list(range(0, total_frames, shot_frames))
        if boundaries[-1] != total_frames:
            boundaries.append(total_frames)
    elif cuts_auto:
        boundaries = detect_cuts(frames_resized)
    else:
        # Default single shot
        boundaries = [0, total_frames]

    results = []
    for s_idx in range(len(boundaries) - 1):
        start_f = boundaries[s_idx]
        end_f = boundaries[s_idx + 1]
        shot_frames_list = frames_resized[start_f:end_f]
        if len(shot_frames_list) >= 9:
            shot_res = analyze_shot_frames(
                shot_frames_list,
                save_heatmaps_dir=save_heatmaps_dir,
                shot_idx=s_idx + 1
            )
            shot_res["start_frame"] = start_f
            shot_res["end_frame"] = end_f
            results.append(shot_res)

    return results

def main():
    parser = argparse.ArgumentParser(description="Motion Analyzer CLI")
    parser.add_argument("clip", help="Path to input video MP4")
    parser.add_argument("--shot-frames", type=int, default=None, help="Fixed shot frame count")
    parser.add_argument("--cuts", choices=["auto"], default=None, help="Auto cut detection")
    parser.add_argument("--json", dest="json_out", default=None, help="Path to output JSON")
    parser.add_argument("--heatmaps", default=None, help="Directory to save flow heatmaps")
    args = parser.parse_args()

    results = analyze_video(
        args.clip,
        shot_frames=args.shot_frames,
        cuts_auto=(args.cuts == "auto"),
        save_heatmaps_dir=args.heatmaps
    )

    # Print summary table
    print("\n" + "=" * 80)
    print(f"MOTION ANALYSIS RESULTS: {args.clip}")
    print("=" * 80)
    print(f"{'shot':<6} {'zoom_total':<12} {'shift_x':<10} {'shift_y':<10} {'depth_spr':<12} {'cum.speed 25/50/75%':<22}")
    print("-" * 80)
    for r in results:
        cum_str = f"{r['cum_speed'][0]:.2f}/{r['cum_speed'][1]:.2f}/{r['cum_speed'][2]:.2f}"
        print(f"{r['shot_index']:<6} {r['zoom_total']:+10.4f}  {r['shift_x']:+8.4f}  {r['shift_y']:+8.4f}  {r['depth_spread']:+10.4f}  {cum_str:<22}")
    print("=" * 80 + "\n")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved analysis JSON to: {args.json_out}")

if __name__ == "__main__":
    main()
