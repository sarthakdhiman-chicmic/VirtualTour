import math
import os
import sys
import argparse
import json
import shutil
import tempfile
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from modules.plan_io import Plan, parse_plan
from modules.remotion_render import render
from tools.motion_presets import preset_to_camera, PRESETS
from tools.motion_analyzer import analyze_video

TARGET_PROFILES = {
    "linear": [0.250, 0.500, 0.750],
    "ease_out": [0.438, 0.750, 0.938],
    "ease_in": [0.062, 0.250, 0.562],
    "ease_in_out": [0.156, 0.500, 0.844]
}

def create_one_shot_plan(
    image_path: str,
    preset: str,
    src_w: int,
    src_h: int,
    subject_x: float = 0.5,
    subject_y: float = 0.5,
    order: int = 0,
    out_w: int = 1920,
    out_h: int = 1080,
    fps: int = 30,
    duration_s: float = 5.0
) -> Tuple[Plan, Dict[str, Any], str]:
    """Build a one-shot cinematic plan (hard cuts, 5s, 16:9, 30fps)."""
    camera, fallback = preset_to_camera(
        preset=preset,
        src_w=src_w,
        src_h=src_h,
        out_w=out_w,
        out_h=out_h,
        subject_x=subject_x,
        subject_y=subject_y,
        order=order
    )

    image_basename = os.path.basename(image_path)
    plan_dict = {
        "plan_version": "1.2",
        "listing_id": "lab_preview",
        "mode": "cinematic",
        "preset": "cinematic_tour",
        "output": {
            "width": out_w,
            "height": out_h,
            "fps": fps,
            "codec": "h264"
        },
        "total_duration_s": duration_s,
        "shots": [
            {
                "shot_id": "shot_001",
                "order": 1,
                "asset_id": "lab_asset",
                "image_uri": f"file:///{image_basename}",
                "source": {
                    "width": src_w,
                    "height": src_h
                },
                "room_key": "room_01",
                "room_type": "interior",
                "narrative_stage": "intro",
                "role": "hero",
                "duration_s": duration_s,
                "camera": camera,
                "transition_in": {
                    "type": "cut",
                    "duration_s": 0.0
                }
            }
        ],
        "warnings": [],
        "provenance": {
            "scoring_config_version": "1.0",
            "planner_config_version": "1.0",
            "plan_hash": ""
        }
    }

    # Compute valid plan_hash
    hash_copy = dict(plan_dict)
    del hash_copy["provenance"]["plan_hash"]
    canonical_json = json.dumps(hash_copy, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
    computed_hash = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    plan_dict["provenance"]["plan_hash"] = computed_hash

    parsed_plan = parse_plan(plan_dict)
    return parsed_plan, camera, fallback

def check_black_edges(video_path: str, min_lum_threshold: float = 10.0) -> Tuple[bool, float]:
    """
    Check if the outermost 4 px of every frame stay above min_lum_threshold.
    Returns (passed, min_observed_luminance).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for black edge check: {video_path}")

    min_lum = float("inf")
    passed = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Sample border pixels (outer 4 rows and cols)
        top = gray[:4, :]
        bottom = gray[-4:, :]
        left = gray[:, :4]
        right = gray[:, -4:]

        frame_min = min(float(np.min(top)), float(np.min(bottom)), float(np.min(left)), float(np.min(right)))
        if frame_min < min_lum:
            min_lum = frame_min
        if frame_min < min_lum_threshold:
            passed = False

    cap.release()
    return passed, min_lum

def create_contact_sheet(video_path: str, out_jpg_path: str):
    """Save contact sheet JPG with first, middle, and last frames horizontally combined."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [0, total_frames // 2, max(0, total_frames - 1)]

    extracted = []
    current_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            extracted.append(frame)
        current_idx += 1

    cap.release()

    if len(extracted) == 3:
        sheet = np.hstack(extracted)
        cv2.imwrite(out_jpg_path, sheet)

def run_lab_single(
    image_path: str,
    preset: str,
    scale: float = 0.25,
    out_mp4: Optional[str] = None,
    out_dir: Optional[str] = None,
    subject_x: float = 0.5,
    subject_y: float = 0.5,
    order: int = 0
) -> Dict[str, Any]:
    """Run Motion Lab for a single image and preset."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    src_h, src_w = img.shape[:2]

    # Target calculation
    cfg = PRESETS[preset]
    if subject_x > 0.58:
        d = 1.0
    elif subject_x < 0.42:
        d = -1.0
    else:
        d = 1.0 if (order % 2 != 0) else -1.0

    # roll is fitted inside the window by an extra scale (remotion math.ts rollFitScale),
    # which adds to the zoom the analyzer measures
    roll = math.radians(cfg.get("roll_deg", 0.0))
    roll_fit = math.cos(roll) + (16 / 9) * math.sin(roll)
    target_zoom = (1.0 + cfg["zoom_total"]) * roll_fit - 1.0
    target_shift_x = cfg["shift_x"] * d
    target_shift_y = cfg["shift_y"]
    target_profile = TARGET_PROFILES.get(cfg["easing"], [0.25, 0.50, 0.75])

    plan, cam, fallback = create_one_shot_plan(
        image_path=image_path,
        preset=preset,
        src_w=src_w,
        src_h=src_h,
        subject_x=subject_x,
        subject_y=subject_y,
        order=order
    )

    workdir = tempfile.mkdtemp(prefix=f"motion_lab_{preset}_")
    try:
        images_dir = os.path.join(workdir, "images")
        os.makedirs(images_dir, exist_ok=True)
        basename = os.path.basename(image_path)
        shutil.copyfile(image_path, os.path.join(images_dir, basename))

        rendered_mp4 = render(plan, workdir, scale=scale)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            base_name = f"{os.path.splitext(basename)[0]}_{preset}"
            dest_mp4 = os.path.join(out_dir, f"{base_name}.mp4")
            shutil.copyfile(rendered_mp4, dest_mp4)
            final_mp4 = dest_mp4
            contact_path = os.path.join(out_dir, f"{base_name}_contact.jpg")
        elif out_mp4:
            shutil.copyfile(rendered_mp4, out_mp4)
            final_mp4 = out_mp4
            contact_path = os.path.splitext(out_mp4)[0] + "_contact.jpg"
        else:
            final_mp4 = rendered_mp4
            contact_path = os.path.join(workdir, "contact.jpg")

        create_contact_sheet(final_mp4, contact_path)
        black_edge_passed, min_lum = check_black_edges(final_mp4)

        # Optical flow analysis
        analysis = analyze_video(final_mp4)
        if not analysis:
            raise RuntimeError(f"Analysis returned no shots for {final_mp4}")
        shot_res = analysis[0]

        measured_zoom = shot_res["zoom_total"]
        measured_shift_x = shot_res["shift_x"]
        measured_shift_y = shot_res["shift_y"]
        measured_profile = shot_res["cum_speed"]

        # Differences
        err_zoom = abs(measured_zoom - target_zoom)
        err_shift_x = abs(measured_shift_x - target_shift_x)
        err_shift_y = abs(measured_shift_y - target_shift_y)
        profile_diffs = [abs(m - t) for m, t in zip(measured_profile, target_profile)]
        max_err_profile = max(profile_diffs) if profile_diffs else 0.0

        # Tolerances
        zoom_ok = err_zoom <= 0.03
        shift_x_ok = err_shift_x <= 0.015
        shift_y_ok = err_shift_y <= 0.015
        profile_ok = max_err_profile <= 0.07

        passed = zoom_ok and shift_x_ok and shift_y_ok and profile_ok and black_edge_passed

        return {
            "image": image_path,
            "preset": preset,
            "fallback": fallback,
            "target": {
                "zoom_total": round(target_zoom, 4),
                "shift_x": round(target_shift_x, 4),
                "shift_y": round(target_shift_y, 4),
                "profile": [round(x, 3) for x in target_profile]
            },
            "measured": {
                "zoom_total": round(measured_zoom, 4),
                "shift_x": round(measured_shift_x, 4),
                "shift_y": round(measured_shift_y, 4),
                "profile": [round(x, 3) for x in measured_profile]
            },
            "errors": {
                "zoom": round(err_zoom, 4),
                "shift_x": round(err_shift_x, 4),
                "shift_y": round(err_shift_y, 4),
                "profile_max": round(max_err_profile, 4)
            },
            "checks": {
                "zoom_ok": zoom_ok,
                "shift_x_ok": shift_x_ok,
                "shift_y_ok": shift_y_ok,
                "profile_ok": profile_ok,
                "black_edge_passed": black_edge_passed,
                "min_luminance": round(min_lum, 2)
            },
            "contact_sheet": contact_path if (out_dir or out_mp4) else None,
            "video_path": final_mp4 if (out_dir or out_mp4) else None,
            "passed": passed
        }
    finally:
        if not (out_dir or out_mp4):
            shutil.rmtree(workdir, ignore_errors=True)

def print_results_table(results: List[Dict[str, Any]]):
    print("\n" + "="*95)
    print(f"{'PRESET':<16} | {'ZOOM (Tgt / Meas / Err)':<22} | {'SHIFT_X (T/M/E)':<18} | {'SHIFT_Y (T/M/E)':<18} | {'PROFILE (MaxErr)':<12}")
    print("="*95)
    for r in results:
        p = r["preset"]
        fb = f" [{r['fallback']}]" if r["fallback"] != "none" else ""
        preset_label = (p + fb)[:16]

        z_t = r["target"]["zoom_total"]
        z_m = r["measured"]["zoom_total"]
        z_e = r["errors"]["zoom"]
        z_flag = "" if r["checks"]["zoom_ok"] else " (!)"
        z_str = f"{z_t:+.3f}/{z_m:+.3f}/{z_e:.3f}{z_flag}"

        sx_t = r["target"]["shift_x"]
        sx_m = r["measured"]["shift_x"]
        sx_e = r["errors"]["shift_x"]
        sx_flag = "" if r["checks"]["shift_x_ok"] else " (!)"
        sx_str = f"{sx_t:+.3f}/{sx_m:+.3f}/{sx_e:.3f}{sx_flag}"

        sy_t = r["target"]["shift_y"]
        sy_m = r["measured"]["shift_y"]
        sy_e = r["errors"]["shift_y"]
        sy_flag = "" if r["checks"]["shift_y_ok"] else " (!)"
        sy_str = f"{sy_t:+.3f}/{sy_m:+.3f}/{sy_e:.3f}{sy_flag}"

        pr_e = r["errors"]["profile_max"]
        pr_flag = "" if r["checks"]["profile_ok"] else " (!)"
        pr_str = f"{pr_e:.3f}{pr_flag}"

        print(f"{preset_label:<16} | {z_str:<22} | {sx_str:<18} | {sy_str:<18} | {pr_str:<12}")

    print("="*95)
    print("Tolerances: zoom <= 0.030, shift <= 0.015, profile <= 0.070. '(!)' indicates out of tolerance.")
    print("="*95 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Motion Lab: one-shot preview tool and analyzer comparison.")
    parser.add_argument("--image", type=str, help="Path to input photo")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), help="Preset name")
    parser.add_argument("--image-dir", type=str, help="Directory of photos")
    parser.add_argument("--all-presets", action="store_true", help="Run all 5 presets")
    parser.add_argument("--scale", type=float, default=0.25, help="Render scale (default 0.25)")
    parser.add_argument("--out", type=str, help="Output MP4 path for single run")
    parser.add_argument("--out-dir", type=str, default="lab_out", help="Output directory")

    args = parser.parse_args()

    presets_to_run = list(PRESETS.keys()) if args.all_presets else ([args.preset] if args.preset else ["push_in_gentle"])

    images = []
    if args.image:
        images.append(args.image)
    elif args.image_dir:
        for fname in sorted(os.listdir(args.image_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                images.append(os.path.join(args.image_dir, fname))

    if not images:
        print("Error: Specify --image or --image-dir")
        sys.exit(1)

    results = []
    for img_path in images:
        print(f"\nProcessing image: {img_path}")
        for p in presets_to_run:
            print(f"  Rendering & analyzing preset: {p}...")
            res = run_lab_single(
                image_path=img_path,
                preset=p,
                scale=args.scale,
                out_mp4=args.out if len(images) == 1 and len(presets_to_run) == 1 else None,
                out_dir=args.out_dir if (len(images) > 1 or len(presets_to_run) > 1) else None
            )
            results.append(res)

    print_results_table(results)

if __name__ == "__main__":
    main()
