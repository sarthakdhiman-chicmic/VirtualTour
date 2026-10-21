"""Quality gate for perceptual motion: the fast render path against the exact one, per shot.

  python -m tools.quality.perceptual_quality <workdir with plan.json + images/> [--frames 6] [--baseline DIR]

For a few frames of every shot with motion regions it reports
  - field error: |fast - exact| displacement of the sampling maps, in SCREEN px (p99, max), over
    the pixels that move; and the brightness (gain) error
  - psnr: the fast frame vs the exact frame, over the moving pixels (dB)
  - motion: how far each region's content moves on screen (p95 px, exact path) - the visibility
    measure from review ("can a viewer see it?"), tracked so a change cannot quietly weaken it
  - ms per frame, exact vs fast (one process)
  - composition: the drawn frame (camera warp + redrawn moving window) vs a full-frame remap of
    the same exact maps: max |diff| and share of pixels that differ (seams, misplaced windows)
  - solid drag: share of moving pixels that sample a protected solid object (a lantern frame,
    a pole, a railing) - the object would bend
  - baseline (optional): exact frames vs PNGs saved earlier (<shot>_<frame>.png), max |diff|
    and share of pixels that differ
Writes <workdir>/quality.json and exits 1 when a limit below is broken.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from tour_worker.contract.plan import parse_plan
from tour_worker.render.frames import ShotWarp, homography_maps
from tour_worker.render.timeline import compute_timeline

LIMITS = {
    "field_err_p99_px": 0.05,     # screen px: far below what an eye can see in moving content
    "field_err_max_px": 0.25,
    "gain_err_max": 0.002,        # brightness: 0.2% (calm_water swings up to 4%)
    "psnr_min_db": 40.0,
    # warpPerspective and remap agree within 1 level on ~0.1% of pixels; anything more is a bug
    "composition_max_diff": 1,
    "composition_diff_share": 0.005,
    "baseline_max_diff": 1,
    "baseline_diff_share": 0.005,
    # visibility: every region must move at least this far on screen (p95, px). Review
    # 2026-10-16: curtains at 0.4-1.5 px were "barely noticeable"; trees at 2.5 px read well
    "motion_min_p95_px": 2.0,
    # solid objects in or beside trees / water (lantern frames, poles, railings): share of moving
    # pixels that pull their content from one - a bending lantern rod (review 2026-10-16)
    "solid_drag_share": 0.002,
    # share of a tree's solid objects (independent reference) that move at all
    "solid_moved_share": 0.02,
}
SOLID_MOVE_PX = 0.5          # source px


def reference_solids(src: np.ndarray, weight: np.ndarray, min_extent: float = 30.0) -> np.ndarray:
    """Non-leaf, non-sky parts >= min_extent px across near a tree region (the original rule)."""
    from tour_worker.motion.masks import big_parts, crop_box, leafiness
    out = np.zeros(weight.shape, bool)
    box = crop_box(weight, 40)
    if box is None:
        return out
    y0, y1, x0, x1 = box
    crop = src[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, sat, val = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    sky = (val > 0.55) & ((((h > 85) & (h < 130)) & (sat > 0.08)) | ((sat < 0.15) & (val > 0.75)))
    cand = cv2.morphologyEx(((leafiness(crop, soften=False) < 0.15) & ~sky).astype(np.uint8),
                            cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out[y0:y1, x0:x1] = big_parts(cand, min_extent)
    return out


def screen_scale(map_x: np.ndarray) -> float:
    """Screen px per source px (the camera's local zoom)."""
    return 1.0 / max(1e-6, float(np.median(np.abs(np.diff(map_x, axis=1)))))


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("inf")
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2)[mask].mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def check_shot(sw: ShotWarp, n_frames: int, baseline: str | None) -> dict:
    frames = sorted({round(i * (sw.n - 1) / max(1, n_frames - 1)) for i in range(n_frames)})
    rep = {"frames": frames, "field_err_p99_px": 0.0, "field_err_max_px": 0.0, "gain_err_max": 0.0,
           "psnr_min_db": float("inf"), "ms_exact": 0.0, "ms_fast": 0.0,
           "motion_p95_px": {r.spec.id: 0.0 for r in sw.regions}, "composition_max_diff": 0,
           "composition_diff_share": 0.0, "baseline_max_diff": None, "baseline_diff_share": None,
           "solid_drag_share": 0.0, "solid_moved_share": 0.0}
    if not sw.regions:                     # every region left still (too faint): nothing moves
        rep["motion_floor_px"] = LIMITS["motion_min_p95_px"] * min(sw.ow, sw.oh) / 1080
        return rep
    solid = None
    for r in sw.regions:
        if getattr(r, "solid", None) is not None:
            solid = r.solid if solid is None else (solid | r.solid)
    # an INDEPENDENT reference too (2026-10-19: a rule in leaf_weight emptied its own solid map and
    # lantern frames bent while this check passed): every non-leaf, non-sky part at least 30 px
    # across inside or beside a tree; none of it may MOVE (solid_moved_share), whatever the effect
    # decided. Pixels water / fire / curtains move are not tree solids.
    ref_solid = None
    other = np.zeros(sw.src.shape[:2], bool)
    for r in sw.regions:
        if r.effect == "canopy_flutter":
            ref = reference_solids(sw.src, r.feather)
            ref_solid = ref if ref_solid is None else (ref_solid | ref)
        else:
            other |= r.feather > 1e-3
    ref8 = (ref_solid & ~other).astype(np.uint8) if ref_solid is not None else None
    rep["solid_moved_share"] = 0.0
    if solid is not None:
        # a solid object is something no effect moves: pixels any region moves (water marked
        # "solid" by a neighbouring tree's search, say) do not count; a lantern inside a tree mask
        # has zero weight there and still counts
        moving = np.zeros(solid.shape, bool)
        for r in sw.regions:
            moving |= r.feather > 1e-3
        solid = solid & ~moving
    solid8 = solid.astype(np.uint8) if solid is not None else None
    for f in frames:
        t0 = time.perf_counter()
        exact = sw.frame(f, step=1)
        t1 = time.perf_counter()
        fast = sw.frame(f)                 # what the render draws
        t2 = time.perf_counter()
        rep["ms_exact"] += (t1 - t0) * 1000 / len(frames)
        rep["ms_fast"] += (t2 - t1) * 1000 / len(frames)
        ex_x, ex_y, ex_g = sw.maps(f, step=1)
        fa_x, fa_y, fa_g = sw.maps(f)
        full = cv2.remap(sw.src, ex_x, ex_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if ex_g is not None:
            full = np.clip(full.astype(np.float32) * ex_g[..., None], 0, 255).astype(np.uint8)
        d = np.abs(full.astype(int) - exact.astype(int)).max(axis=2)
        rep["composition_max_diff"] = max(rep["composition_max_diff"], int(d.max()))
        rep["composition_diff_share"] = max(rep["composition_diff_share"], float((d > 0).mean()))
        base_x, base_y = homography_maps(sw.homography(f), sw.grid)
        k = screen_scale(base_x)
        moved = (ex_x != base_x) | (ex_y != base_y)
        if solid8 is not None:
            far = np.hypot(ex_x - base_x, ex_y - base_y) > 0.25        # source px
            at = cv2.remap(solid8, ex_x, ex_y, cv2.INTER_NEAREST) > 0
            home = cv2.remap(solid8, base_x, base_y, cv2.INTER_NEAREST) > 0
            if far.any():
                rep["solid_drag_share"] = max(rep["solid_drag_share"], float((far & at & ~home).sum() / far.sum()))
        if ref8 is not None:
            on_solid = cv2.remap(ref8, base_x, base_y, cv2.INTER_NEAREST) > 0
            if on_solid.any():
                shift = np.hypot(ex_x - base_x, ex_y - base_y)[on_solid]
                rep["solid_moved_share"] = max(rep["solid_moved_share"],
                                               float((shift > SOLID_MOVE_PX).mean()))
        if moved.any():
            err = np.hypot(fa_x - ex_x, fa_y - ex_y)[moved] * k
            rep["field_err_p99_px"] = max(rep["field_err_p99_px"], float(np.percentile(err, 99)))
            rep["field_err_max_px"] = max(rep["field_err_max_px"], float(err.max()))
            rep["psnr_min_db"] = min(rep["psnr_min_db"], psnr(fast, exact, moved))
        if ex_g is not None or fa_g is not None:
            g0 = ex_g if ex_g is not None else np.ones_like(fa_g)
            g1 = fa_g if fa_g is not None else np.ones_like(ex_g)
            rep["gain_err_max"] = max(rep["gain_err_max"], float(np.abs(g1 - g0).max()))
        if baseline:
            p = os.path.join(baseline, f"{sw.shot.shot_id}_{f:04d}.png")
            if os.path.exists(p):
                d = np.abs(cv2.imread(p).astype(int) - exact.astype(int)).max(axis=2)
                rep["baseline_max_diff"] = max(rep["baseline_max_diff"] or 0, int(d.max()))
                rep["baseline_diff_share"] = max(rep["baseline_diff_share"] or 0.0, float((d > 0).mean()))
    # visibility: the same measure the render used to keep or still each region
    # (tour_worker/render/frames.region_motion_px: exact field, probe frames across the whole shot)
    # (fire is judged by its light, not its displacement: not listed)
    rep["motion_p95_px"] = {r.spec.id: sw.motion_px[r.spec.id] for r in sw.regions
                            if r.effect != "fire_flicker"}
    rep["motion_floor_px"] = LIMITS["motion_min_p95_px"] * min(sw.ow, sw.oh) / 1080
    return rep


def failures(rep: dict) -> list[str]:
    out = []
    for key in ("field_err_p99_px", "field_err_max_px", "gain_err_max", "solid_drag_share",
                "solid_moved_share"):
        if rep[key] > LIMITS[key]:
            out.append(f"{key} {rep[key]:.4f} > {LIMITS[key]}")
    if rep["psnr_min_db"] < LIMITS["psnr_min_db"]:
        out.append(f"psnr {rep['psnr_min_db']:.1f} dB < {LIMITS['psnr_min_db']}")
    for key in ("composition_max_diff", "composition_diff_share", "baseline_max_diff", "baseline_diff_share"):
        if rep[key] is not None and rep[key] > LIMITS[key]:
            out.append(f"{key} {rep[key]} > {LIMITS[key]}")
    floor = rep.get("motion_floor_px", LIMITS["motion_min_p95_px"])
    for rid, px in rep["motion_p95_px"].items():
        if px < floor:
            out.append(f"weak_motion {rid} {px:.1f} px < {floor:.2f}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--baseline")
    a = ap.parse_args(argv)
    cv2.setNumThreads(1)
    plan = parse_plan(json.load(open(os.path.join(a.workdir, "plan.json"))))
    images = os.path.join(a.workdir, "images")
    report, bad = {"limits": LIMITS, "shots": {}}, []
    for shot, tl in zip(plan.shots, compute_timeline(plan)):
        # shots with motion regions and a keyframe_path / compound camera (the gate's scope since
        # the Remotion days, when only those shots were drawn by OpenCV)
        if not shot.motion_regions or shot.camera.type not in ("compound", "keyframe_path"):
            continue
        sw = ShotWarp(plan, shot, tl["render_frames"], images)
        if not sw.regions:
            continue
        rep = check_shot(sw, a.frames, a.baseline)
        report["shots"][shot.shot_id] = rep
        fails = failures(rep)
        bad += [f"{shot.shot_id}: {x}" for x in fails]
        motion = "  ".join(f"{r.spec.label}/{r.spec.effect} "
                           + (f"{rep['motion_p95_px'][r.spec.id]:.1f}px" if r.spec.id in rep["motion_p95_px"]
                              else f"{sw.motion_px[r.spec.id]:.1f}px (light-led)")
                           for r in sw.regions)
        dropped = [f"{k} {v:.1f}px" for k, v in sw.motion_px.items() if k not in rep["motion_p95_px"]]
        if dropped:
            motion += "  (left still, too faint to read: " + ", ".join(dropped) + ")"
        print(f"{shot.shot_id}  err p99 {rep['field_err_p99_px']:.3f}px max {rep['field_err_max_px']:.3f}px  "
              f"gain {rep['gain_err_max']:.4f}  psnr {rep['psnr_min_db']:.1f}dB  "
              f"{rep['ms_exact']:.0f}->{rep['ms_fast']:.0f} ms/frame  "
              f"composition {rep['composition_max_diff']}/{rep['composition_diff_share']:.2%}  "
              f"baseline {rep['baseline_max_diff']}/{(rep['baseline_diff_share'] or 0):.2%}  "
              f"{'OK' if not fails else 'FAIL'}\n      motion: {motion}")
    report["failures"] = bad
    json.dump(report, open(os.path.join(a.workdir, "quality.json"), "w"), indent=1)
    print("PASS" if not bad else "FAIL:\n  " + "\n  ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
