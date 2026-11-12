"""Time a 3D tour (plan 1.6) on this host, through the production tour_render job.

The plan is a two-shot tour of the sample apartment scene (green_model_house_3dgs.ply): a push-in
over the dining table, then a 180-degree pan from the kitchen past the corridor to the living
room while the camera drifts 0.9 m (parallax). Same job, storage and checks as a backend request.

  python -m tools.runpod.bench --scene s3://bucket/key.ply --sha256 <hex> --output-prefix s3://bucket/prefix/
      [--width 1920 --height 1080]     output size (3840x2160 for a 4K run)
      [--fps 30]
      [--write-plan plan.json]         also save the request (e.g. to post it to the service)
      [--plan-only]                    build and check the request, do not render

Settings as for the service: SPLAT_BACKEND, SPLAT_SUPERSAMPLE, SPLAT_SH_DEGREE, SCENE_CACHE_DIR (unset:
the scene is downloaded into the job's folder and deleted after loading).
Prints the job result (incl. `timings`: scene_fetch, scene_load, rasterise, encode, upload) as JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from tour_worker.contract.plan import compute_plan_hash, parse_plan

# the apartment's axes (z up, floor -0.96, ceiling 1.51); eye height 0.55 = 1.5 m above the floor
PUSH_EYE = ([-0.67, 0.5, 0.55], [-0.67, 0.05, 0.50])
PUSH_TARGET = [-0.67, -3.0, 0.15]
PAN_EYE = ([-0.85, -1.95, 0.55], [-1.35, -2.70, 0.55])
PAN_HEADING = (70.0, 250.0)          # degrees from +x: kitchen -> corridor -> living room
PAN_RADIUS, PAN_TARGET_Z = 2.5, 0.30


def _line(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [list(map(float, np.round(a + (b - a) * j / 3, 4))) for j in range(4)]


def pan_segments(n: int = 3) -> list[dict]:
    """The pan as n cubic segments: the target on a circular arc around the moving eye."""
    e0, e1 = (np.asarray(p, float) for p in PAN_EYE)
    a0, a1 = PAN_HEADING
    step = (a1 - a0) / n
    k = 4.0 / 3.0 * math.tan(math.radians(step) / 4.0) * PAN_RADIUS      # arc handle length
    segs = []
    for i in range(n):
        a, b = math.radians(a0 + i * step), math.radians(a0 + (i + 1) * step)
        s0, s1 = e0 + (e1 - e0) * i / n, e0 + (e1 - e0) * (i + 1) / n
        c0 = s0[:2] + PAN_RADIUS * np.array([math.cos(a), math.sin(a)])
        c1 = s1[:2] + PAN_RADIUS * np.array([math.cos(b), math.sin(b)])
        h0 = c0 + k * np.array([-math.sin(a), math.cos(a)])
        h1 = c1 - k * np.array([-math.sin(b), math.cos(b)])
        target = [list(map(float, np.round(np.r_[p, PAN_TARGET_Z], 4))) for p in (c0, h0, h1, c1)]
        segs.append({"eye": _line(s0, s1), "target": target})
    return segs


def _shot(i, sid, room_key, room_type, stage, role, dur, move, segments, hfov, transition):
    return {"shot_id": sid, "order": i, "asset_id": "scene", "room_key": room_key, "room_type": room_type,
            "narrative_stage": stage, "role": role, "duration_s": dur,
            "camera": {"type": "bezier_path", "easing": "ease_in_out", "preset": None, "start": None, "end": None,
                       "move": move, "keyframes": None, "segments": segments, "timing": None,
                       "projection": None, "hfov_deg": hfov, "time_warp": None, "punches": None},
            "transition_in": transition, "motion_regions": None}


def build_request(scene_uri: str, sha256: str, output_prefix: str, width=1920, height=1080, fps=30) -> dict:
    shots = [
        _shot(1, "s1", "dining_1", "dining_room", "arrival", "establishing", 5.0, "push_in",
              [{"eye": _line(*PUSH_EYE), "target": _line(PUSH_TARGET, PUSH_TARGET)}], [72.0, 64.0],
              {"type": "cut", "duration_s": 0.0}),
        _shot(2, "s2", "living_1", "living_room", "heart", "hero", 7.0, "pan",
              pan_segments(), [68.0, 68.0], {"type": "fade", "duration_s": 0.5}),
    ]
    plan = {"plan_version": "1.6", "listing_id": "splat_demo_green_house", "mode": "cinematic_3d",
            "fallback_reason": None, "preset": "property_showcase",
            "output": {"width": width, "height": height, "fps": fps, "codec": "h264"},
            "total_duration_s": sum(s["duration_s"] for s in shots), "shots": shots, "warnings": [],
            "provenance": {"scoring_config_version": "1", "planner_config_version": "1"},
            "music": None, "scene": {"uri": scene_uri, "sha256": sha256, "format": scene_uri.rsplit(".", 1)[-1]}}
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    parse_plan(plan)                                    # the worker's own checks
    return {"job_type": "tour_render", "job_id": f"bench-{int(time.time())}", "plan": plan,
            "output_prefix": output_prefix}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--write-plan")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    req = build_request(a.scene, a.sha256, a.output_prefix, a.width, a.height, a.fps)
    if a.write_plan:
        with open(a.write_plan, "w") as f:
            json.dump({"input": req}, f, indent=1)
    if a.plan_only:
        print(json.dumps({"plan_hash": req["plan"]["provenance"]["plan_hash"], "shots": len(req["plan"]["shots"])}))
        return 0
    from tour_worker.jobs.router import run_job
    t = time.perf_counter()
    out = run_job(req)
    out.setdefault("bench_wall_s", round(time.perf_counter() - t, 1))
    print(json.dumps(out, indent=1))
    return 0 if out.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
