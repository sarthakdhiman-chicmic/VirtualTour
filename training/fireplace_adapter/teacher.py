"""Teacher labels: ADE20K OneFormer fireplace masks + lit flames (tour_worker/vision/fire_detect.py).

  python tools/safe_run.py --cpus 8,10,12,14 --memory 7G -- \
      .venv/bin/python training/fireplace_adapter/teacher.py [--known] [--target-fire 150 --min-neg 250]

Per photo (at segment.MAX_SIDE), in out/fireplace_adapter/teacher/ (or known/teacher/ with --known):
  <id>.json            has_fireplace, lit, areas (fractions of the photo), scores, space, ADE scene classes
  <id>_fireplace.png   union of the fireplace regions production would keep (score >= 0.5, parts >= 0.3%)
  <id>_flame.png       flame pixels (fire.flame_mask) inside those fireplaces
  <id>_zone.png        lit-fire zones (fire.fire_regions: fireplaces whose flames pass MIN_FLAME_AREA)
Resumable (existing JSONs are skipped); progress in logs/teacher.log; scan order = common.scan_order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["SEGMENT_MODEL"] = "ade20k"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tour_worker.vision import segment as S  # noqa: E402
from tour_worker.vision import fire_detect as F  # noqa: E402

assert S.SEGMENT_MODEL == "ade20k"


def label_one(seg: S.Segmenter, bgr: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    bgr = C.at_max_side(bgr, S.MAX_SIDE)
    sh, sw = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = seg.processor(images=rgb, task_inputs=["panoptic"], return_tensors="pt").to(seg.device)
    with seg.torch.no_grad():
        out = seg.model(**inputs)
    res = seg.processor.post_process_panoptic_segmentation(out, target_sizes=[(sh, sw)])[0]
    pan = res["segmentation"].cpu().numpy()
    regions = S.segments_to_regions(pan, res["segments_info"], seg.id2label, (sw, sh))
    scene = S.scene_summary(pan, res["segments_info"], seg.id2label, (sw, sh))
    fps = [r for r in regions if r["class_name"] == "fireplace"]
    fp = np.zeros((sh, sw), np.uint8)
    flame = np.zeros((sh, sw), np.uint8)
    for r in fps:
        fp |= r["mask"]
        flame |= F.flame_mask(bgr, r["mask"] > 0)
    zone = np.zeros((sh, sw), np.uint8)
    fires = [r for r in F.fire_regions(bgr, fps) if r["class_name"] == "fire"]
    for r in fires:
        zone |= r["mask"]
    # raw ADE fireplace pixels before the score / part filters (for reference only)
    raw = np.zeros((sh, sw), bool)
    for s in res["segments_info"]:
        if seg.id2label.get(s["label_id"], seg.id2label.get(str(s["label_id"]))) == "fireplace":
            raw |= pan == s["id"]
    tot = float(sh * sw)
    info = {"space": [sw, sh], "has_fireplace": bool(fps), "lit": bool(fires),
            "n_fireplace": len(fps), "n_fire": len(fires),
            "fireplace_area": round((fp > 0).sum() / tot, 5), "flame_area": round((flame > 0).sum() / tot, 6),
            "zone_area": round((zone > 0).sum() / tot, 6), "raw_fireplace_area": round(raw.sum() / tot, 5),
            "fireplace_scores": [r["confidence"] for r in fps],
            "scene": {k: v["area"] for k, v in scene["classes"].items()}}
    return info, {"fireplace": fp, "flame": flame, "zone": zone}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", action="store_true", help="label the known local listings instead")
    ap.add_argument("--target-fire", type=int, default=150)
    ap.add_argument("--min-neg", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out_dir = os.path.join(C.KNOWN, "teacher") if a.known else C.TEACHER
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(C.LOGS, exist_ok=True)
    lg = os.path.join(C.LOGS, "teacher_known.log" if a.known else "teacher.log")
    if a.known:
        items = [(f"{name}__{f[:-4]}", os.path.join(d, f)) for name, d in C.KNOWN_LISTINGS.items()
                 if name != "turkey_mix" for f in sorted(os.listdir(d)) if f.endswith(".jpg")]
    else:
        items = [(r["hash"], C.photo_path(r)) for r in C.scan_order(C.manifest())]
    n_fire = n_neg = n_lit = 0
    for pid, _ in items:                                 # count what is already done (resume)
        p = os.path.join(out_dir, pid + ".json")
        if os.path.exists(p):
            j = json.load(open(p))
            if j.get("skipped"):
                continue
            n_fire += j["has_fireplace"]
            n_neg += not j["has_fireplace"]
            n_lit += j["lit"]
    C.log(lg, f"start: {len(items)} items, already fire {n_fire} lit {n_lit} neg {n_neg}")
    seg = S.Segmenter(threads=a.threads)
    for i, (pid, path) in enumerate(items):
        if not a.known and n_fire >= a.target_fire and n_neg >= a.min_neg:
            C.log(lg, f"targets reached after {i} items")
            break
        jp = os.path.join(out_dir, pid + ".json")
        if os.path.exists(jp):
            continue
        bgr = C.read_bgr(path)
        if bgr is None or min(bgr.shape[:2]) < C.MIN_SIDE:
            json.dump({"skipped": "unreadable" if bgr is None else "small"}, open(jp, "w"))
            continue
        t = time.time()
        info, masks = label_one(seg, bgr)
        for k, m in masks.items():
            if k == "fireplace" or m.any():
                cv2.imwrite(os.path.join(out_dir, f"{pid}_{k}.png"), m)
        info["seconds"] = round(time.time() - t, 2)
        json.dump(info, open(jp + ".tmp", "w"))
        os.replace(jp + ".tmp", jp)
        n_fire += info["has_fireplace"]
        n_neg += not info["has_fireplace"]
        n_lit += info["lit"]
        C.log(lg, f"{i + 1}/{len(items)} {pid[:12]} fp={int(info['has_fireplace'])} lit={int(info['lit'])} "
                  f"{info['seconds']}s | fire {n_fire} lit {n_lit} neg {n_neg}")
    C.log(lg, f"done: fire {n_fire} lit {n_lit} neg {n_neg}")


if __name__ == "__main__":
    main()
