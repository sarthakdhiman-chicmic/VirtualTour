"""Recompute "lit" from the cached ADE20K fireplace masks with the CURRENT modules/perceptual/fire.py.

  .venv/bin/python tools/fireplace_adapter/relabel_lit.py [--known]

No model is loaded (cheap, seconds per photo). fire.py is being revised (2026-10-21: firebox constraint
+ emissive brightness test), so the "lit" in teacher/<id>.json is interim; run this right before
evaluation. Each cached fireplace mask is split into its connected parts (production regions are
per-part too: segment.segments_to_regions) and handed to fire.fire_regions on the photo at MAX_SIDE.
Writes teacher/<id>_lit.json: {lit, n_fire, zone_area, fire_py_sha256} (overwritten every run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from modules.perceptual import fire as F  # noqa: E402

MAX_SIDE = 2048                       # modules.segment.MAX_SIDE (not imported: it would pick a model)
FIRE_PY = os.path.join(C.ROOT, "modules", "perceptual", "fire.py")


def mask_regions(mask: np.ndarray, kind: str = "fireplace") -> list[dict]:
    """0/255 mask -> one production-style region per connected part."""
    sh, sw = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return [{"object_id": f"seg_{i - 1:03d}", "class_name": kind, "confidence": 1.0,
             "bbox": [int(st[i, 0]), int(st[i, 1]), int(st[i, 0] + st[i, 2]), int(st[i, 1] + st[i, 3])],
             "mask": ((lab == i) * 255).astype(np.uint8), "space": [sw, sh]} for i in range(1, n)]


def lit_in(bgr_at_space: np.ndarray, fireplace: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """fire.py inside a fireplace mask -> (fire regions, union zone mask)."""
    fires = [r for r in F.fire_regions(bgr_at_space, mask_regions(fireplace)) if r["class_name"] == "fire"]
    zone = np.zeros(fireplace.shape, np.uint8)
    for r in fires:
        zone |= r["mask"]
    return fires, zone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", action="store_true")
    a = ap.parse_args()
    sha = hashlib.sha256(open(FIRE_PY, "rb").read()).hexdigest()
    tdir = os.path.join(C.KNOWN, "teacher") if a.known else C.TEACHER
    if a.known:
        photo = lambda pid: os.path.join(C.KNOWN_LISTINGS[pid.split("__")[0]], pid.split("__")[1] + ".jpg")  # noqa: E731
    else:
        man = {r["hash"]: r for r in C.manifest()}
        photo = lambda pid: C.photo_path(man[pid])   # noqa: E731
    n = n_lit = changed = 0
    for f in sorted(os.listdir(tdir)):
        if not f.endswith(".json") or f.endswith("_lit.json"):
            continue
        pid = f[:-5]
        j = json.load(open(os.path.join(tdir, f)))
        if j.get("skipped") or not j["has_fireplace"]:
            continue
        bgr = C.at_max_side(C.read_bgr(photo(pid)), MAX_SIDE)
        fp = cv2.imread(os.path.join(tdir, pid + "_fireplace.png"), 0)
        fires, zone = lit_in(bgr, fp)
        lit = bool(fires)
        json.dump({"lit": lit, "n_fire": len(fires), "zone_area": round((zone > 0).sum() / zone.size, 6),
                   "interim_lit": j["lit"], "fire_py_sha256": sha},
                  open(os.path.join(tdir, pid + "_lit.json"), "w"))
        n += 1
        n_lit += lit
        changed += lit != j["lit"]
    print(f"fireplace photos {n}: lit {n_lit} (changed vs interim label: {changed}); fire.py sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
