"""Shared paths and helpers for the fireplace adapter (offline research code, not used in production)."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "fireplace_adapter")
PHOTOS = os.path.join(OUT, "photos")
TEACHER = os.path.join(OUT, "teacher")
FEATURES = os.path.join(OUT, "features")
MODEL = os.path.join(OUT, "model")
EVAL = os.path.join(OUT, "eval")
LOGS = os.path.join(OUT, "logs")
KNOWN = os.path.join(OUT, "known")          # teacher / features for the known local listings
MIN_SIDE = 400                              # smaller photos are skipped (thumbnails)
PER_FOLDER_CAP = 4                          # first pass: at most 4 photos per folder ...
PER_FOLDER_MAX = 10_000                     # ... then the rest of the big folders (scan only; select caps at 12)
SEED = 0

# known local listings (backend scratch/backend_tour_e2e.py LISTINGS): the same files the
# perceptual e2e runs copied as img_XX.jpg (sorted original names)
KNOWN_LISTINGS = {
    "turkey": os.path.join(ROOT, "out", "perceptual_turkey", "images"),
    "downing": os.path.join(ROOT, "out", "perceptual_downing", "images"),
    "turkey_mix": os.path.join(ROOT, "out", "perceptual_turkey_mix", "images"),
}


def log(path: str, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")


def manifest() -> list[dict]:
    p = os.path.join(OUT, "manifest.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def photo_path(row: dict) -> str:
    return os.path.join(PHOTOS, row["hash"] + row["ext"])


def read_bgr(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        try:
            from PIL import Image
            img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:                          # noqa: BLE001
            return None
    return img


def at_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    k = min(1.0, max_side / max(w, h))
    if k < 1.0:
        bgr = cv2.resize(bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return bgr


def scan_order(rows: list[dict]) -> list[dict]:
    """Round-robin over listing folders (seeded shuffle): every folder's first PER_FOLDER_CAP photos come
    first, then the rest of the big folders (shared web-image / upload folders)."""
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["hash"]):
        by[r["folder"]].append(r)
    folders = sorted(by)
    rng.shuffle(folders)
    for f in folders:
        rng.shuffle(by[f])
    out = []
    for i in range(PER_FOLDER_MAX):
        out += [by[f][i] for f in folders if len(by[f]) > i]
    return out
