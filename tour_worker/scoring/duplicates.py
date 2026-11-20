"""Near-duplicate photos (2026-11-20: a listing's dining room came twice, the same frame, and the
tour showed it twice): each photo is reduced to a 32x24 grey thumbnail, zero-mean and unit-norm;
two photos whose thumbnails correlate at DUPLICATE_MIN or more are the same picture (an export
twice over, a retouch, a slight crop). The better-scoring one stays; the other is left out of the
tour, with a warning duplicate_photo:<left out>:<kept>.

Measured on a 26-photo listing: the duplicate pair 0.9999; the closest two different photos 0.63."""
from __future__ import annotations

import cv2
import numpy as np

DUPLICATE_MIN = 0.92
THUMB = (32, 24)


def thumb(image_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = cv2.resize(g, THUMB, interpolation=cv2.INTER_AREA).ravel()
    t = t - t.mean()
    n = float(np.linalg.norm(t))
    return t / n if n > 1e-6 else t


def find_duplicates(thumbs: dict[str, np.ndarray], totals: dict[str, float]) -> dict[str, str]:
    """{left-out photo: the photo kept}. Best first (score, then photo_id): a photo is left out
    when it repeats one already kept."""
    order = sorted(thumbs, key=lambda p: (-totals.get(p, 0.0), p))
    kept: list[str] = []
    out: dict[str, str] = {}
    for p in order:
        twin = next((k for k in kept if float(thumbs[p] @ thumbs[k]) >= DUPLICATE_MIN), None)
        if twin is None:
            kept.append(p)
        else:
            out[p] = twin
    return out
