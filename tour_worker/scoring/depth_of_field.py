"""Depth of field, interpreted as DEEP FOCUS (everything sharp), which is what
real-estate photography wants. Confirm with client: the SOW does not say deep vs shallow.

Sharpness is measured as the strength of the strongest edges per depth band, so flat
white walls (no edges) do not look 'blurry'. With a depth map the bands are depth
terciles (near/mid/far). Without one we fall back to three horizontal bands as a rough
proxy (bottom of frame ~ near, top ~ far)."""
import cv2
import numpy as np

from .config import DepthOfFieldConfig
from .schemas import CriterionResult


def _band_sharpness(abs_lap: np.ndarray, mask: np.ndarray, cfg: DepthOfFieldConfig) -> float | None:
    vals = abs_lap[mask]
    if vals.size == 0:
        return None
    if float((vals > cfg.edge_threshold).mean()) < cfg.min_edge_fraction:
        return None
    k = max(1, int(vals.size * cfg.top_fraction))
    top = float(np.partition(vals, -k)[-k:].mean())
    return float(1.0 - np.exp(-top / cfg.sharpness_k))


def _band_masks(shape: tuple[int, int], depth: np.ndarray | None) -> tuple[list[np.ndarray], str]:
    h, w = shape
    if depth is not None:
        d = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        q1, q2 = np.quantile(d, [1 / 3, 2 / 3])
        return [d <= q1, (d > q1) & (d <= q2), d > q2], "depth_terciles"
    rows = np.arange(h)[:, None] * np.ones((1, w))
    return [rows < h / 3, (rows >= h / 3) & (rows < 2 * h / 3), rows >= 2 * h / 3], "horizontal_thirds"


def score_depth_of_field(gray_u8: np.ndarray, depth: np.ndarray | None,
                         cfg: DepthOfFieldConfig) -> CriterionResult:
    abs_lap = np.abs(cv2.Laplacian(gray_u8.astype(np.float32), cv2.CV_32F, ksize=3))
    masks, mode = _band_masks(gray_u8.shape[:2], depth)
    bands = [_band_sharpness(abs_lap, m, cfg) for m in masks]
    valid = [b for b in bands if b is not None]
    details: dict[str, float | int | None] = {
        "band_near": None if bands[0] is None else round(bands[0], 4),
        "band_mid": None if bands[1] is None else round(bands[1], 4),
        "band_far": None if bands[2] is None else round(bands[2], 4),
        "bands_used": len(valid),
    }
    note = f"bands={mode}"
    if len(valid) >= cfg.min_bands:
        score = cfg.min_weight * min(valid) + cfg.mean_weight * (sum(valid) / len(valid))
    else:
        whole = _band_sharpness(abs_lap, np.ones(gray_u8.shape[:2], dtype=bool), cfg)
        if whole is None:
            return CriterionResult(score=None, details=details, note="too few edges to measure sharpness")
        score = whole
        note += "; fell back to whole-image sharpness"
    return CriterionResult(score=round(float(score), 4), details=details, note=note)
