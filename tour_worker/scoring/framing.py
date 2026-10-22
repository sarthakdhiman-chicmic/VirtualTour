"""Framing = rule-of-thirds placement of the main subject + vertical-line tilt.

Saliency uses the spectral-residual method implemented directly in numpy, so we avoid
the opencv-contrib dependency and the result is fully deterministic.
NOTE: the SOW asks for rule-of-thirds compliance. Real-estate one-point-perspective
shots are often deliberately centred; this will under-score them. Flag to client.
"""
import math

import cv2
import numpy as np

from .config import FramingConfig
from .schemas import CriterionResult

_POWER_POINTS = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
_SAL_SIZE = 64


def _spectral_residual(gray01: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray01, (_SAL_SIZE, _SAL_SIZE), interpolation=cv2.INTER_AREA).astype(np.float64)
    spec = np.fft.fft2(small)
    log_amp = np.log(np.abs(spec) + 1e-8)
    phase = np.angle(spec)
    resid = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(resid + 1j * phase))) ** 2
    return cv2.GaussianBlur(sal, (0, 0), 2.0)


def _subject_centroid(sal: np.ndarray, percentile: float) -> tuple[float, float] | None:
    thr = np.percentile(sal, percentile)
    w = np.where(sal >= thr, sal, 0.0)
    total = w.sum()
    if total <= 0:
        return None
    ys, xs = np.mgrid[0:_SAL_SIZE, 0:_SAL_SIZE]
    cx = float(((xs + 0.5) / _SAL_SIZE * w).sum() / total)
    cy = float(((ys + 0.5) / _SAL_SIZE * w).sum() / total)
    return cx, cy


def _vertical_tilt_deg(gray_u8: np.ndarray, cfg: FramingConfig) -> tuple[float | None, int]:
    """Length-weighted median roll of near-vertical lines, in degrees.
    Signed median (not mean of |angle|) so perspective convergence, which leans left
    and right lines in opposite directions, cancels out and only true roll remains."""
    h = gray_u8.shape[0]
    edges = cv2.Canny(gray_u8, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=max(10, int(0.15 * h)), maxLineGap=8)
    if lines is None:
        return None, 0
    devs: list[float] = []
    lens: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):   # OpenCV 4: (N, 1, 4); OpenCV 5: (N, 4)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        if dy <= 0:
            continue
        ang = math.degrees(math.atan2(dx, dy))
        if abs(ang) <= cfg.max_vertical_search_deg:
            devs.append(ang)
            lens.append(math.hypot(dx, dy))
    if len(devs) < cfg.min_vertical_lines:
        return None, len(devs)
    order = np.argsort(devs)
    d = np.array(devs)[order]
    l = np.array(lens)[order]
    cum = np.cumsum(l)
    median = float(d[int(np.searchsorted(cum, cum[-1] / 2.0))])
    return median, len(devs)


def score_framing(gray01: np.ndarray, gray_u8: np.ndarray, cfg: FramingConfig) -> CriterionResult:
    details: dict[str, float | int | None] = {}
    parts: list[tuple[float, float]] = []   # (score, weight)

    has_contrast = float(gray01.std()) >= cfg.min_contrast_std
    centroid = _subject_centroid(_spectral_residual(gray01), cfg.saliency_percentile) if has_contrast else None
    if centroid is not None:
        cx, cy = centroid
        d = min(math.hypot(cx - px, cy - py) for px, py in _POWER_POINTS)
        thirds = math.exp(-((d / cfg.thirds_sigma) ** 2))
        details.update(subject_x=round(cx, 4), subject_y=round(cy, 4),
                       thirds_distance=round(d, 4), thirds_score=round(thirds, 4))
        parts.append((thirds, cfg.thirds_weight))

    tilt, n_lines = _vertical_tilt_deg(gray_u8, cfg)
    details["vertical_lines"] = n_lines
    if tilt is not None:
        tilt_score = math.exp(-((abs(tilt) / cfg.tilt_sigma_deg) ** 2))
        details.update(tilt_deg=round(tilt, 3), tilt_score=round(tilt_score, 4))
        parts.append((tilt_score, cfg.tilt_weight))

    if not parts:
        return CriterionResult(score=None, details=details, note="no usable saliency or vertical lines")
    wsum = sum(w for _, w in parts)
    score = sum(s * w for s, w in parts) / wsum
    note = None if len(parts) == 2 else "partial: only one framing sub-score available"
    return CriterionResult(score=round(score, 4), details=details, note=note)
