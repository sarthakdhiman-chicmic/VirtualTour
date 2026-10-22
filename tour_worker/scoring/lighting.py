"""Lighting = exposure health: blown highlights, crushed shadows, brightness band,
evenness across the frame, and dynamic range. Pure OpenCV/numpy, CPU only."""
import cv2
import numpy as np

from .config import LightingConfig
from .imaging import clamp01
from .schemas import CriterionResult


def score_lighting(gray01: np.ndarray, cfg: LightingConfig) -> CriterionResult:
    blown = float((gray01 >= cfg.blown_threshold).mean())
    crushed = float((gray01 <= cfg.crushed_threshold).mean())
    mean = float(gray01.mean())
    blocks = cv2.resize(gray01, (4, 4), interpolation=cv2.INTER_AREA)
    unevenness = float(blocks.std())
    p5, p95 = np.percentile(gray01, [5, 95])
    dyn_range = float(p95 - p5)

    pen_blown = clamp01(blown / cfg.blown_limit)
    pen_crushed = clamp01(crushed / cfg.crushed_limit)
    if mean < cfg.brightness_low:
        pen_bright = clamp01((cfg.brightness_low - mean) / cfg.brightness_low)
    elif mean > cfg.brightness_high:
        pen_bright = clamp01((mean - cfg.brightness_high) / (1.0 - cfg.brightness_high))
    else:
        pen_bright = 0.0
    pen_even = clamp01(unevenness / cfg.evenness_limit)
    pen_range = clamp01((cfg.dynamic_range_target - dyn_range) / cfg.dynamic_range_target)

    penalty = (cfg.w_blown * pen_blown + cfg.w_crushed * pen_crushed + cfg.w_brightness * pen_bright
               + cfg.w_evenness * pen_even + cfg.w_range * pen_range)
    # Additive penalties alone let a black or white frame score ~0.4 (unrelated terms act as
    # free passes), so the worst exposure failure also scales the whole score down.
    worst = max(pen_blown, pen_crushed, pen_bright)
    score = clamp01((1.0 - penalty) * (1.0 - cfg.severity_gate * worst))
    return CriterionResult(score=round(score, 4), details={
        "blown_fraction": round(blown, 4), "crushed_fraction": round(crushed, 4),
        "mean_brightness": round(mean, 4), "unevenness": round(unevenness, 4),
        "dynamic_range": round(dyn_range, 4),
    })
