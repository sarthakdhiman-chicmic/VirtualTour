"""Motion signals per photo: what the shot planner needs to aim and size a camera move.

Not a quality criterion (it does not change the photo's score or ranking). Spectral-residual
saliency (Hou & Zhang 2007), the same maths as framing.py but with the tour worker's parameters,
so the planner sees exactly the numbers the approved v8 tours were planned with
(worker tools/tour_v2/saliency.py):

  peak     centroid (x, y), normalised, of the top 5% of saliency pixels, ignoring a 5% border
  side     "left" / "right": the half with more saliency mass; mass_ratio = larger / smaller
  spread   fraction of pixels above the mean saliency (high = busy wide view, low = one subject)

coverage_class ("wide" / "medium" / "tight") comes from a small matrix: the room's expected size
(from its room type) x the photo's spread band, bumped one class by the EXIF field of view when
the photo has one (coverage.py's fov_score). The coverage criterion alone is not used: without
EXIF it scored 0.0 on several wide living rooms and 1.0 on a bedroom (29 photos measured).
"""

import cv2
import numpy as np

from .config import MotionConfig
from .rooms import canon
from .schemas import MotionSignals

CLASSES = ("tight", "medium", "wide")


def spectral_residual(gray01: np.ndarray, cfg: MotionConfig) -> np.ndarray:
    small = cv2.resize(gray01, (cfg.sr_size, cfg.sr_size), interpolation=cv2.INTER_AREA)
    spec = np.fft.fft2(small)
    log_amp = np.log(np.abs(spec) + 1e-12)
    residual = log_amp - cv2.blur(log_amp, (cfg.sr_box, cfg.sr_box))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spec)))) ** 2
    sal = cv2.GaussianBlur(
        sal.astype(np.float32), (cfg.sr_blur_ksize, cfg.sr_blur_ksize), cfg.sr_blur_sigma
    )
    h = round(gray01.shape[0] * cfg.analysis_w / gray01.shape[1])
    sal = cv2.resize(sal, (cfg.analysis_w, h), interpolation=cv2.INTER_LINEAR)
    sal = cv2.GaussianBlur(sal, (0, 0), cfg.smooth_sigma_frac * cfg.analysis_w)
    sal -= sal.min()
    return sal / max(float(sal.max()), 1e-12)


def room_size(room_type: str | None, cfg: MotionConfig) -> str:
    c = canon(room_type)
    if c in cfg.large_rooms:
        return "large"
    if c in cfg.small_rooms:
        return "small"
    return "medium"  # medium list, unknown and untagged rooms


def coverage_class(
    spread: float, size: str, fov_score: float | None, cfg: MotionConfig
) -> tuple[str, str]:
    band = "low" if spread < cfg.spread_low else "high" if spread >= cfg.spread_high else "mid"
    k = CLASSES.index(cfg.matrix[size][band])
    if fov_score is not None:
        if fov_score >= cfg.fov_wide_bump:
            k = min(k + 1, 2)
        elif fov_score <= cfg.fov_narrow_bump:
            k = max(k - 1, 0)
    return CLASSES[k], band


def motion_signals(
    gray01: np.ndarray, room_type: str | None, fov_score: float | None, cfg: MotionConfig
) -> MotionSignals | None:
    """gray01: the full-resolution photo, grayscale in [0, 1]. None for a flat (blank) frame."""
    if float(gray01.std()) * 255.0 < cfg.min_content_std:
        return None
    sal = spectral_residual(gray01, cfg)
    h, w = sal.shape
    by, bx = round(h * cfg.border_fraction), round(w * cfg.border_fraction)
    inner = sal[by : h - by, bx : w - bx]
    thr = np.quantile(inner, 1 - cfg.top_fraction)
    ys, xs = np.nonzero(inner >= thr)
    peak_x, peak_y = (xs.mean() + bx + 0.5) / w, (ys.mean() + by + 0.5) / h
    left, right = float(sal[:, : w // 2].sum()), float(sal[:, w - w // 2 :].sum())
    spread = float((sal > sal.mean()).mean())
    size = room_size(room_type, cfg)
    cls, band = coverage_class(spread, size, fov_score, cfg)
    gh = max(1, round(cfg.grid_w * h / w))
    grid = cv2.resize(sal, (cfg.grid_w, gh), interpolation=cv2.INTER_AREA)
    grid = grid / max(float(grid.max()), 1e-12)
    return MotionSignals(
        peak_x=round(float(peak_x), 4),
        peak_y=round(float(peak_y), 4),
        side="left" if left > right else "right",
        mass_ratio=round(max(left, right) / max(min(left, right), 1e-12), 4),
        spread=round(spread, 4),
        spread_band=band,
        room_size=size,
        coverage_class=cls,
        grid=[[round(float(v), 3) for v in row] for row in grid],
    )
