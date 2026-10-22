"""Coverage = how much of the room the frame reveals. OpenCV + EXIF only.

HONEST LIMITS: the SOW defines coverage as "% of room visible". That needs room geometry
and camera pose, which a single photo does not contain. This is a HEURISTIC PROXY built from
three cues, none of which measures room area directly:

  1. Field of view (strong cue): wide-angle shots show more of a room. Taken from the EXIF
     35mm-equivalent focal length. Missing EXIF => this cue is skipped, not guessed.
  2. Ceiling/floor visibility (weak): long near-horizontal edges in the top and bottom zones
     (wall/ceiling and wall/floor boundaries).
  3. Wall planes (weak): number of distinct clusters of long vertical edges (corners, door and
     window frames count too, so this over-counts).

It must be validated against human judgement on real photos before its weight is trusted.
A pose-based version (camera frustum vs floor polygon) can replace score_coverage() later;
the rest of the pipeline only sees a 0-1 score."""
import math

import cv2
import numpy as np

from .config import CoverageConfig
from .imaging import clamp01
from .schemas import CriterionResult

_DIAG_35MM_MM = 43.27   # diagonal of a 36 x 24 mm frame


def fov_diag_deg(focal_35mm: float) -> float:
    return math.degrees(2.0 * math.atan(_DIAG_35MM_MM / (2.0 * focal_35mm)))


def _segments(gray_u8: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = gray_u8.shape
    edges = cv2.Canny(gray_u8, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=max(10, int(0.2 * min(h, w))), maxLineGap=10)
    if lines is None:
        return []
    # OpenCV 4 returns (N, 1, 4), OpenCV 5 (N, 4): the same segments either way
    return [tuple(int(v) for v in l) for l in lines.reshape(-1, 4)]


def _ceiling_floor(segs, w: int, h: int, cfg: CoverageConfig) -> tuple[float, float]:
    top = bottom = 0.0
    for x1, y1, x2, y2 in segs:
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx == 0 or math.degrees(math.atan2(dy, dx)) > cfg.horizontal_max_angle_deg:
            continue
        length = math.hypot(dx, dy)
        if length < cfg.horizontal_min_len_frac * w:
            continue
        mid_y = (y1 + y2) / 2.0 / h
        if mid_y <= cfg.top_zone:
            top += length
        elif mid_y >= 1.0 - cfg.bottom_zone:
            bottom += length
    full = cfg.edge_span_target * w
    return clamp01(top / full), clamp01(bottom / full)


def _wall_planes(segs, w: int, h: int, cfg: CoverageConfig) -> int:
    xs = []
    for x1, y1, x2, y2 in segs:
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy == 0 or math.degrees(math.atan2(dx, dy)) > cfg.vertical_max_angle_deg:
            continue
        if math.hypot(dx, dy) >= cfg.vertical_min_len_frac * h:
            xs.append((x1 + x2) / 2.0)
    if not xs:
        return 0
    xs.sort()
    clusters, last = 1, xs[0]
    for x in xs[1:]:
        if x - last > cfg.plane_merge_gap_frac * w:
            clusters += 1
        last = x
    return clusters


def score_coverage(gray_u8: np.ndarray, focal_35mm: float | None, cfg: CoverageConfig) -> CriterionResult:
    h, w = gray_u8.shape[:2]
    details: dict[str, float | int | None] = {"focal_35mm": focal_35mm}
    if float(gray_u8.std()) < cfg.min_content_std:
        # a lens setting says nothing about what a blank frame shows
        return CriterionResult(score=None, details=details, note="flat frame: no content to measure")
    parts: list[tuple[float, float]] = []

    if focal_35mm:
        fov = fov_diag_deg(focal_35mm)
        fov_score = clamp01((fov - cfg.fov_low_deg) / (cfg.fov_high_deg - cfg.fov_low_deg))
        details.update(fov_diag_deg=round(fov, 1), fov_score=round(fov_score, 4))
        parts.append((fov_score, cfg.w_fov))

    segs = _segments(gray_u8)
    details["long_edges"] = len(segs)
    if segs:
        top, bottom = _ceiling_floor(segs, w, h, cfg)
        cf = (top + bottom) / 2.0
        n = _wall_planes(segs, w, h, cfg)
        planes = clamp01((n - 1) / max(1, cfg.plane_clusters_for_full - 1))
        details.update(ceiling_zone=round(top, 4), floor_zone=round(bottom, 4),
                       ceiling_floor_score=round(cf, 4), vertical_clusters=n,
                       wall_planes_score=round(planes, 4))
        parts.append((cf, cfg.w_ceiling_floor))
        parts.append((planes, cfg.w_wall_planes))

    if not parts:
        return CriterionResult(score=None, details=details,
                               note="no EXIF focal length and no long straight edges found")
    wsum = sum(wt for _, wt in parts)
    score = sum(s * wt for s, wt in parts) / wsum
    note = "heuristic proxy" + ("" if focal_35mm else "; no EXIF focal length, structure cues only")
    return CriterionResult(score=round(score, 4), details=details, note=note)
