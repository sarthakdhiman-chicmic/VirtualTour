"""Pre-warp shots that use projection "rotate" (addendum 1.3.1) into per-frame images.

Chrome's CSS matrix3d path made renders non-deterministic: two renders of the same plan differed
in ~300-400 of 2250 frames, including crop shots sharing a render chunk with a rotate shot
(out/tour_v5/report.md); will-change and per-frame remounting did not fix it. So the rotation is
warped here with OpenCV (exact homography from modules/camera_path.rotate_homography, linear
interpolation, which matches Chrome's sampling of crop shots: SSIM 0.96-0.97, same sharpness)
and the renderer only shows the finished frames.

Frame f of a shot with n rendered frames is the camera at t = f / (n - 1), as in KenBurnsShot.

Plan 1.4 perceptual motion: a shot with motion regions is warped with one cv2.remap whose maps
are the camera homography plus the regions' displacement (modules/perceptual/effects.py), so the
content moves inside its masks with a single resample. Shots without regions keep the exact
warpPerspective path (byte-identical to before).
"""
import os

import cv2
import numpy as np

import math

from modules.camera_path import (KeyframePath, base_window, roll_fit_scale, rotate_homography, DEFAULT_HFOV_DEG,
                                 time_warp_at)
from modules.perceptual.effects import ROI_PROBE, apply_regions, field_align, load_regions, probe_box

def crop_roll_homography(cx, cy, zoom, roll_deg, src_w, src_h, out_w, out_h):
    """OUTPUT px -> SOURCE px for a crop window rolled by roll_deg about the frame centre and scaled
    by roll_fit_scale, exactly as KenBurnsShot renders it (CSS rotate(roll) scale(fit) about the
    centre over the window)."""
    bw, bh = base_window(src_w, src_h, out_w, out_h)
    w = bw / zoom * src_w
    h = bh / zoom * src_h
    k = w / out_w                                 # source px per unrotated output px
    th = math.radians(roll_deg)
    f = roll_fit_scale(th, out_w, out_h)
    c, s_ = math.cos(-th) / f, math.sin(-th) / f  # q = centre + R(-th)(p - centre) / f
    ox, oy = out_w / 2, out_h / 2
    a = [[c, -s_, ox - c * ox + s_ * oy], [s_, c, oy - s_ * ox - c * oy], [0.0, 0.0, 1.0]]
    left, top = cx * src_w - w / 2, cy * src_h - h / 2
    m = [[k, 0.0, left], [0.0, k, top], [0.0, 0.0, 1.0]]
    return (np.array(m) @ np.array(a)).tolist()


def needs_prewarp(cam, shot=None) -> bool:
    if cam.type not in ("compound", "keyframe_path"):
        return False
    if cam.projection == "rotate" or (shot is not None and shot.motion_regions) or getattr(cam, "punches", None):
        return True
    return cam.type == "keyframe_path" and any(k.roll_deg for k in cam.keyframes or [])


def homography_maps(m: np.ndarray, grid: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Source sampling maps (float32) for OUTPUT px -> SOURCE px homography m, on a regular grid
    (meshgrid of output cols / rows). Each of the three projective terms is a column part plus a
    row part, computed in float64 per row / col and summed per pixel in float32 (remap resolves
    1/32 px; this differs from all-float64 by < 0.001 px) - full-frame float64 passes were the
    prewarp's main memory traffic."""
    x = np.asarray(grid[0][0], np.float64)          # output cols
    y = np.asarray(grid[1][:, 0], np.float64)       # output rows
    ax, ay = (m[0, 0] * x).astype(np.float32), (m[0, 1] * y + m[0, 2]).astype(np.float32)
    bx, by = (m[1, 0] * x).astype(np.float32), (m[1, 1] * y + m[1, 2]).astype(np.float32)
    map_x = ay[:, None] + ax[None, :]
    map_y = by[:, None] + bx[None, :]
    if not (m[2, 0] == 0 and m[2, 1] == 0 and m[2, 2] == 1):     # projective (rotate): divide by w
        wx, wy = (m[2, 0] * x).astype(np.float32), (m[2, 1] * y + m[2, 2]).astype(np.float32)
        w = wy[:, None] + wx[None, :]
        np.divide(map_x, w, out=map_x)
        np.divide(map_y, w, out=map_y)
    return map_x, map_y


_EASINGS = {"linear": lambda t: t, "ease_in": lambda t: t * t,
            "ease_out": lambda t: 1 - (1 - t) * (1 - t), "ease_in_out": lambda t: t * t * (3 - 2 * t)}


def ken_burns_frames(cam, n: int, ts: list | None = None) -> list:
    """Plan <= 1.2 "ken_burns" cameras (start -> end with easing), exactly as the browser drew
    them (remotion/src/math.ts cameraAt): centre linear, zoom in log space, roll linear."""
    ease = _EASINGS.get(getattr(cam, "easing", None) or "ease_in_out", _EASINGS["ease_in_out"])
    a, b = cam.start, cam.end
    out = []
    for f in range(n):
        p = ease(max(0.0, min(1.0, ts[f] if ts is not None else f / max(1, n - 1))))
        z = a.zoom * (b.zoom / a.zoom) ** p if a.zoom > 0 and b.zoom > 0 else a.zoom + (b.zoom - a.zoom) * p
        r0, r1 = getattr(a, "roll_deg", None) or 0.0, getattr(b, "roll_deg", None) or 0.0
        out.append(((a.cx + (b.cx - a.cx) * p, a.cy + (b.cy - a.cy) * p, z), r0 + (r1 - r0) * p))
    return out


def punch_scale(punches, t: float) -> float:
    """Zoom factor of the shot's punches at t s into the shot (beat sync): each rises over its
    attack (smoothstep) to 1 + amount at at_s and eases back over decay_s ((1-x)^2, fast then
    soft - a hit, not a breath). Zoom IN only, about the frame centre: the frame stays inside the
    window the planner validated."""
    z = 1.0
    for p in punches or []:
        dt = t - p.at_s
        if dt < 0:
            if p.attack_s <= 0 or dt <= -p.attack_s:
                continue
            x = 1.0 + dt / p.attack_s
            w = x * x * (3 - 2 * x)
        elif dt < p.decay_s:
            w = (1.0 - dt / p.decay_s) ** 2
        else:
            continue
        z *= 1.0 + p.amount * w
    return z


# A region whose motion would not read on screen is left still (the backend's rule "too small on
# screen to read as motion", applied with the real leaf-weighted mask and the real camera path;
# the backend sees only a bounding box). Review 2026-10-19: trees whose on-screen part was trunk,
# hedge or low branches - held still by leaf_weight - moved 0-0.6 px and failed the quality gate.
VISIBLE_MOTION_PX = 2.0         # = tools/perceptual_quality LIMITS["motion_min_p95_px"], at 1080p


def visible_motion_px(ow: int, oh: int) -> float:
    """The readability floor for this output (reviewed at 1080p; scales with the short side)."""
    return VISIBLE_MOTION_PX * min(ow, oh) / 1080
MOTION_PROBE_FRAMES = 9
MOTION_PROBE_STEP = 6           # output px between probe points (exact field at each point)


def probe_frames(n: int) -> list[int]:
    return sorted({round(i * (n - 1) / (MOTION_PROBE_FRAMES - 1)) for i in range(MOTION_PROBE_FRAMES)})


def region_motion_px(sw, region, frames: list[int] | None = None) -> float:
    """How far a region's content moves on screen: p95 of its moving points' displacement
    (screen px), the most over the probe frames. Exact field on a coarse output grid."""
    xs = np.arange(0, sw.ow, MOTION_PROBE_STEP, dtype=np.float64)
    ys = np.arange(0, sw.oh, MOTION_PROBE_STEP, dtype=np.float64)
    grid = np.meshgrid(xs, ys)
    best = 0.0
    for f in probe_frames(sw.n) if frames is None else frames:
        bx, by = homography_maps(sw.homography(f), grid)
        scale = MOTION_PROBE_STEP / max(1e-9, float(np.median(np.abs(np.diff(bx, axis=1)))))
        mx, my = bx.copy(), by.copy()
        apply_regions(mx, my, [region], f / sw.fps, step=1)
        d = np.hypot(mx - bx, my - by)
        if (d > 0).any():
            best = max(best, float(np.percentile(d[d > 0], 95)) * scale)
    return best


class ShotWarp:
    """Everything needed to draw any frame of one prewarped shot (shared by the render, the
    parallel workers and tools/perceptual_quality.py, so they cannot drift apart)."""

    def __init__(self, plan, shot, n: int, images: str, warnings: list | None = None):
        from modules.plan_io import compound_path  # plan_io imports camera_path; avoid a cycle
        cam = shot.camera
        self.shot, self.n, self.fps = shot, n, plan.output.fps
        self.ow, self.oh = plan.output.width, plan.output.height
        path = (compound_path(shot, self.ow, self.oh) if cam.type == "compound"
                else KeyframePath([k.model_dump() for k in cam.keyframes]) if cam.type == "keyframe_path"
                else None)                                   # ken_burns: start -> end (below)
        self.src = cv2.imread(os.path.join(images, shot.image_uri.split("/")[-1]), cv2.IMREAD_COLOR)
        if self.src is None:
            raise RuntimeError(f"prewarp: cannot read the image for shot {shot.shot_id}")
        sh, sw = self.src.shape[:2]
        pw, ph = shot.source.width, shot.source.height
        if (sw, sh) != (pw, ph):
            # the browser path drew every image at the plan's size; same here, if it is the same
            # photo (same aspect) - a different aspect means a different image: refuse
            if abs(sw / sh - pw / ph) > 0.01 * (pw / ph):
                raise RuntimeError(f"prewarp: shot {shot.shot_id} image is {sw}x{sh}, plan says {pw}x{ph}")
            interp = cv2.INTER_AREA if sw > pw else cv2.INTER_CUBIC
            self.src = cv2.resize(self.src, (pw, ph), interpolation=interp)
            if warnings is not None:
                warnings.append(f"image_resized:{shot.shot_id}:{sw}x{sh}->{pw}x{ph}")
        self.sw, self.sh = sw, sh
        self.rotate = cam.projection == "rotate"
        self.hfov = cam.hfov_deg if cam.hfov_deg is not None else DEFAULT_HFOV_DEG
        warp = getattr(cam, "time_warp", None)
        ts = [time_warp_at(warp, f / max(1, n - 1)) for f in range(n)]    # beat-sync retiming
        if path is None:
            self.frames = ken_burns_frames(cam, n, ts)
        else:
            rolls = ([path.roll_deg_at(t) for t in ts] if cam.type == "keyframe_path" else [0.0] * n)
            self.frames = list(zip((path.at(t) for t in ts), rolls))
        if getattr(cam, "punches", None):                                   # beat-sync punches
            self.frames = [((cx, cy, z * punch_scale(cam.punches, f / self.fps)), r)
                           for f, ((cx, cy, z), r) in enumerate(self.frames)]
        self.regions, region_warnings = load_regions(shot, images, sw, sh, self.src)
        self.motion_px = {}
        keep = []
        for r in self.regions:
            px = region_motion_px(self, r)
            self.motion_px[r.spec.id] = px
            # fire lives mostly in its light (2026-10-21): the displacement floor does not apply
            if r.effect == "fire_flicker" or px >= visible_motion_px(self.ow, self.oh):
                keep.append(r)
            else:
                region_warnings.append(f"motion_region_dropped:{shot.shot_id}:{r.spec.id}:faint_{px:.1f}px")
        self.regions = keep
        if warnings is not None:
            warnings.extend(region_warnings)
        self.grid = (np.meshgrid(np.arange(self.ow, dtype=np.float64), np.arange(self.oh, dtype=np.float64))
                     if self.regions else None)
        self.align = field_align(self.regions)
        self.probe_ys = np.unique(np.r_[np.arange(0, self.oh, ROI_PROBE), self.oh - 1])
        self.probe_xs = np.unique(np.r_[np.arange(0, self.ow, ROI_PROBE), self.ow - 1])

    def homography(self, f: int) -> np.ndarray:
        (cx, cy, z), r = self.frames[f]
        return np.array(rotate_homography(cx, cy, z, self.sw, self.sh, self.ow, self.oh, self.hfov) if self.rotate
                        else crop_roll_homography(cx, cy, z, r, self.sw, self.sh, self.ow, self.oh))

    def maps(self, f: int, step: int | None = None):
        """Full-frame (map_x, map_y, gain or None) of frame f with the motion regions applied
        (the reference for tools/perceptual_quality.py; frame() only builds the moving window)."""
        map_x, map_y = homography_maps(self.homography(f), self.grid)
        gain = apply_regions(map_x, map_y, self.regions, f / self.fps, step=step)
        return map_x, map_y, gain

    def window(self, m: np.ndarray):
        """(r0, r1, c0, c1): the part of the frame where any region can be, aligned so the
        regions' coarse field grids fit inside; None when no region is on screen."""
        px, py = np.meshgrid(self.probe_xs.astype(np.float64), self.probe_ys.astype(np.float64))
        pu, pv = homography_maps(m, (px, py))
        boxes = [b for r in self.regions
                 if (b := probe_box(r, pu, pv, self.probe_ys, self.probe_xs, self.oh, self.ow)) is not None]
        if not boxes:
            return None
        a = self.align
        r0, c0 = min(b[0] for b in boxes) // a * a, min(b[2] for b in boxes) // a * a
        r1 = min(self.oh - 1, -(-(max(b[1] for b in boxes) - 1) // a) * a) + 1
        c1 = min(self.ow - 1, -(-(max(b[3] for b in boxes) - 1) // a) * a) + 1
        return r0, r1, c0, c1

    def frame(self, f: int, step: int | None = None) -> np.ndarray:
        """The camera warp of the whole frame, then the window with moving regions redrawn
        through the displaced maps (warpPerspective and remap agree within 1 level)."""
        m = self.homography(f)
        frame = cv2.warpPerspective(self.src, m, (self.ow, self.oh), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        win = self.window(m) if self.regions else None
        if win is None:
            return frame
        r0, r1, c0, c1 = win
        map_x, map_y = homography_maps(m, (self.grid[0][r0:r1, c0:c1], self.grid[1][r0:r1, c0:c1]))
        gain = apply_regions(map_x, map_y, self.regions, f / self.fps, step=step, origin=(r0, c0),
                             frame_shape=(self.oh, self.ow))
        patch = cv2.remap(self.src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if gain is not None:            # calm_water light follows the wave slope
            patch = np.clip(patch.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
        frame[r0:r1, c0:c1] = patch
        return frame


# Frames are independent (frame f depends only on f), so they are drawn in parallel processes
# (modules/direct_render.py); the output is byte-identical to drawing them one by one.
WORKER_MEM_BYTES = 600 * 2**20          # a generous per-worker peak (4K photo, 3 masks, 1080p maps)


def _available_memory() -> int | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def prewarp_workers() -> int:
    env = os.environ.get("PREWARP_WORKERS", "").strip()
    if env:
        return max(1, int(env))
    n = min(8, len(os.sched_getaffinity(0)) - 1)          # CPUs this job may use
    mem = _available_memory()
    if mem is not None:
        n = min(n, int(mem * 0.6 // WORKER_MEM_BYTES))   # never crowd the machine's memory
    return max(1, n)
