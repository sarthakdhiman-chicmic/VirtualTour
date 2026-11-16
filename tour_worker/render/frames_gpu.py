"""Photo frames drawn on the GPU (torch): ShotWarp.frame()'s work - the camera warp of the whole frame,
then the window with moving regions redrawn through displaced maps - as tensor operations.

Why (user 2026-11-16): a 4K photo tour on the A100 pod spent 742 s drawing and encoding on the CPU;
drawing (the motion fields above all) was most of the CPU it used. On the GPU a frame is drawn in
milliseconds and the CPU is left to x264 - the same split as 3D tours (render/splat_shot.py).

Everything a shot sets up once (camera frames, regions and their masks, the sky, the window probe)
is ShotWarp's: GpuShot wraps one and only replaces the per-frame pixel work. The effects run the
same code as on the CPU (tour_worker.motion.xp: their fields take torch tensors), the coarse field
grids are the CPU path's (motion.region), and every sample is bilinear with pixel centres at
integers (grid_sample, align_corners=True) as OpenCV's remap - frames match the CPU path to within
a level or two (tests/render/test_frames_gpu.py), and are deterministic for a given GPU and driver.
"""
from __future__ import annotations

import threading

import numpy as np

from tour_worker.motion import region as R

GPU_LOCK = threading.Lock()


def photo_device(backend: str = "auto"):
    """The torch device photo frames are drawn on, or None for the CPU path: "auto" / "gpu" use
    CUDA when it is there, "cpu" never. (None also when torch is missing.)"""
    if backend == "cpu":
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None      # one frame at a time on the device: identical frames however many encoders run


def _t():
    import torch
    return torch


def _sample(img, u, v, border: str):
    """img (H, W, C) or (H, W) float32 tensor at points (u, v) (any shape, source px): bilinear,
    border 'border' (replicate) or 'zeros'. Returns (*u.shape, C) or u.shape."""
    torch = _t()
    import torch.nn.functional as F
    h, w = img.shape[:2]
    chw = img.permute(2, 0, 1)[None] if img.dim() == 3 else img[None, None]
    gx = (u.to(torch.float32) / max(w - 1, 1)) * 2 - 1
    gy = (v.to(torch.float32) / max(h - 1, 1)) * 2 - 1
    grid = torch.stack([gx.reshape(1, 1, -1), gy.reshape(1, 1, -1)], -1)
    out = F.grid_sample(chw, grid, mode="bilinear", padding_mode=border, align_corners=True)[0]   # (C, 1, N)
    if img.dim() == 3:
        return out[:, 0].permute(1, 0).reshape(*u.shape, img.shape[2])
    return out[0, 0].reshape(u.shape)


class GpuShot:
    """frame(f) -> BGR uint8 (numpy) of one photo shot, drawn on `device`."""

    def __init__(self, sw, device):
        torch = _t()
        self.sw, self.dev = sw, device
        self.src = torch.as_tensor(sw.src, device=device).to(torch.float32)
        ys = torch.arange(sw.oh, dtype=torch.float64, device=device)
        xs = torch.arange(sw.ow, dtype=torch.float64, device=device)
        self.xs, self.ys = xs, ys
        self.feathers = {id(r): torch.as_tensor(np.ascontiguousarray(r.feather), device=device).to(torch.float32)
                         for r in sw.regions}
        self.k = R.grid_scale((sw.oh, sw.ow))

    # ---- maps -------------------------------------------------------------------------------
    def maps(self, m: np.ndarray, r0=0, r1=None, c0=0, c1=None):
        """homography_maps on the device: OUTPUT px -> SOURCE px for rows r0:r1, cols c0:c1, float32
        (each projective term a row part plus a column part, as on the CPU)."""
        torch = _t()
        x, y = self.xs[c0:c1], self.ys[r0:r1]
        ax, ay = (m[0, 0] * x).to(torch.float32), (m[0, 1] * y + m[0, 2]).to(torch.float32)
        bx, by = (m[1, 0] * x).to(torch.float32), (m[1, 1] * y + m[1, 2]).to(torch.float32)
        mx = ay[:, None] + ax[None, :]
        my = by[:, None] + bx[None, :]
        if not (m[2, 0] == 0 and m[2, 1] == 0 and m[2, 2] == 1):
            wx, wy = (m[2, 0] * x).to(torch.float32), (m[2, 1] * y + m[2, 2]).to(torch.float32)
            wgt = wy[:, None] + wx[None, :]
            mx, my = mx / wgt, my / wgt
        return mx, my

    # ---- motion regions (motion.region.apply_regions on tensors) ---------------------------
    def _screen_box(self, r, mx, my):
        torch = _t()
        h, w = mx.shape
        ys, xs = R._sample_grid(h, w, R.ROI_PROBE, R.ROI_PROBE)
        iy, ix = torch.as_tensor(ys, device=self.dev), torch.as_tensor(xs, device=self.dev)
        pu = mx.index_select(0, iy).index_select(1, ix).cpu().numpy()
        pv = my.index_select(0, iy).index_select(1, ix).cpu().numpy()
        spacing = max([1e-6] + [float(np.median(np.abs(np.diff(c, axis=ax)))) for c, ax in ((pu, 1), (pv, 0))
                                if c.shape[ax] > 1])
        return R.probe_box(r, pu, pv, ys, xs, h, w), R.ROI_PROBE / spacing

    def _coarse_eval(self, r, fn, n_out, mx, my, step, box, origin, frame_shape):
        torch = _t()
        oy, ox = origin
        ys, xs = R._sample_grid(*frame_shape, *step)
        r0, r1, c0, c1 = box
        i0, i1 = np.searchsorted(ys, r0, "right") - 1, np.searchsorted(ys, r1 - 1, "left")
        j0, j1 = np.searchsorted(xs, c0, "right") - 1, np.searchsorted(xs, c1 - 1, "left")
        iy = torch.as_tensor(ys[i0:i1 + 1] - oy, device=self.dev)
        ix = torch.as_tensor(xs[j0:j1 + 1] - ox, device=self.dev)
        cu = mx.index_select(0, iy).index_select(1, ix).to(torch.float64)
        cv = my.index_select(0, iy).index_select(1, ix).to(torch.float64)
        cell = 1.0
        for c in (cu, cv):
            for ax in (0, 1):
                if c.shape[ax] > 1:
                    cell = max(cell, float(torch.diff(c, dim=ax).abs().max()))
        mg = 1.5 * cell
        need = (cu >= r.x0 - mg) & (cu <= r.x0 + r.bw + mg) & (cv >= r.y0 - mg) & (cv <= r.y0 + r.bh + mg)
        anyneed = bool(need.any())
        vals = fn(cu[need], cv[need]) if anyneed else [None] * n_out
        G = (slice(int(ys[i0]), int(ys[i1]) + 1), slice(int(xs[j0]), int(xs[j1]) + 1))
        # the full-res pixels' positions in the coarse grid (motion.region._coarse_grid's np.interp,
        # piecewise linear over the uneven last cell), made here on the device: built on the CPU and
        # uploaded per region per frame (up to 2 x 32 MB at 4K) they were most of a frame's time
        fx = self._grid_pos(xs, step[1], frame_shape[1])[G[1]] - float(j0)
        fy = self._grid_pos(ys, step[0], frame_shape[0])[G[0]] - float(i0)
        lx, ly = torch.broadcast_tensors(fx[None, :], fy[:, None])
        out = []
        for k, val in enumerate(vals):
            if val is None:
                out.append(None)
                continue
            c = torch.full(cu.shape, 1.0 if k == 2 else 0.0, dtype=torch.float32, device=self.dev)
            c[need] = val.to(torch.float32) if torch.is_tensor(val) else torch.tensor(float(val), device=self.dev)
            out.append(_sample(c, lx, ly, "border"))
        if not anyneed and n_out >= 2:
            z = torch.zeros(lx.shape, dtype=torch.float32, device=self.dev)
            out[:2] = [z, z]
        Rs = (slice(G[0].start - oy, G[0].stop - oy), slice(G[1].start - ox, G[1].stop - ox))
        return Rs, out

    def _grid_pos(self, pts: np.ndarray, step: int, n: int):
        """np.interp(arange(n), pts, arange(len(pts))) as a float32 device vector, cached."""
        torch = _t()
        cache = self.__dict__.setdefault("_gpos", {})
        key = (len(pts), int(pts[-1]), step, n)
        if key not in cache:
            p = torch.as_tensor(pts, dtype=torch.float64, device=self.dev)
            x = torch.arange(n, dtype=torch.float64, device=self.dev)
            k = (torch.bucketize(x, p, right=True) - 1).clamp_(0, len(pts) - 2)
            cache[key] = (k + (x - p[k]) / (p[k + 1] - p[k])).to(torch.float32)
        return cache[key]

    def apply_regions(self, mx, my, t: float, origin, frame_shape):
        """motion.region.apply_regions on device maps (in place); returns the gain or None."""
        torch = _t()
        gain = None
        k = self.k
        oy, ox = origin
        for r in self.sw.regions:
            box, screen_per_src = self._screen_box(r, mx, my)
            if box is None:
                continue
            st = R.adaptive_step(r, R.field_step(r.effect, k), screen_per_src, k)
            gbox = (box[0] + oy, box[1] + oy, box[2] + ox, box[3] + ox)
            smooth = R.smooth_step(r.effect, k)
            irr = None
            if max(st) > 1:
                Rs, (fdx, fdy, fb) = self._coarse_eval(r, lambda u, v: r.fields(u, v, t), 3, mx, my, st, gbox,
                                                       origin, frame_shape)
            elif smooth:
                Rs, (irr,) = self._coarse_eval(r, lambda u, v: (r.calm_irregular(u, v, t),), 1, mx, my, smooth,
                                               gbox, origin, frame_shape)
            else:
                Rs = (slice(box[0], box[1]), slice(box[2], box[3]))
            u, v = mx[Rs], my[Rs]
            w = _sample(self.feathers[id(r)], u, v, "zeros")
            ww = torch.where(w > 1e-3, w, torch.zeros_like(w)).to(torch.float64)
            if not bool((ww > 0).any()):
                continue
            u64, v64 = u.to(torch.float64), v.to(torch.float64)
            if max(st) <= 1:
                dx, dy, b = r.fields(u64, v64, t, None if irr is None else irr.to(torch.float64))
            else:
                dx, dy = fdx.to(torch.float64), fdy.to(torch.float64)
                b = fb.to(torch.float64) if fb is not None else None
            if b is not None:
                if gain is None:
                    gain = torch.ones(mx.shape, dtype=torch.float32, device=self.dev)
                gain[Rs] = gain[Rs] * (1.0 + ww * (b - 1.0)).to(torch.float32)
            mx[Rs] = (u64 - ww * dx).to(torch.float32)
            my[Rs] = (v64 - ww * dy).to(torch.float32)
        return gain

    # ---- the frame ---------------------------------------------------------------------------
    def frame(self, f: int) -> np.ndarray:
        return self.frame_t(f).cpu().numpy()

    def frame_t(self, f: int, stats: dict | None = None):
        """Frame f as a uint8 (oh, ow, 3) tensor on the device. The moving sky (CPU, OpenCV - it
        releases the GIL) is made before the GPU lock, so parallel encoders make theirs at once."""
        import time
        torch = _t()
        sw = self.sw
        t0 = time.perf_counter()
        sky = sw.sky.frame(f / sw.fps) if sw.sky is not None else None
        t1 = time.perf_counter()
        with GPU_LOCK, torch.no_grad():
            t2 = time.perf_counter()
            m = sw.homography(f)
            src = torch.as_tensor(sky, device=self.dev).to(torch.float32) if sky is not None else self.src
            mx, my = self.maps(m)
            img = _sample(src, mx, my, "border")                            # (oh, ow, 3)
            win = sw.window(m) if sw.regions else None
            if win is not None:
                r0, r1, c0, c1 = win
                wx, wy = mx[r0:r1, c0:c1].clone(), my[r0:r1, c0:c1].clone()
                gain = self.apply_regions(wx, wy, f / sw.fps, (r0, c0), (sw.oh, sw.ow))
                patch = _sample(src, wx, wy, "border")
                if gain is not None:
                    patch = patch * gain[..., None]
                img[r0:r1, c0:c1] = patch
            out = torch.round(img).clamp_(0, 255).to(torch.uint8)
        if stats is not None:
            t3 = time.perf_counter()
            _add(stats, sky_s=t1 - t0, gpu_wait_s=t2 - t1, gpu_draw_s=t3 - t2)
        return out


_STATS_LOCK = threading.Lock()


def _add(stats: dict, **kw) -> None:
    with _STATS_LOCK:
        for k, v in kw.items():
            stats[k] = stats.get(k, 0.0) + v
