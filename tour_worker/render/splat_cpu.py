"""A CPU reference rasteriser for Gaussian splat scenes (plan 1.6), in vectorised torch.

The production path is gsplat on CUDA (render/splat_shot.py). This one exists so a scene can be
checked and previewed on a machine without a GPU, and as the reference image the GPU path is
compared against. It follows the 3DGS forward pass (as gsplat's "classic" mode):

  project each Gaussian (EWA: J W Sigma W^T J^T + 0.3 px low-pass), 3-sigma screen radius,
  view-dependent colour from spherical harmonics, then per pixel front-to-back alpha compositing
  in depth order with alpha = min(0.99, opacity * exp(-d^T Sigma2^-1 d / 2)), alpha >= 1/255,
  stopping once transmittance falls below 1e-4.

Compositing is exact and order-independent of thread scheduling (sorted keys, one segmented scan),
so a frame is deterministic. Memory is bounded by rendering in horizontal bands.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from tour_worker.scene.splat_io import SplatModel

NEAR = 0.05                  # scene units in front of the camera
LOW_PASS = 0.3               # px^2 added to the screen covariance (3DGS anti-aliasing floor)
ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_MIN = 1e-4
BAND_ROWS = 16
MAX_RADIUS_FRAC = 1.0        # a splat's radius is capped at this x the larger image side
FRUSTUM_MARGIN = 1.3         # keep splats whose centre is within 1.3x the image half-extent
PROJECT_CHUNK = 1_000_000    # splats projected at once
PAIR_BUDGET = 6_000_000      # (splat, pixel) pairs composited at once

_SH1 = 0.4886025119029199
_SH2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_SH3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
        -0.4570457994644658, 1.445305721320277, -0.5900435899266435)
_SH0 = 0.28209479177387814


def sh_to_rgb(sh: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """(M, K, 3) coefficients, (M, 3) unit view directions -> (M, 3) colour, clamped >= 0 (3DGS)."""
    c = _SH0 * sh[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        c = c - _SH1 * y * sh[:, 1] + _SH1 * z * sh[:, 2] - _SH1 * x * sh[:, 3]
        if degree >= 2:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            c = (c + _SH2[0] * xy * sh[:, 4] + _SH2[1] * yz * sh[:, 5] + _SH2[2] * (2 * zz - xx - yy) * sh[:, 6]
                 + _SH2[3] * xz * sh[:, 7] + _SH2[4] * (xx - yy) * sh[:, 8])
            if degree >= 3:
                c = (c + _SH3[0] * y * (3 * xx - yy) * sh[:, 9] + _SH3[1] * xy * z * sh[:, 10]
                     + _SH3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                     + _SH3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                     + _SH3[4] * x * (4 * zz - xx - yy) * sh[:, 13] + _SH3[5] * z * (xx - yy) * sh[:, 14]
                     + _SH3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return torch.clamp(c + 0.5, min=0.0)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) unit quaternions w x y z -> (N, 3, 3)."""
    w, x, y, z = q.unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


class CpuSplatRenderer:
    """Holds a scene's tensors (and each Gaussian's 3D covariance, computed once) and renders views."""

    def __init__(self, model: SplatModel, sh_degree: int | None = None, background=(0.0, 0.0, 0.0)):
        self.means = torch.from_numpy(model.means)
        self.opac = torch.from_numpy(model.opacities)
        self.sh = torch.from_numpy(model.sh)
        self.degree = model.sh_degree if sh_degree is None else min(sh_degree, model.sh_degree)
        self.bg = torch.tensor(background, dtype=torch.float32)
        r = quat_to_rotmat(torch.from_numpy(model.quats))
        m = r * torch.from_numpy(model.scales)[:, None, :]          # R S
        cov = m @ m.transpose(1, 2)                                 # R S S^T R^T
        self.cov6 = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                                 cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], 1).contiguous()
        del r, m, cov

    @torch.no_grad()
    def render(self, viewmat: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
        """(H, W, 3) float32 image, 0-1 (colour may exceed 1 slightly; the caller clips)."""
        R = torch.from_numpy(np.asarray(viewmat[:3, :3], np.float32))
        t = torch.from_numpy(np.asarray(viewmat[:3, 3], np.float32))
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        campos = -(R.T @ t)
        parts = [self._project(a, min(a + PROJECT_CHUNK, len(self.means)), R, t, campos, fx, fy, cx, cy, width, height)
                 for a in range(0, len(self.means), PROJECT_CHUNK)]
        parts = [q for q in parts if q is not None]
        if not parts:
            return np.broadcast_to(self.bg.numpy(), (height, width, 3)).copy()
        z, u, v, radius, ca, cb, cc, rgb, op = (torch.cat(col) for col in zip(*parts))
        # depth order (front to back); stable, so ties keep file order: deterministic
        order = torch.argsort(z, stable=True)
        u, v, radius, ca, cb, cc, rgb, op = (q[order] for q in (u, v, radius, ca, cb, cc, rgb, op))
        x0 = torch.clamp(torch.floor(u - radius), 0, width - 1).long()
        x1 = torch.clamp(torch.floor(u + radius), 0, width - 1).long()
        y0 = torch.clamp(torch.floor(v - radius), 0, height - 1).long()
        y1 = torch.clamp(torch.floor(v + radius), 0, height - 1).long()
        out = torch.empty(height, width, 3)
        for band in range(0, height, BAND_ROWS):
            b1 = min(height, band + BAND_ROWS) - 1
            hit = torch.nonzero((y0 <= b1) & (y1 >= band)).squeeze(1)       # still in depth order
            out[band:b1 + 1] = self._band(hit, band, b1, width, u, v, ca, cb, cc, rgb, op, x0, x1, y0, y1)
        return out.numpy()

    def _project(self, a, b, R, t, campos, fx, fy, cx, cy, width, height):
        """Splats a..b: the visible ones' (depth, screen centre, radius, conic, colour, opacity)."""
        pc = self.means[a:b] @ R.T + t
        z = pc[:, 2]
        zc = z.clamp(min=NEAR)
        lim_x, lim_y = FRUSTUM_MARGIN * (width / 2) / fx, FRUSTUM_MARGIN * (height / 2) / fy
        keep = (z > NEAR) & ((pc[:, 0] / zc).abs() < lim_x) & ((pc[:, 1] / zc).abs() < lim_y)
        idx = torch.nonzero(keep).squeeze(1)
        if idx.numel() == 0:
            return None
        pc, z = pc[idx], z[idx]
        # screen covariance: J (R Sigma R^T) J^T, with the projection's tangent clamped like 3DGS
        x = (pc[:, 0] / z).clamp(-lim_x, lim_x) * z
        y = (pc[:, 1] / z).clamp(-lim_y, lim_y) * z
        c6 = self.cov6[a:b][idx]
        S = torch.stack([c6[:, 0], c6[:, 1], c6[:, 2], c6[:, 1], c6[:, 3], c6[:, 4],
                         c6[:, 2], c6[:, 4], c6[:, 5]], 1).reshape(-1, 3, 3)
        J = torch.zeros(len(idx), 2, 3)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * x / (z * z)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * y / (z * z)
        T = J @ R                                   # (M, 2, 3)
        S2 = T @ S @ T.transpose(1, 2)
        del S, J, T
        sa = S2[:, 0, 0] + LOW_PASS
        sb = S2[:, 0, 1]
        sc = S2[:, 1, 1] + LOW_PASS
        det = sa * sc - sb * sb
        mid = 0.5 * (sa + sc)
        lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        radius = torch.ceil(3.0 * torch.sqrt(lam))
        u = fx * pc[:, 0] / z + cx
        v = fy * pc[:, 1] / z + cy
        rmax = MAX_RADIUS_FRAC * max(width, height)
        ok = (det > 1e-12) & (radius > 0) & (radius <= rmax) & (u + radius >= 0) & (u - radius < width) \
            & (v + radius >= 0) & (v - radius < height)
        sel = torch.nonzero(ok).squeeze(1)
        if sel.numel() == 0:
            return None
        g = idx[sel] + a
        inv = 1.0 / det[sel]
        dirs = self.means[g] - campos
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-12)
        rgb = sh_to_rgb(self.sh[g][:, :(self.degree + 1) ** 2], dirs, self.degree)
        return (z[sel], u[sel], v[sel], radius[sel], sc[sel] * inv, -sb[sel] * inv, sa[sel] * inv,
                rgb, self.opac[g])

    def _band(self, hit, band, b1, width, u, v, ca, cb, cc, rgb, op, x0, x1, y0, y1) -> torch.Tensor:
        """One band of rows: the splats in hit (front to back) composited in chunks of at most
        PAIR_BUDGET (splat, pixel) pairs; each pixel's transmittance carries over between chunks."""
        rows = b1 - band + 1
        img = torch.zeros(rows * width, 3)
        logt_acc = torch.zeros(rows * width, dtype=torch.float64)      # log transmittance so far
        if hit.numel():
            hy0, hy1 = torch.clamp(y0[hit], min=band), torch.clamp(y1[hit], max=b1)
            counts = (x1[hit] - x0[hit] + 1) * (hy1 - hy0 + 1)
            ends = torch.cumsum(counts, 0)
            a = 0
            while a < len(hit):
                lo = int(ends[a - 1]) if a else 0
                b = int(torch.searchsorted(ends, lo + PAIR_BUDGET, right=True))
                b = max(b, a + 1)
                self._composite(hit[a:b], hy0[a:b], hy1[a:b], counts[a:b], band, width, u, v, ca, cb, cc,
                                rgb, op, x0, x1, img, logt_acc)
                a = b
        img += torch.exp(logt_acc).float()[:, None] * self.bg
        return img.reshape(rows, width, 3)

    @staticmethod
    def _composite(hit, hy0, hy1, counts, band, width, u, v, ca, cb, cc, rgb, op, x0, x1, img, logt_acc):
        hx0 = x0[hit]
        bw = x1[hit] - hx0 + 1
        total = int(counts.sum())
        sid = torch.repeat_interleave(torch.arange(len(hit)), counts)
        starts = torch.cumsum(counts, 0) - counts
        local = torch.arange(total) - starts[sid]
        px = hx0[sid] + local % bw[sid]
        py = hy0[sid] + local // bw[sid]
        g = hit[sid]
        dx = (px.float() + 0.5) - u[g]
        dy = (py.float() + 0.5) - v[g]
        power = -0.5 * (ca[g] * dx * dx + cc[g] * dy * dy) - cb[g] * dx * dy
        alpha = torch.clamp(op[g] * torch.exp(power), max=ALPHA_MAX)
        live = (power <= 0) & (alpha >= ALPHA_MIN)
        del dx, dy, power, local
        sid, alpha = sid[live], alpha[live]
        pix = (py[live] - band) * width + px[live]
        if pix.numel() == 0:
            return
        # sort by (pixel, depth rank); sid is the depth rank within this chunk
        key = pix * len(hit) + sid
        _, perm = torch.sort(key, stable=True)
        pix, sid, alpha = pix[perm], sid[perm], alpha[perm].double()
        logt = torch.log1p(-alpha)
        cs = torch.cumsum(logt, 0)
        _, cnt = torch.unique_consecutive(pix, return_counts=True)
        first = torch.cumsum(cnt, 0) - cnt
        base = torch.where(first > 0, cs[(first - 1).clamp(min=0)], torch.zeros_like(cs[first]))
        base = torch.repeat_interleave(base, cnt)
        # transmittance before this splat: the pixel's carried-over value x this chunk's splats in front
        log_excl = cs - logt - base + logt_acc[pix]
        t_excl = torch.exp(log_excl)
        weight = torch.where(t_excl >= T_MIN, alpha * t_excl, torch.zeros_like(alpha)).float()
        img.index_add_(0, pix, weight[:, None] * rgb[hit[sid]])
        # the pixel's transmittance after the chunk: its carried value + the chunk's total log
        last = first + cnt - 1
        upix = pix[last]
        logt_acc[upix] = logt_acc[upix] + (cs[last] - base[last])
