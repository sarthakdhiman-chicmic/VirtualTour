"""Seeded 3D gradient noise (classic Perlin construction), vectorised NumPy, float32 (half the
memory traffic of float64; the noise is about +-1, far inside float32's precision - 4K render time).

Used for organic, smooth-in-time motion (foliage sway, fabric ripple). No dependency: the noise
libraries (`noise`, `opensimplex`, `pyfastnoiselite`) are not installed, and our own version is
deterministic by construction (permutation from a seeded generator, pure arithmetic).
Output is in about [-1, 1], continuous with a continuous first derivative (quintic fade).
"""
from __future__ import annotations

import numpy as np

_GRAD3 = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                   [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                   [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float32)
_GX, _GY, _GZ = (np.ascontiguousarray(_GRAD3[:, i]) for i in range(3))


class GradientNoise3D:
    def __init__(self, seed: int):
        p = np.random.default_rng(seed & 0xFFFFFFFF).permutation(256)
        self.perm = np.concatenate([p, p]).astype(np.int32)
        self.perm12 = self.perm % 12        # gradient index per hash, looked up once

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _grad(self, h, x, y, z):
        # same arithmetic as _GRAD3[h % 12] . (x, y, z), without an (N, 3) temporary
        g = self.perm12[h]
        return _GX[g] * x + _GY[g] * y + _GZ[g] * z

    def _torch_call(self, x, y, z):
        """The same construction on torch tensors (render/frames_gpu.py): x is a tensor, y / z
        tensors or scalars; float32 arithmetic, integer cells from float64 as on the CPU."""
        import torch
        ref = next(a for a in (x, y, z) if torch.is_tensor(a))
        dev = ref.device
        tabs = self.__dict__.setdefault("_dev", {})
        if dev not in tabs:
            tabs[dev] = (torch.as_tensor(self.perm, dtype=torch.int64, device=dev),
                         torch.as_tensor(self.perm12, dtype=torch.int64, device=dev),
                         *(torch.as_tensor(g, device=dev) for g in (_GX, _GY, _GZ)))
        P, P12, GX, GY, GZ = tabs[dev]
        x, y, z = (a.to(torch.float64) if torch.is_tensor(a) else torch.tensor(float(a), dtype=torch.float64, device=dev)
                   for a in (x, y, z))
        x, y, z = torch.broadcast_tensors(x, y, z)
        fx, fy, fz = torch.floor(x), torch.floor(y), torch.floor(z)
        xi, yi, zi = (fx.to(torch.int64) & 255), (fy.to(torch.int64) & 255), (fz.to(torch.int64) & 255)
        xf, yf, zf = (x - fx).to(torch.float32), (y - fy).to(torch.float32), (z - fz).to(torch.float32)
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)
        a, b = P[xi] + yi, P[xi + 1] + yi
        aa, ab, ba, bb = P[a] + zi, P[a + 1] + zi, P[b] + zi, P[b + 1] + zi

        def grad(h, x_, y_, z_):
            g = P12[h]
            return GX[g] * x_ + GY[g] * y_ + GZ[g] * z_

        def lerp(t, p0, p1):
            return p0 + t * (p1 - p0)

        x1 = lerp(u, grad(aa, xf, yf, zf), grad(ba, xf - 1, yf, zf))
        x2 = lerp(u, grad(ab, xf, yf - 1, zf), grad(bb, xf - 1, yf - 1, zf))
        x3 = lerp(u, grad(aa + 1, xf, yf, zf - 1), grad(ba + 1, xf - 1, yf, zf - 1))
        x4 = lerp(u, grad(ab + 1, xf, yf - 1, zf - 1), grad(bb + 1, xf - 1, yf - 1, zf - 1))
        return lerp(w, lerp(v, x1, x2), lerp(v, x3, x4))

    def __call__(self, x, y, z):
        if any(type(a).__module__.startswith("torch") for a in (x, y, z)):
            return self._torch_call(x, y, z)
        # the integer cell from float64 (a large coordinate keeps its exact cell), the rest in float32
        x, y, z = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64)
        x, y, z = np.broadcast_arrays(x, y, z)
        fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
        xi, yi, zi = fx.astype(np.int32), fy.astype(np.int32), fz.astype(np.int32)
        x, y, z = (x - fx).astype(np.float32), (y - fy).astype(np.float32), (z - fz).astype(np.float32)
        fx = fy = fz = None
        xf, yf, zf = x, y, z
        xi, yi, zi = xi & 255, yi & 255, zi & 255
        u, v, w = self._fade(xf), self._fade(yf), self._fade(zf)
        P = self.perm
        a, b = P[xi] + yi, P[xi + 1] + yi
        aa, ab, ba, bb = P[a] + zi, P[a + 1] + zi, P[b] + zi, P[b + 1] + zi

        def lerp(t, p0, p1):
            return p0 + t * (p1 - p0)

        x1 = lerp(u, self._grad(aa, xf, yf, zf), self._grad(ba, xf - 1, yf, zf))
        x2 = lerp(u, self._grad(ab, xf, yf - 1, zf), self._grad(bb, xf - 1, yf - 1, zf))
        x3 = lerp(u, self._grad(aa + 1, xf, yf, zf - 1), self._grad(ba + 1, xf - 1, yf, zf - 1))
        x4 = lerp(u, self._grad(ab + 1, xf, yf - 1, zf - 1), self._grad(bb + 1, xf - 1, yf - 1, zf - 1))
        return lerp(w, lerp(v, x1, x2), lerp(v, x3, x4))
