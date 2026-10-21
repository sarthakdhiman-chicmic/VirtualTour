"""Seeded 3D gradient noise (classic Perlin construction), vectorised NumPy, float64.

Used for organic, smooth-in-time motion (foliage sway, fabric ripple). No dependency: the noise
libraries (`noise`, `opensimplex`, `pyfastnoiselite`) are not installed, and our own version is
deterministic by construction (permutation from a seeded generator, pure arithmetic).
Output is in about [-1, 1], continuous with a continuous first derivative (quintic fade).
"""
from __future__ import annotations

import numpy as np

_GRAD3 = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                   [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                   [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float64)
_GX, _GY, _GZ = (np.ascontiguousarray(_GRAD3[:, i]) for i in range(3))


class GradientNoise3D:
    def __init__(self, seed: int):
        p = np.random.default_rng(seed & 0xFFFFFFFF).permutation(256)
        self.perm = np.concatenate([p, p]).astype(np.int64)
        self.perm12 = self.perm % 12        # gradient index per hash, looked up once

    @staticmethod
    def _fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _grad(self, h, x, y, z):
        # same arithmetic as _GRAD3[h % 12] . (x, y, z), without an (N, 3) temporary
        g = self.perm12[h]
        return _GX[g] * x + _GY[g] * y + _GZ[g] * z

    def __call__(self, x, y, z):
        x, y, z = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64)
        x, y, z = np.broadcast_arrays(x, y, z)
        xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
        xf, yf, zf = x - xi, y - yi, z - zi
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
