"""fabric_sway (curtains, blinds): anchored at the top; a slow wave travels down the fabric, folds
move slightly differently, a little low-frequency noise keeps it from looking mechanical.

Every function takes the Region (tour_worker.motion.region) as `r`."""
from __future__ import annotations

import math

import numpy as np


def feature_px(r) -> float:
    return min(max(40.0, 0.12 * r.bw), 0.5 * r.bw, 0.5 * r.bh)       # folds


def displacement(r, u, v, t):
    """(dx, dy) in source px at source points (u, v), time t seconds."""
    a, s = r.anchor_weight(v)
    A, f = r.amp, r.freq
    ph = 2 * math.pi * f * t - 2 * math.pi * 0.6 * s + r.phase
    fold = 1.0 + 0.35 * np.sin(2 * math.pi * (u - r.x0) / max(40.0, 0.12 * r.bw) + r.phase2)
    n = r.noise_a(u / (0.5 * r.bw), v / (0.5 * r.bh), 0.8 * f * t)
    dx = A * a * (0.75 * np.sin(ph) * fold + 0.35 * n)
    dy = 0.12 * A * a * np.cos(ph)
    return dx, dy


def fields(r, u, v, t, irregular=None):
    dx, dy = displacement(r, u, v, t)
    return dx, dy, None
