"""One wind per shot (2026-10-22): every tree, plant and curtain in a shot responds to the same
breeze (fire and water keep their own motion), instead of each running its own random timing - the scene reads as one real
moment.

The wind is seeded from the shot, so renders stay deterministic and the plan contract does not
change. It has a direction (left-to-right or right-to-left across the frame) and gusts at
irregular times; a gust takes WIND_CROSS_S to cross the frame, so each object feels it with a
delay set by where it stands. Objects keep their own timing (a tree's sway, the curtains vs
plants phases); the wind sets how STRONG that motion is (envelope) and, for trees, a downwind lean
during a gust. The delay is per object (a tree's centre), never across one crown: parts of one
tree reversing at different moments reads as rotation (review 2026-10-22)."""
from __future__ import annotations

import zlib

import numpy as np

WIND_CROSS_S = 2.0          # a gust crosses the whole frame in this time
GUST_GAP_S = (4.0, 9.0)     # time between gusts
GUST_LEN_S = (0.9, 1.7)     # a gust's half-width (Gaussian)
CALM = 0.7                  # motion strength between gusts (1.0 at a gust's peak)


class Wind:
    def __init__(self, seed: int, src_w: int, duration_s: float = 130.0):
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        self.direction = 1 if rng.random() < 0.5 else -1        # +1: blowing left to right
        self.src_w = max(1, int(src_w))
        times, t = [], -float(rng.uniform(0.0, GUST_GAP_S[1]))
        while t < duration_s + GUST_GAP_S[1]:
            times.append(t)
            t += float(rng.uniform(*GUST_GAP_S))
        self.times = np.array(times, np.float64)
        self.amps = rng.uniform(0.6, 1.0, len(times))
        self.widths = rng.uniform(*GUST_LEN_S, len(times))

    @classmethod
    def for_shot(cls, shot_id: str, salt: int, src_w: int) -> "Wind":
        return cls(zlib.crc32(f"{shot_id}|wind|{salt}".encode()), src_w)

    def delay(self, x: float) -> float:
        """Seconds until a gust reaching the frame's upwind edge reaches source column x."""
        frac = min(1.0, max(0.0, x / self.src_w))
        return WIND_CROSS_S * (frac if self.direction > 0 else 1.0 - frac)

    def gust(self, t: float) -> float:
        """0 (calm) .. 1 (a gust's peak) at time t, smooth."""
        g = float(np.sum(self.amps * np.exp(-(((t - self.times) / self.widths) ** 2))))
        return min(1.0, g)

    def envelope(self, t: float) -> float:
        """Strength of the motion the wind drives: CALM between gusts, 1 at a gust's peak (never
        above the planned amplitude)."""
        return CALM + (1.0 - CALM) * self.gust(t)


def shot_wind(shot, src_w: int) -> Wind:
    """The shot's wind, seeded from the shot id and its regions' seeds (deterministic)."""
    salt = 0
    for spec in getattr(shot, "motion_regions", None) or []:
        salt = (salt * 31 + int(spec.seed)) & 0xFFFFFFFF
    return Wind.for_shot(str(shot.shot_id), salt, src_w)


def envelope_at(r, t: float) -> float:
    """A region's wind envelope at t (1.0 when the region has no wind attached)."""
    w = getattr(r, "wind", None)
    return 1.0 if w is None else w.envelope(t - getattr(r, "wind_delay", 0.0))
