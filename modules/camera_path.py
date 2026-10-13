"""Camera path math for plan 1.3 `keyframe_path` cameras (contract addendum 1.3).

The renderer mirrors this file in remotion/src/math.ts (keyframePathAt); tests/test_camera_path.py
checks the two agree. Everything that judges a path (plan validation, the planner's margin
solver, the measurement scripts) evaluates frames through this module.

Per shot with n rendered frames, frame f has raw progress t = f / (n - 1) (as ken_burns).
1. Global time profile: u = hermite(t), cubic Hermite with p(0)=0, p(1)=1, start slope 1.6 and
   end slope 0.8, so a shot enters moving and is still moving at the cut. It is monotonic
   (minimum slope 0.7667 at t = 5/6), asserted below.
2. Path lookup: cx, cy and zoom are each interpolated over the keyframe times with a monotone
   cubic (Fritsch-Carlson): no overshoot between keyframes and continuous velocity at them. A
   channel that changes direction (or holds) at a keyframe has zero velocity there.
"""
from __future__ import annotations

import math

PROFILE_S0 = 1.6
PROFILE_S1 = 0.8
CHANNELS = ("cx", "cy", "zoom")


def hermite_profile(t: float, s0: float = PROFILE_S0, s1: float = PROFILE_S1) -> float:
    t = min(1.0, max(0.0, t))
    t2, t3 = t * t, t * t * t
    return (t3 - 2 * t2 + t) * s0 + (-2 * t3 + 3 * t2) + (t3 - t2) * s1


def hermite_profile_slope(t: float, s0: float = PROFILE_S0, s1: float = PROFILE_S1) -> float:
    return (3 * t * t - 4 * t + 1) * s0 + (-6 * t * t + 6 * t) + (3 * t * t - 2 * t) * s1


def assert_profile_monotonic(s0: float = PROFILE_S0, s1: float = PROFILE_S1, steps: int = 10000) -> float:
    lo = min(hermite_profile_slope(i / steps, s0, s1) for i in range(steps + 1))
    assert lo > 0.0, f"time profile not monotonic (min slope {lo})"
    return lo


PROFILE_MIN_SLOPE = assert_profile_monotonic()


def monotone_slopes(ts: list[float], ys: list[float]) -> list[float]:
    """Fritsch-Carlson tangents for strictly increasing ts."""
    n = len(ts)
    if n == 2:
        d = (ys[1] - ys[0]) / (ts[1] - ts[0])
        return [d, d]
    h = [ts[k + 1] - ts[k] for k in range(n - 1)]
    delta = [(ys[k + 1] - ys[k]) / h[k] for k in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = delta[0], delta[-1]
    for k in range(1, n - 1):
        m[k] = 0.0 if delta[k - 1] * delta[k] <= 0 else (delta[k - 1] + delta[k]) / 2
    for k in range(n - 1):
        if delta[k] == 0:
            m[k] = m[k + 1] = 0.0
            continue
        a, b = m[k] / delta[k], m[k + 1] / delta[k]
        if a < 0:
            m[k], a = 0.0, 0.0
        if b < 0:
            m[k + 1], b = 0.0, 0.0
        s = a * a + b * b
        if s > 9:
            tau = 3 / math.sqrt(s)
            m[k], m[k + 1] = tau * a * delta[k], tau * b * delta[k]
    return m


def monotone_eval(ts: list[float], ys: list[float], ms: list[float], x: float) -> float:
    if x <= ts[0]:
        return ys[0]
    if x >= ts[-1]:
        return ys[-1]
    k = 0
    while k < len(ts) - 2 and x > ts[k + 1]:
        k += 1
    h = ts[k + 1] - ts[k]
    s = (x - ts[k]) / h
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * ys[k] + (s3 - 2 * s2 + s) * h * ms[k]
            + (-2 * s3 + 3 * s2) * ys[k + 1] + (s3 - s2) * h * ms[k + 1])


class KeyframePath:
    def __init__(self, keyframes: list[dict]):
        self.ts = [k["t"] for k in keyframes]
        self.ys = {c: [k[c] for k in keyframes] for c in CHANNELS}
        self.ms = {c: monotone_slopes(self.ts, self.ys[c]) for c in CHANNELS}

    def at_progress(self, u: float) -> tuple[float, float, float]:
        return tuple(monotone_eval(self.ts, self.ys[c], self.ms[c], u) for c in CHANNELS)

    def at(self, t: float) -> tuple[float, float, float]:
        """Camera at raw shot progress t (time profile applied)."""
        return self.at_progress(hermite_profile(t))

    def frames(self, n: int) -> list[tuple[float, float, float]]:
        return [self.at(f / max(1, n - 1)) for f in range(n)]


def validate_keyframes(keyframes: list[dict]) -> None:
    if not 2 <= len(keyframes) <= 5:
        raise ValueError(f"keyframe_path needs 2-5 keyframes, got {len(keyframes)}")
    ts = [k["t"] for k in keyframes]
    if ts[0] != 0 or ts[-1] != 1:
        raise ValueError("keyframe_path: first keyframe t must be 0 and last t must be 1")
    if any(b <= a for a, b in zip(ts, ts[1:])):
        raise ValueError("keyframe_path: keyframe t must be strictly increasing")
    for k in keyframes:
        if k["zoom"] <= 0:
            raise ValueError("keyframe_path: zoom must be positive")


# ---------------------------------------------------------------------------------------------
# window geometry
# ---------------------------------------------------------------------------------------------

def base_window(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[float, float]:
    """Window at zoom 1 as fractions of the source: the largest output-aspect rect inside it."""
    a_src, a_out = src_w / src_h, out_w / out_h
    return (a_out / a_src, 1.0) if a_src >= a_out else (1.0, a_src / a_out)


def margin_out_px(cx: float, cy: float, zoom: float, src_w: int, src_h: int, out_w: int, out_h: int) -> float:
    """Smallest distance from the window to the image edge, in OUTPUT pixels (negative = the
    window leaves the image)."""
    bw, bh = base_window(src_w, src_h, out_w, out_h)
    hw, hh = bw / (2 * zoom), bh / (2 * zoom)
    m_x = min(cx - hw, 1 - cx - hw) * src_w   # source px
    m_y = min(cy - hh, 1 - cy - hh) * src_h
    out_per_src = out_w / (bw * src_w / zoom)
    return min(m_x, m_y) * out_per_src


def source_px_per_output_px(zoom: float, src_w: int, src_h: int, out_w: int, out_h: int) -> float:
    bw, _ = base_window(src_w, src_h, out_w, out_h)
    return bw * src_w / zoom / out_w
