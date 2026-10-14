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


# ---------------------------------------------------------------------------------------------
# projection "rotate" (addendum 1.3.1, optional, keyframe_path only)
#
# "crop" (the default) slides and scales a window over the photo. "rotate" treats the same
# (cx, cy, zoom) as a camera that turns in place: it points at the photo pixel (cx, cy) with a
# field of view narrowed by zoom, and the frame is the photo reprojected onto that camera. A pure
# rotation about the lens is exactly a homography of a single photo, so it needs no depth and
# keeps straight lines straight; moving to look higher or lower is a real tilt (verticals lean as
# they do through a tilted lens), left or right a real pan, with no roll. At the photo centre it
# equals "crop". The photo's horizontal field of view is not known; hfov_deg assumes one.
# ---------------------------------------------------------------------------------------------
PROJECTIONS = ("crop", "rotate")
DEFAULT_HFOV_DEG = 75.0


def rotate_homography(cx: float, cy: float, zoom: float, src_w: int, src_h: int, out_w: int, out_h: int,
                      hfov_deg: float = DEFAULT_HFOV_DEG) -> list[list[float]]:
    """Row-major 3x3 homography mapping OUTPUT px to SOURCE px."""
    f = (src_w / 2) / math.tan(math.radians(hfov_deg) / 2)
    bw, _ = base_window(src_w, src_h, out_w, out_h)
    fv = f * zoom * out_w / (bw * src_w)          # virtual focal length, output px
    x0, y0 = cx * src_w - src_w / 2, cy * src_h - src_h / 2
    yaw = math.atan2(x0, f)
    pitch = math.atan2(y0, math.hypot(x0, f))
    cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    # R = Ry(yaw) @ Rx(pitch): virtual-camera ray -> source-camera ray (camera x stays level)
    r = [[cyw, -syw * sp, syw * cp],
         [0.0, cp, sp],
         [-syw, -cyw * sp, cyw * cp]]
    # M = Ks @ R @ inv(Kv)
    kv_inv = [[1 / fv, 0.0, -out_w / 2 / fv], [0.0, 1 / fv, -out_h / 2 / fv], [0.0, 0.0, 1.0]]
    ks = [[f, 0.0, src_w / 2], [0.0, f, src_h / 2], [0.0, 0.0, 1.0]]
    return _matmul(ks, _matmul(r, kv_inv))


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def apply_homography(m, x: float, y: float) -> tuple[float, float]:
    w = m[2][0] * x + m[2][1] * y + m[2][2]
    return (m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w


def frame_margin_out_px(cx, cy, zoom, src_w, src_h, out_w, out_h, projection="crop",
                        hfov_deg=DEFAULT_HFOV_DEG) -> float:
    """margin_out_px for either projection. For "rotate" the frame is a quadrilateral on the photo
    (straight edges), so it is inside iff its four corners are."""
    if projection != "rotate":
        return margin_out_px(cx, cy, zoom, src_w, src_h, out_w, out_h)
    m = rotate_homography(cx, cy, zoom, src_w, src_h, out_w, out_h, hfov_deg)
    worst = min(min(u, src_w - u, v, src_h - v)
                for u, v in (apply_homography(m, x, y) for x, y in ((0, 0), (out_w, 0), (0, out_h), (out_w, out_h))))
    bw, _ = base_window(src_w, src_h, out_w, out_h)
    return worst * out_w / (bw * src_w / zoom)


def frame_spp(cx, cy, zoom, src_w, src_h, out_w, out_h, projection="crop", hfov_deg=DEFAULT_HFOV_DEG) -> float:
    """Smallest source px per output px over the frame. For "rotate", the local linear scale
    (sqrt of the Jacobian determinant) at the centre and the four corners."""
    if projection != "rotate":
        return source_px_per_output_px(zoom, src_w, src_h, out_w, out_h)
    m = rotate_homography(cx, cy, zoom, src_w, src_h, out_w, out_h, hfov_deg)
    out = []
    for x, y in ((out_w / 2, out_h / 2), (0, 0), (out_w, 0), (0, out_h), (out_w, out_h)):
        u0, v0 = apply_homography(m, x, y)
        ux, vx = apply_homography(m, x + 1, y)
        uy, vy = apply_homography(m, x, y + 1)
        out.append(math.sqrt(abs((ux - u0) * (vy - v0) - (uy - u0) * (vx - v0))))
    return min(out)


# ---------------------------------------------------------------------------------------------
# camera type "compound" (addendum 1.3.2). Mirrors compoundPathAt in remotion/src/math.ts.
#
#   camera: {type: "compound", move, start: {cx, cy, zoom},
#            segments: [{t0, t1, cx, cy, zoom}, ...],   1-4, t0 non-decreasing, first t0 = 0,
#            timing: {s0, s1, turn?}}                   last t1 = 1, 0 <= s0, s1 <= 3
#
# Shape: each segment moves the camera from the previous segment's target to its own over its
# own window [t0, t1] of the path parameter u (smootherstep weight; cx, cy linear, zoom in log
# space). Overlapping windows blend one phase into the next, so corners are rounded and the
# camera never comes to rest between phases. The end state is the last segment's target.
# Timing: the path is re-timed by visual distance (RMS displacement of the frame's corners, in
# output px), and the fraction of distance covered at raw shot progress t is a cubic Hermite with
# start slope s0 and end slope s1 (in units of the shot's mean speed; monotone for s <= 3). The
# speed on screen is therefore set by the plan: s0 * mean at the first frame, s1 * mean at the
# cut, and at most max(hermite') * mean in between.
# Turning (1.3.2a, optional timing.turn, px per radian, default 0): a change of direction costs
# time as well as distance. Each arc sample costs its distance plus turn x the angle the motion
# direction turns there (lateral px, vertical px, zoom px; smoothed over +-TURN_SMOOTH samples), so
# the camera slows into a corner and speeds up out of it, as an operator would, instead of
# swinging round it at full speed. "mean" above is then the mean cost per frame.
# ---------------------------------------------------------------------------------------------
ARC_SAMPLES = 512
TURN_SMOOTH = 40


def smootherstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (6 * x - 15) + 10)


def hermite_slope_max(s0: float, s1: float, steps: int = 2000) -> float:
    return max(hermite_profile_slope(i / steps, s0, s1) for i in range(steps + 1))


def corner_rms_px(a, b, src_w, src_h, out_w, out_h) -> float:
    """Visual distance between two crop states: RMS displacement of the frame's four corners in
    output px (zoom is measured at the mean of the two states' scales)."""
    bw, bh = base_window(src_w, src_h, out_w, out_h)
    acc = 0.0
    wa, ha, wb, hb = bw / a[2], bh / a[2], bw / b[2], bh / b[2]
    scale = 0.5 * (out_w / (wa * src_w) + out_w / (wb * src_w))
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            dx = ((b[0] + sx * wb) - (a[0] + sx * wa)) * src_w
            dy = ((b[1] + sy * hb) - (a[1] + sy * ha)) * src_h
            acc += dx * dx + dy * dy
    return math.sqrt(acc / 4) * scale


def motion_px(a, b, src_w, src_h, out_w, out_h) -> tuple[float, float, float]:
    """Motion from crop state a to b as (lateral, vertical, zoom) output px; zoom is the radial
    motion of the frame corners."""
    bw, _ = base_window(src_w, src_h, out_w, out_h)
    k = out_w / (bw / a[2] * src_w)
    return ((b[0] - a[0]) * src_w * k, (b[1] - a[1]) * src_h * k,
            math.log(b[2] / a[2]) * math.hypot(out_w, out_h) / 2)


def vec_angle(a, b) -> float:
    na = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    nb = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
    if na * nb <= 1e-12:
        return 0.0
    c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)
    return math.acos(max(-1.0, min(1.0, c)))


class CompoundPath:
    def __init__(self, start: dict, segments: list[dict], timing: dict, src_w: int, src_h: int,
                 out_w: int, out_h: int):
        self.c0 = (start["cx"], start["cy"], math.log(start["zoom"]))
        prev = self.c0
        self.segs = []
        for s in segments:
            tgt = (s["cx"], s["cy"], math.log(s["zoom"]))
            self.segs.append((s["t0"], s["t1"], tuple(t - p for t, p in zip(tgt, prev))))
            prev = tgt
        self.s0, self.s1 = timing["s0"], timing["s1"]
        self.turn = timing.get("turn") or 0.0
        self.dims = (src_w, src_h, out_w, out_h)
        states = [self.at_u(j / ARC_SAMPLES) for j in range(ARC_SAMPLES + 1)]
        dist = [corner_rms_px(states[j - 1], states[j], *self.dims) for j in range(1, ARC_SAMPLES + 1)]
        self.distance = sum(dist)
        cost = dist
        if self.turn > 0:
            vec = [motion_px(states[j - 1], states[j], *self.dims) for j in range(1, ARC_SAMPLES + 1)]
            theta = [0.0] * ARC_SAMPLES          # theta[j]: turn between increment j and j + 1
            for j in range(ARC_SAMPLES - 1):
                theta[j] = vec_angle(vec[j], vec[j + 1])
            r = TURN_SMOOTH
            cost = [dist[j] + self.turn * sum(theta[max(0, j - r):j + r + 1]) / (2 * r + 1)
                    for j in range(ARC_SAMPLES)]
        self.table = [0.0]
        for c in cost:
            self.table.append(self.table[-1] + c)
        self.cost = self.table[-1]

    def at_u(self, u: float) -> tuple[float, float, float]:
        cx, cy, lz = self.c0
        for t0, t1, (dx, dy, dz) in self.segs:
            w = smootherstep((u - t0) / (t1 - t0))
            cx += dx * w
            cy += dy * w
            lz += dz * w
        return cx, cy, math.exp(lz)

    def u_at(self, t: float) -> float:
        if self.cost <= 0:
            return 0.0
        target = hermite_profile(t, self.s0, self.s1) * self.cost
        lo, hi = 0, ARC_SAMPLES
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.table[mid] <= target:
                lo = mid
            else:
                hi = mid
        seg = self.table[hi] - self.table[lo]
        frac = 0.0 if seg <= 0 else (target - self.table[lo]) / seg
        return min(1.0, (lo + frac) / ARC_SAMPLES)

    def at(self, t: float) -> tuple[float, float, float]:
        return self.at_u(self.u_at(t))

    def frames(self, n: int) -> list[tuple[float, float, float]]:
        return [self.at(f / max(1, n - 1)) for f in range(n)]


def validate_compound(start: dict | None, segments: list[dict], timing: dict | None) -> None:
    if start is None or timing is None:
        raise ValueError("compound camera needs start, segments and timing")
    if not 1 <= len(segments) <= 4:
        raise ValueError(f"compound camera needs 1-4 segments, got {len(segments)}")
    if segments[0]["t0"] != 0 or segments[-1]["t1"] != 1:
        raise ValueError("compound camera: first segment t0 must be 0 and last t1 must be 1")
    for a, b in zip(segments, segments[1:]):
        if b["t0"] < a["t0"] or b["t1"] < a["t1"]:
            # with both non-decreasing the weights are ordered, so every frame is a convex blend
            # of the targets (each inside the image, crop windows form a convex set)
            raise ValueError("compound camera: segment t0 and t1 must be non-decreasing")
    for s in segments:
        if not 0 <= s["t0"] < s["t1"] <= 1:
            raise ValueError("compound camera: each segment needs 0 <= t0 < t1 <= 1")
        if s["zoom"] <= 0:
            raise ValueError("compound camera: zoom must be positive")
    if start["zoom"] <= 0:
        raise ValueError("compound camera: zoom must be positive")
    for k in ("s0", "s1"):
        if not 0 <= timing[k] <= 3:
            raise ValueError(f"compound camera: timing {k} must be 0-3 (monotone)")
    if not 0 <= (timing.get("turn") or 0) <= 2000:
        raise ValueError("compound camera: timing turn must be 0-2000 px per radian")
