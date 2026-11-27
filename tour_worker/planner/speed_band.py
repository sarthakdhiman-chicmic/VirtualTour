"""One speed band per tour (client review 2026-11-20, item 4), for photo shots with continuous cuts
(camera.path_s; render/camera_path.ramp_profile).

The pacing preset sets one tour-wide target on-screen speed (BAND, % of the frame width per second,
the unit of render/continuity.py). Each shot's move is scaled about its END keyframe - the end
composition (the whole room, the subject of a push) stays exactly where the planner put it; the
travel to it grows or shrinks - so that it runs at the target where the photo allows:
  - faster than the target: shrunk to it;
  - slower: grown to it, as far as every frame stays inside the photo (MIN_MARGIN_OUT_PX) and is
    never enlarged past 1:1 (NO_UPSCALE) - or, for each, no worse than the planned move already is;
then across every cut the incoming speed is brought within SPEED_RATIO of the outgoing one by
slowing the faster side (a shot that cannot reach the band sets its neighbours' pace). The
tour's first and last shots ramp in and out of the band at the tour's ends (their easing);
their cut ends obey the same rule.

Speeds are measured where the cut can fall: a shot's entry over its first MEASURE_S of path, its
exit around its nominal cut (its planned length, the sync's handle being the path's tail).
"""
from __future__ import annotations

import math

from tour_worker.render.camera_path import KeyframePath, base_window, frame_margin_out_px, frame_spp

BAND = {"slow": 1.9, "normal": 2.6, "fast": 3.4}    # target speed, % frame width / s
SPEED_RATIO = 1.25          # the limit (render/continuity.py) ...
PLAN_RATIO = 1.2            # ... planned to with room for frame rounding
MEASURE_S = 0.2
MIN_MARGIN_OUT_PX = 1.0
MAX_GROW = 6.0             # a move is never made more than 6x its planned travel (and only inside the photo)
NO_UPSCALE = 1.0           # source px per output px a grown move keeps (unless the plan already had less)
CHECK_SAMPLES = 48
PASSES = 3


def target_speed(pacing: str | None) -> float:
    return BAND.get(pacing or "normal", BAND["normal"])


# 3D tours (user 2026-11-27): the photo band's 2.6 %fw/s is a 1.8 deg/s pan and a 0.16 m/s walk - under
# the planner's own "seen to move" floor (MIN_TURN_PS 2.5 deg/s, MIN_TRAVEL_PS 0.12 m/s), so pans read
# as stills and every 3D move was planned 3-4x faster than it played. 3.6 %fw/s normal: a 2.5 deg/s
# pan, a 0.22 m/s walk; slow / fast in the photo band's proportions
BAND_3D = {"slow": 2.6, "normal": 3.6, "fast": 4.7}


def target_speed_3d(pacing: str | None) -> float:
    return BAND_3D.get(pacing or "normal", BAND_3D["normal"])


def _speed(path: KeyframePath, u0: float, u1: float, dt: float, src_w: int, src_h: int, out_w: int, out_h: int,
           easing: str | None = "linear") -> float:
    """On-screen speed (%fw/s) between path times u0 and u1 (through the shot's easing), dt s apart."""
    if dt <= 0:
        return 0.0
    from tour_worker.render.camera_path import ramp_profile
    bw, _ = base_window(src_w, src_h, out_w, out_h)
    cx0, cy0, z0 = path.at_progress(ramp_profile(easing, u0))
    cx1, cy1, z1 = path.at_progress(ramp_profile(easing, u1))
    win = bw / math.sqrt(z0 * z1)
    x = (cx1 - cx0) / win / dt * 100.0
    y = (cy1 - cy0) * src_h / (win * src_w) / dt * 100.0
    f = math.log(z1 / z0) / dt * 50.0
    return math.sqrt(x * x + y * y + f * f)


def _ends(cam, src_w, src_h, out_w, out_h, cut_s: float, fps: float = 30.0) -> tuple[float, float]:
    """(entry, exit) speed, on the frames render/continuity.py measures: the first MEASURE_S of the
    shot and the MEASURE_S up to its last frame."""
    path = KeyframePath([k.model_dump() for k in cam.keyframes])
    ps = cam.path_s
    n = max(2, int(round(cut_s * fps)))
    m = max(1, min(int(round(MEASURE_S * fps)), (n - 1) // 2))
    u = lambda f: min(1.0, f / fps / ps)          # noqa: E731
    e = getattr(cam.easing, "value", cam.easing)
    dt = m / fps
    return (_speed(path, u(0), u(m), dt, src_w, src_h, out_w, out_h, e),
            _speed(path, u(n - 1 - m), u(n - 1), dt, src_w, src_h, out_w, out_h, e))


def _scaled(keyframes: list, k: float) -> list:
    """Keyframes scaled about the end one by k: centre linearly, zoom in log space (the camera's
    on-screen speed scales by k for both)."""
    end = keyframes[-1]
    out = []
    for kf in keyframes:
        z = math.exp(math.log(end.zoom) + k * (math.log(kf.zoom) - math.log(end.zoom)))
        upd = {"cx": end.cx + k * (kf.cx - end.cx), "cy": end.cy + k * (kf.cy - end.cy), "zoom": z}
        if getattr(kf, "roll_deg", None) is not None:
            upd["roll_deg"] = (end.roll_deg or 0.0) + k * ((kf.roll_deg or 0.0) - (end.roll_deg or 0.0))
        out.append(kf.model_copy(update=upd))
    return out


def _kw(cam) -> dict:
    hfov = cam.hfov_deg if isinstance(cam.hfov_deg, (int, float)) else None
    return {"projection": cam.projection or "crop", **({"hfov_deg": hfov} if hfov else {})}


def _worst(cam, keyframes, src_w, src_h, out_w, out_h) -> tuple[float, float]:
    """(smallest margin inside the photo, output px; smallest source density) over the move."""
    path = KeyframePath([k.model_dump() for k in keyframes])
    kw = _kw(cam)
    margin, spp = math.inf, math.inf
    for i in range(CHECK_SAMPLES + 1):
        cx, cy, z = path.at_progress(i / CHECK_SAMPLES)
        if z <= 0:
            return -math.inf, 0.0
        margin = min(margin, frame_margin_out_px(cx, cy, z, src_w, src_h, out_w, out_h, **kw))
        spp = min(spp, frame_spp(cx, cy, z, src_w, src_h, out_w, out_h, **kw))
    return margin, spp


def _limits(cam, src_w, src_h, out_w, out_h) -> tuple[float, float]:
    """What a scaled move must keep: inside the photo by MIN_MARGIN_OUT_PX and at NO_UPSCALE density
    - or no worse than the planned move already is (the planner's own margins can be tighter)."""
    margin, spp = _worst(cam, cam.keyframes, src_w, src_h, out_w, out_h)
    return min(MIN_MARGIN_OUT_PX, margin) - 1e-6, min(NO_UPSCALE, spp) - 1e-9


def _fits(cam, keyframes, src_w, src_h, out_w, out_h, limits: tuple[float, float]) -> bool:
    margin, spp = _worst(cam, keyframes, src_w, src_h, out_w, out_h)
    return margin >= limits[0] and spp >= limits[1]


def _max_grow(cam, src_w, src_h, out_w, out_h, limits) -> float:
    """The largest k (1..MAX_GROW) whose scaled move still fits the photo (bisection)."""
    if not _fits(cam, _scaled(cam.keyframes, 1.0 + 1e-3), src_w, src_h, out_w, out_h, limits):
        return 1.0
    lo, hi = 1.0, MAX_GROW
    if _fits(cam, _scaled(cam.keyframes, hi), src_w, src_h, out_w, out_h, limits):
        return hi
    for _ in range(12):
        mid = (lo + hi) / 2
        if _fits(cam, _scaled(cam.keyframes, mid), src_w, src_h, out_w, out_h, limits):
            lo = mid
        else:
            hi = mid
    return lo


def _rep(cam, ends: tuple[float, float]) -> float:
    """A shot's pace: the end(s) a cut falls on - the first shot's exit, the last shot's entry (their
    other end is the tour's ease), both for the rest."""
    e = getattr(cam.easing, "value", cam.easing)
    if e == "ease_in":
        return ends[1]
    if e == "ease_out":
        return ends[0]
    return (ends[0] + ends[1]) / 2


def apply_speed_band(shots, pacing: str | None, out_w: int, out_h: int, fps: float = 30.0) -> list[str]:
    """Scale each continuous-cut photo shot's move into the tour's speed band and match speeds
    across cuts (module docstring). shots: planner Shots in tour order, duration_s where each is
    cut, camera.path_s set. Scaling changes on-screen speed only about in proportion (the window
    width follows the zoom), so it runs PASSES times, measuring afresh. Returns warnings
    (speed_band_short:<shot>:<speed>, speed_band_unscaled:<shot>)."""
    target = target_speed(pacing)
    idx = [i for i, s in enumerate(shots)
           if s.camera.type == "keyframe_path" and getattr(s.camera, "path_s", None) and s.camera.keyframes
           and s.source is not None]
    if not idx:
        return []
    dims = {i: (shots[i].source.width, shots[i].source.height) for i in idx}
    limits = {i: _limits(shots[i].camera, *dims[i], out_w, out_h) for i in idx}   # from the planned moves
    unscaled: set[int] = set()
    caps: dict[int, float] = {}
    for _ in range(PASSES):
        ends = {i: _ends(shots[i].camera, *dims[i], out_w, out_h, shots[i].duration_s, fps) for i in idx}
        k = {}
        for i in idx:
            rep = _rep(shots[i].camera, ends[i])
            if rep < 1e-6 or i in unscaled:    # a still shot stays still
                k[i] = 1.0
                continue
            want = target / rep
            if want > 1.0 and i not in caps:
                caps[i] = _max_grow(shots[i].camera, *dims[i], out_w, out_h, limits[i])
            k[i] = min(want, caps.get(i, 1.0)) if want > 1.0 else want
        # across each cut: the faster side slowed to within PLAN_RATIO of the slower (only ever
        # slower, so it settles); neighbours are consecutive photo shots
        for _ in range(4 * len(idx)):
            changed = False
            for a, b in zip(idx, idx[1:]):
                if b != a + 1:
                    continue
                out_v, in_v = ends[a][1] * k[a], ends[b][0] * k[b]
                if min(out_v, in_v) < 1e-6:
                    continue
                if out_v > PLAN_RATIO * in_v + 1e-9:
                    k[a] *= PLAN_RATIO * in_v / out_v
                    changed = True
                elif in_v > PLAN_RATIO * out_v + 1e-9:
                    k[b] *= PLAN_RATIO * out_v / in_v
                    changed = True
            if not changed:
                break
        moved = False
        for i in idx:
            if abs(k[i] - 1.0) <= 1e-3:
                continue
            cam = shots[i].camera
            scaled = _scaled(cam.keyframes, k[i])
            if _fits(cam, scaled, *dims[i], out_w, out_h, limits[i]):
                cam.keyframes = scaled
                if i in caps:
                    caps[i] = max(1.0, caps[i] / k[i])     # what is left of its room to grow
                moved = True
            else:                               # (a shrunk move always fits in practice; never break one)
                unscaled.add(i)
        if not moved:
            break
    warnings = [f"speed_band_unscaled:{shots[i].shot_id}" for i in sorted(unscaled)]
    for i in idx:
        v = _rep(shots[i].camera, _ends(shots[i].camera, *dims[i], out_w, out_h, shots[i].duration_s, fps))
        if 1e-6 < v < target / SPEED_RATIO - 1e-6:
            warnings.append(f"speed_band_short:{shots[i].shot_id}:{v:.2f}")
    return warnings


# ---- 3D shots (bezier_path) -----------------------------------------------------------------
# A 3D move runs where the room lets it: it can be slowed but not safely sped up (a longer flight
# meets walls the planner never checked). So the band only slows: a shot faster than the target, or
# than PLAN_RATIO x its neighbour across a cut, has the START of its flight trimmed (the curve split
# exactly, de Casteljau) - its end framing kept, the same path_s, so it covers less ground.

def _scene_ends(cam, cut_s: float, fps: float, up) -> tuple[float, float]:
    from tour_worker.render.continuity import _scene_motion
    from tour_worker.scene.camera import BezierCameraPath
    path = BezierCameraPath([s.model_dump() for s in cam.segments], cam.hfov_deg,
                            easing=getattr(cam.easing, "value", cam.easing),
                            speed_profile=getattr(cam, "speed_profile", None))
    n = max(2, int(round(cut_s * fps)))
    m = max(1, min(int(round(MEASURE_S * fps)), (n - 1) // 2))
    a = _scene_motion(path, n, 0, m, fps, up, cam.path_s).speed
    b = _scene_motion(path, n, n - 1 - m, n - 1, fps, up, cam.path_s).speed
    return a, b


def _split(p, u: float):
    """The right part (from u) of a cubic Bezier's control points p (4, 3)."""
    import numpy as np
    p = np.asarray(p, float)
    a, b = p[:-1] * (1 - u) + p[1:] * u, None
    b = a[:-1] * (1 - u) + a[1:] * u
    c = b[0] * (1 - u) + b[1] * u
    return np.array([c, b[1], a[2], p[3]])


def trim_start(cam, keep: float):
    """The camera with only the last `keep` share (0-1) of its flight (by the path's arc length)."""
    return trim_window(cam, 1.0 - keep, 1.0)


def trim_window(cam, a: float, b: float):
    """The camera with only the share a..b (0-1, by the path's arc length) of its flight."""
    import numpy as np
    from tour_worker.scene.camera import BezierCameraPath
    seg_cls = type(cam.segments[0])
    segs = [s.model_dump() for s in cam.segments]
    if b < 1.0 - 1e-6:                              # the end first: a's place is the same share of it
        k, u = BezierCameraPath(segs, cam.hfov_deg)._locate(b)
        left = lambda p: _split(np.asarray(p, float)[::-1], 1.0 - u)[::-1]   # noqa: E731 - the part before u
        segs = segs[:k] + [{"eye": left(segs[k]["eye"]).round(5).tolist(),
                            "target": left(segs[k]["target"]).round(5).tolist()}]
        a = a / b
    if a > 1e-6:
        k, u = BezierCameraPath(segs, cam.hfov_deg)._locate(a)
        segs = [{"eye": _split(segs[k]["eye"], u).round(5).tolist(),
                 "target": _split(segs[k]["target"], u).round(5).tolist()}] + segs[k + 1:]
    return cam.model_copy(update={"segments": [seg_cls(**s) for s in segs]})


DOOR_LEAD = 0.6            # m of a through flight kept before its door (the way through is the shot)


def apply_speed_band_3d(shots, pacing: str | None, fps: float = 30.0, up=None, keep_from=None) -> list[str]:
    """Slow the 3D shots into the band and match them within PLAN_RATIO across every cut (only ever
    slower - module note above). shots: in tour order, duration_s where each is cut, path_s set.
    A move is slowed by playing a share of its path (its start dropped first) in the same time - at
    least MIN_KEEP of it a pass. keep_from: {shot index: (path share, least share)} a through flight
    keeps from (DOOR_LEAD before its room's door) to its end, at least - the walk in and the turn
    onto the room always play (user 2026-11-27, green house: the walk
    through the study door was trimmed away and the shot stood at the door frame)."""
    keep_from = keep_from or {}
    target = target_speed_3d(pacing)
    up = tuple(up) if up else (0.0, 0.0, 1.0)
    idx = [i for i, s in enumerate(shots) if s.camera.type == "bezier_path" and getattr(s.camera, "path_s", None)
           and s.camera.segments]
    if not idx:
        return []
    cam0 = {i: shots[i].camera for i in idx}
    kept = dict.fromkeys(idx, 1.0)
    for _ in range(PASSES):
        ends = {i: _scene_ends(shots[i].camera, shots[i].duration_s, fps, up) for i in idx}
        k = {}
        for i in idx:
            rep = _rep(shots[i].camera, ends[i])
            k[i] = min(1.0, target / rep) if rep > 1e-6 else 1.0
        for _ in range(4 * len(idx)):
            changed = False
            for a, b in zip(idx, idx[1:]):
                if b != a + 1:
                    continue
                out_v, in_v = ends[a][1] * k[a], ends[b][0] * k[b]
                if min(out_v, in_v) < 1e-6:
                    continue
                if out_v > PLAN_RATIO * in_v + 1e-9:
                    k[a] *= PLAN_RATIO * in_v / out_v
                    changed = True
                elif in_v > PLAN_RATIO * out_v + 1e-9:
                    k[b] *= PLAN_RATIO * out_v / in_v
                    changed = True
            if not changed:
                break
        moved = False
        for i in idx:
            a0, least = keep_from.get(i, (1.0, 0.0))
            share = max(kept[i] * max(MIN_KEEP, k[i]), least)
            if share < kept[i] - 1e-3:
                kept[i] = share
                a = min(1.0 - share, a0)
                shots[i].camera = trim_window(cam0[i], a, a + share)
                moved = True
        if not moved:
            break
    return []


MIN_KEEP = 0.5             # a 3D move keeps at least half its flight a pass (a walk-in trimmed to a fifth only turns)
