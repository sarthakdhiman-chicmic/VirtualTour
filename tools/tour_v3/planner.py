"""Tour v4 planner (worker harness; backend port later): compound camera moves (addendum 1.3.2).

Builds on the v3 tilt_reveal checkpoint (tag checkpoint/v3-tilt-reveal) and the review of it:
1. speed: every shot is re-timed by visual distance, so the plan sets the speed on screen. Each
   shot enters and leaves at V_CUT px/frame and never exceeds V_MAX px/frame; durations follow
   each move's distance, and a move that would still be too fast is shrunk toward its end state.
2. cuts: a shot's opening motion follows the previous shot's closing motion. A zoom that would
   reverse across the cut (out -> in, in -> out) is ruled out, and pan and tilt directions that
   carry through the cut score higher.
3. direction: every move can be mirrored left/right, and those with a tilt flipped up/down.
4. vocabulary: nine compound moves (MOVES), each made of overlapping phases (push, pan, tilt,
   pull), instead of single zooms and pans.

The sequence is chosen with a beam search over (move, mirror, flip) per shot. It scores fit for
the room (wide, small, bathroom, exterior, final), how interesting the opening and closing frames
are (saliency), continuity at each cut, and variety. There is no room context; aim comes only
from spectral-residual saliency. Every decision is written to the notes file.

  python -m tools.tour_v3.planner --baseline fixtures/downing/baseline_plan.json \\
      --saliency fixtures/downing/saliency.json --images-dir <photos> --out-dir out/tour_v4
"""
import argparse
import copy
import json
import math
import os
import statistics

import cv2
import numpy as np

from modules.camera_path import (CompoundPath, base_window, frame_margin_out_px, frame_spp, hermite_slope_max,
                                 motion_px, vec_angle)
from modules.plan_io import compute_plan_hash, parse_plan
from modules.timeline import compute_timeline
import tools.tour_v2.planner as _v2
from tools.tour_v2.planner import Geometry, is_bathroom, is_exterior, parse_plan_loose
from tools.tour_v2.saliency import load_gray, spectral_residual

PLANNER_VERSION = "tour_v5-0.1"
OUT_W, OUT_H, FPS = 1920, 1080, 30
MIN_SPP = 0.75
MIN_MARGIN_OUT_PX = 1.0

V_CUT = 1.6          # px/frame at the first and last frame of every shot
V_MAX = 5.0          # px/frame, never exceeded (v3 peaked at 7.8)
M_TARGET = 2.6       # mean px/frame a move's duration is sized for
SHOT_RANGE_S = (3.5, 8.0)
FINAL_MIN_S = 5.0
TOTAL_TOLERANCE_S = 5.0
SHRINK_STEP = 0.05

ZT, ZM, ZW = 1.30, 1.12, 1.012   # tight, medium, wide (full coverage keeps a 2 px margin)
BEAM = 400
MAX_USES = 3                      # per move per tour
TURN_COST = 60.0                  # timing.turn, px of time per radian of turn (addendum 1.3.2a)
# v5 cut rules (review of v4): the next room starts at about the framing the last one ended on,
# and moves the opposite way, so a pull-back is answered by a push-in, a pan left by a pan right
SCALE_JUMP_FREE = 0.03            # |ln(zoom ratio)| across a cut allowed without penalty
SCALE_JUMP_WEIGHT = 30.0
REVEAL_FAMILY = {"tilt_reveal", "tilt_reveal_rotate", "tilt_sweep", "rise_reveal"}


# ---------------------------------------------------------------------------------------------
# moves. Each returns (start, segments). d = +1 travels right, -1 left; v = +1 travels up
# (starts low), -1 travels down (starts high). E(a, b, z) is the furthest fitting window toward
# horizontal side a and vertical side b (-1 left/top .. +1 right/bottom) at zoom z.
# ---------------------------------------------------------------------------------------------

def _state(cx, cy, z):
    return {"cx": round(cx, 6), "cy": round(cy, 6), "zoom": round(z, 6)}


def _seg(t0, t1, st):
    return {"t0": t0, "t1": t1, **st}


def E(g, a, b, z):
    return _state(*g.fit(0.5 + 0.5 * a, 0.5 + 0.5 * b, z), z)


def F(g, cx, cy, z):
    """A window at (cx, cy, z) pulled inside the photo."""
    return _state(*g.fit(cx, cy, z), z)


def edge_held(g, st, d, z):
    """Window at zoom z keeping st's edge on side d where it is (pulling back from that corner)."""
    edge = st["cx"] + d * g.width(st["zoom"]) / 2
    return _state(*g.fit(edge - d * g.width(z) / 2, 0.5, z), z)


def peak_b(sal):
    return max(-1.0, min(1.0, (sal["peak"]["y"] - 0.5) * 2))


def peak_a(sal):
    return max(-1.0, min(1.0, (sal["peak"]["x"] - 0.5) * 2))


# ---- ported effects: the v2 and v3 keyframe moves, same shapes, re-timed as compound -------
OVERLAP = 0.25  # each phase starts this much before the previous keyframe is reached
# (0.06 in v4 left near-sharp corners that the camera swung round at full speed: the jerky pull
# back; 0.25 rounds them. reveal_sweep has three corners close together and uses 0.18)


def from_keyframes(kfs, v, overlap=None):
    """keyframe_path keyframes -> (start, segments). Phase k runs from keyframe k-1 to k,
    widened by the overlap each side so the camera rounds the keyframe instead of stopping on it.
    v = -1 mirrors the move vertically (a tilt down instead of up)."""
    flip = (lambda st: {**st, "cy": round(1 - st["cy"], 6)}) if v < 0 else (lambda st: st)  # noqa: E731
    pts = [flip(_state(k["cx"], k["cy"], k["zoom"])) for k in kfs]
    ts = [k["t"] for k in kfs]
    ov = OVERLAP if overlap is None else overlap
    segs = [_seg(round(max(0.0, ts[k - 1] - ov), 4) if k > 1 else 0,
                 round(min(1.0, ts[k] + ov), 4) if k < len(kfs) - 1 else 1, pts[k])
            for k in range(1, len(kfs))]
    return pts[0], segs


def mv_reveal_sweep(g, sal, d, v):
    """v2 reveal_sweep: near-full view, diagonal push toward one side, scroll across at the same
    height, pull back from that corner."""
    return from_keyframes(_v2.move_reveal_sweep(g, sal, d), v, 0.18)


def mv_tilt_reveal(g, sal, d, v):
    """v3 tilt_reveal (the liked checkpoint): tight on a low corner, pan across while tilting up
    to eye level, pull back to the whole room from the far corner."""
    return from_keyframes(_v2.move_tilt_reveal(g, sal, d), v)


def mv_tilt_reveal_rotate(g, sal, d, v):
    """v3 tilt_reveal through the "rotate" projection: the same path as a camera turning in place
    (true pan and tilt, verticals lean). Its reach is ~40% of the crop version's, so it reads as
    a subtle look-around; it suits medium rooms."""
    return from_keyframes(_v2.move_tilt_reveal(g, sal, d), v)


def mv_tilt_sweep(g, sal, d, v):
    """Tight on a low (or high) corner, pan across while tilting to eye level, pull back to the
    whole room from the far corner (a rounder, wider-blended tilt_reveal)."""
    start = E(g, -d, v, ZT)
    mid = E(g, d, 0.25 * peak_b(sal), ZT - 0.06)
    return start, [_seg(0, 0.75, mid), _seg(0.4, 1, edge_held(g, mid, d, ZW))]


def mv_swoop(g, sal, d, v):
    """Push diagonally into one corner, scroll across to the other, pull straight back from it."""
    b = 0.7 * v
    start = E(g, 0.12 * d, 0, ZM)
    a = E(g, -d, b, ZT)
    bb = E(g, d, b, ZT)
    return start, [_seg(0, 0.46, a), _seg(0.2, 0.8, bb), _seg(0.55, 1, edge_held(g, bb, d, ZW))]


def mv_traverse(g, sal, d, v):
    """Corner to opposite corner at a tight zoom, easing out slightly; x leads y (a gentle S)."""
    start = E(g, -d, v, ZT)
    end = E(g, d, -0.8 * v, ZT - 0.1)
    return start, [_seg(0, 0.8, F(g, end["cx"], start["cy"], ZT - 0.05)), _seg(0.25, 1, end)]


def mv_push_pan(g, sal, d, v):
    """Push in while panning, curving onto the salient height at the far side."""
    start = E(g, -0.5 * d, 0, 1.05)
    end = E(g, 0.8 * d, 0.6 * peak_b(sal), ZT - 0.04)
    return start, [_seg(0, 0.75, F(g, end["cx"], start["cy"], ZM + 0.04)), _seg(0.3, 1, end)]


def mv_pan_reveal(g, sal, d, v):
    """Tight on one side at the salient height, drift across while pulling back to the room."""
    start = E(g, -d, 0.7 * peak_b(sal), ZT)
    mid = E(g, 0.6 * d, 0.4 * peak_b(sal), 1.16)
    return start, [_seg(0, 0.66, mid), _seg(0.28, 1, _state(*g.fit(0.5, 0.5, ZW), ZW))]


def mv_slide_push(g, sal, d, v):
    """Open on a plain lateral slide, then curve into a push on the salient height. Its entry is
    neutral, so it follows a reveal without a zoom reversal and hands a push to the next shot."""
    pb = peak_b(sal)
    start = E(g, -d, 0.3 * pb, 1.1)
    mid = F(g, start["cx"] + d * 0.35 * (1 - g.width(1.1)), start["cy"], 1.11)
    return start, [_seg(0, 0.55, mid), _seg(0.3, 1, E(g, 0.7 * d, 0.6 * pb, ZT - 0.04))]


def mv_crane(g, sal, d, v):
    """Exterior: rise from low, pushing in as it climbs, with a slight drift."""
    start = E(g, -0.3 * d, 1, 1.04)
    mid = E(g, 0.1 * d, -0.4, 1.08)
    return start, [_seg(0, 0.8, mid), _seg(0.35, 1, E(g, 0.3 * d, -0.7, 1.16))]


def mv_settle_push(g, sal, d, v):
    """Small rooms: one gentle push toward the subject with a slight drift."""
    pa, pb = peak_a(sal), peak_b(sal)
    start = E(g, 0.3 * pa - 0.2 * d, 0.3 * pb, 1.03)
    return start, [_seg(0, 1, E(g, 0.5 * pa + 0.2 * d, 0.5 * pb, 1.14))]


def mv_pull_back(g, sal, d, v):
    """Start on the subject, pull back to the whole room with a lateral drift."""
    pa, pb = peak_a(sal), peak_b(sal)
    start = E(g, 0.8 * pa, 0.8 * pb, ZT - 0.05)
    mid = E(g, 0.3 * d, 0.3 * pb, 1.1)
    return start, [_seg(0, 0.8, mid), _seg(0.3, 1, _state(*g.fit(0.5, 0.5, ZW), ZW))]


def mv_rise_reveal(g, sal, d, v):
    """Tight on the floor-level foreground, tilt up while pulling back to the whole room."""
    start = E(g, 0.4 * d, v, ZT)
    mid = E(g, -0.3 * d, -0.3 * v, 1.14)
    return start, [_seg(0, 0.74, mid), _seg(0.3, 1, _state(*g.fit(0.5, 0.5, ZW), ZW))]


MOVES = {
    "reveal_sweep": (mv_reveal_sweep, True), "tilt_reveal_rotate": (mv_tilt_reveal_rotate, True),
    "tilt_reveal": (mv_tilt_reveal, True), "tilt_sweep": (mv_tilt_sweep, True), "swoop": (mv_swoop, True), "traverse": (mv_traverse, True),
    "push_pan": (mv_push_pan, False), "pan_reveal": (mv_pan_reveal, False), "crane": (mv_crane, False),
    "settle_push": (mv_settle_push, False), "slide_push": (mv_slide_push, False), "pull_back": (mv_pull_back, False),
    "rise_reveal": (mv_rise_reveal, True),
}  # name -> (builder, uses v)

# Fit per room category. Coverage is the photo's saliency spread (fraction above its mean):
# "wide" is above the tour's median, "medium" between the 25th percentile and the median,
# "tight" below. The room score is the shot's role: heroes add HERO_BONUS to the feature moves.
SUIT = {  # category -> {move: score}; a move missing from a category is not allowed there
    "exterior": {"crane": 3, "push_pan": 1, "swoop": 1},
    "final": {"pull_back": 3, "pan_reveal": 3, "tilt_reveal_rotate": 3, "tilt_reveal": 2, "rise_reveal": 2},
    "bath": {"settle_push": 2, "pull_back": 1.5, "tilt_reveal_rotate": 1.5, "push_pan": 1},
    "wide": {"tilt_reveal_rotate": 5, "tilt_reveal": 2, "reveal_sweep": 3, "swoop": 2.5, "tilt_sweep": 2.5, "traverse": 2,
             "rise_reveal": 2, "slide_push": 2, "pan_reveal": 1.5, "push_pan": 1},
    "medium": {"tilt_reveal_rotate": 3.5, "push_pan": 2, "pan_reveal": 2, "traverse": 2, "rise_reveal": 1.5,
               "slide_push": 2.5, "tilt_reveal": 1.5, "swoop": 1, "settle_push": 1},
    "tight": {"push_pan": 2, "settle_push": 2, "slide_push": 2, "tilt_reveal_rotate": 2, "pan_reveal": 1.5,
              "pull_back": 1.5, "rise_reveal": 1},
}
FEATURE_MOVES = {"tilt_reveal", "tilt_reveal_rotate", "reveal_sweep", "swoop", "tilt_sweep", "traverse",
                 "rise_reveal"}
HERO_BONUS = 0.75
ROTATE_MOVES = {"tilt_reveal_rotate"}  # rendered with projection "rotate" (addendum 1.3.1)


# ---------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------

def make_path(g, start, segments, timing=None):
    return CompoundPath(start, segments, timing or {"s0": 1.0, "s1": 1.0}, g.src_w, g.src_h, OUT_W, OUT_H)


def motion_vector(path, at_end):
    """Direction of travel at the shot's start or end, in output px: (lateral, vertical, zoom).
    Zoom is measured as the radial motion of the frame corners (positive = zooming in)."""
    u0, u1 = (0.97, 1.0) if at_end else (0.0, 0.03)
    a, b = path.at_u(u0), path.at_u(u1)
    bw, bh = base_window(path.dims[0], path.dims[1], OUT_W, OUT_H)
    scale = OUT_W / (bw / a[2] * path.dims[0])
    lat = (b[0] - a[0]) * path.dims[0] * scale
    ver = -(b[1] - a[1]) * path.dims[1] * scale            # + = moving up
    zoom = math.log(b[2] / a[2]) * math.hypot(OUT_W, OUT_H) / 2
    return lat, ver, zoom


def classify(vec):
    lat, ver, zoom = vec
    n = math.sqrt(lat * lat + ver * ver + zoom * zoom) or 1.0
    sgn = lambda x: (1 if x > 0 else -1) if abs(x) / n >= 0.35 else 0  # noqa: E731
    return {"lateral": sgn(lat), "vertical": sgn(ver), "zoom": sgn(zoom)}


def scale_jump(prev, nxt):
    return abs(math.log(nxt["z_start"] / prev["z_end"]))


def continuity(prev, nxt):
    """Score for cutting from prev into nxt (v5).
    Framing: the next room opens at about the zoom the last one closed on (a wide close opens
    wide, a tight close opens tight), so the cut never jumps in scale.
    Motion: the next room moves the opposite way to the last one's closing motion - a pull-back
    is answered by a push-in, a pan left by a pan right, a tilt up by a tilt down or level."""
    s = -SCALE_JUMP_WEIGHT * max(0.0, scale_jump(prev, nxt) - SCALE_JUMP_FREE)
    pe, ne = prev["exit"], nxt["entry"]
    for k, opp, same in (("zoom", 2.0, -2.0), ("lateral", 1.0, -0.5), ("vertical", 0.3, -0.3)):
        p = pe[k] * ne[k]
        s += opp if p < 0 else same if p > 0 else 0
    return s


def signature(c):
    """What a viewer reads as 'the same move again': the side it opens on and its first motion."""
    st = c["start"]
    side = (1 if st["cx"] > 0.54 else -1 if st["cx"] < 0.46 else 0,
            1 if st["cy"] > 0.54 else -1 if st["cy"] < 0.46 else 0)
    e = c["entry"]
    first = max(("zoom", "lateral", "vertical"), key=lambda k: abs(c["entry_vec"][k]))
    return side, first, e[first]


class SalMap:
    def __init__(self, path):
        self.m = spectral_residual(load_gray(path))
        self.mean = float(self.m.mean()) or 1e-9

    def window_interest(self, g, st):
        """Mean saliency inside the window relative to the whole photo."""
        h, w = self.m.shape
        ww, wh = g.width(st["zoom"]), g.height(st["zoom"])
        x0, x1 = int((st["cx"] - ww / 2) * w), int(math.ceil((st["cx"] + ww / 2) * w))
        y0, y1 = int((st["cy"] - wh / 2) * h), int(math.ceil((st["cy"] + wh / 2) * h))
        return float(self.m[max(0, y0):y1, max(0, x0):x1].mean()) / self.mean


def category(shots, i, sal, thr, q25):
    s = shots[i]
    if i == 0 and is_exterior(s):
        return "exterior"
    if i == len(shots) - 1:
        return "final"
    if is_exterior(s):
        return "exterior"
    if is_bathroom(s):
        return "bath"
    if sal["spread"] > thr:
        return "wide"
    return "medium" if sal["spread"] > q25 else "tight"


# ---------------------------------------------------------------------------------------------
# selection: beam search over (move, d, v) per shot
# ---------------------------------------------------------------------------------------------

def geometry(s, move):
    proj = "rotate" if move in ROTATE_MOVES else "crop"
    return Geometry(s["source"]["width"], s["source"]["height"], OUT_W, OUT_H, proj)


def candidates(shots, sal, maps, thr, q25):
    out = []
    for i, s in enumerate(shots):
        name = s["image_uri"].split("/")[-1]
        cat = category(shots, i, sal[name], thr, q25)
        cands = []
        for move, suit in SUIT[cat].items():
            g = geometry(s, move)
            if s["role"] == "hero" and move in FEATURE_MOVES:
                suit += HERO_BONUS
            fn, uses_v = MOVES[move]
            lo = FINAL_MIN_S if i == len(shots) - 1 else SHOT_RANGE_S[0]
            for d in (1, -1):
                for v in ((1, -1) if uses_v else (1,)):
                    start, segs = fn(g, sal[name], d, v)
                    raw = make_path(g, start, segs, {"s0": 1, "s1": 1, "turn": TURN_COST})
                    if raw.distance < 1.0:
                        continue
                    secs = min(max(raw.cost / M_TARGET / FPS, lo), SHOT_RANGE_S[1])
                    note = {}
                    start, segs, timing = fit_speed(g, start, segs, round(secs * FPS), note)
                    path = make_path(g, start, segs, timing)
                    open_i = maps[name].window_interest(g, start)
                    close_i = maps[name].window_interest(g, segs[-1])
                    interest = 1.5 * max(-1, min(1, open_i - 1)) + 0.75 * max(-1, min(1, close_i - 1))
                    ev, xv = motion_vector(path, False), motion_vector(path, True)
                    c = {"move": move, "d": d, "v": v, "start": start, "segments": segs, "timing": timing,
                         "entry": classify(ev), "exit": classify(xv),
                         "entry_vec": dict(zip(("lateral", "vertical", "zoom"), ev)),
                         "z_start": start["zoom"], "z_end": segs[-1]["zoom"], "secs": round(secs, 2),
                         "suit": suit, "interest": round(interest, 4), "category": cat,
                         "distance": path.distance, "cost": path.cost, "fit": note}
                    c["signature"] = signature(c)
                    cands.append(c)
        out.append(cands)
    return out


def select(cands):
    beams = [(0.0, [], {})]
    for i, cs in enumerate(cands):
        nxt = []
        for score, seq, uses in beams:
            for ci, c in enumerate(cs):
                s = score + c["suit"] + c["interest"]
                if seq:
                    prev = cands[i - 1][seq[-1]]
                    s += continuity(prev, c)
                    if prev["move"] == c["move"]:
                        s -= 5
                    fam = c["move"] in REVEAL_FAMILY
                    if fam and prev["move"] in REVEAL_FAMILY:
                        s -= 4      # "start on a side, push, tilt, pull back" twice running
                    if prev["signature"][0] == c["signature"][0]:
                        s -= 1.0    # opens on the same side of the frame as the last shot
                    if prev["signature"][1:] == c["signature"][1:]:
                        s -= 1.0    # opens with the same first motion
                    if len(seq) > 1:
                        p2 = cands[i - 2][seq[-2]]
                        if p2["move"] == c["move"]:
                            s -= 2.5
                        if fam and p2["move"] in REVEAL_FAMILY:
                            s -= 1.5
                        if p2["signature"] == c["signature"]:
                            s -= 1.0
                if uses.get(c["move"], 0) >= MAX_USES:
                    s -= 8
                s -= 2.0 * uses.get((c["move"], c["d"], c["v"]), 0)   # same move, same direction again
                s -= 1.5 * uses.get((c["move"], c["d"]), 0)           # same move, same pan direction
                if c["move"] in REVEAL_FAMILY and sum(uses.get(m, 0) for m in REVEAL_FAMILY) >= 4:
                    s -= 6
                u = dict(uses)
                u[c["move"]] = u.get(c["move"], 0) + 1
                u[(c["move"], c["d"], c["v"])] = u.get((c["move"], c["d"], c["v"]), 0) + 1
                u[(c["move"], c["d"])] = u.get((c["move"], c["d"]), 0) + 1
                nxt.append((s, seq + [ci], u))
        if not nxt:
            raise RuntimeError(f"no continuous sequence reaches shot {i + 1}")
        nxt.sort(key=lambda b: (-b[0], b[1]))
        beams = nxt[:BEAM]
    best = beams[0]
    return [cands[i][ci] for i, ci in enumerate(best[1])], best[0]


# ---------------------------------------------------------------------------------------------
# durations and speed
# ---------------------------------------------------------------------------------------------

def allocate_durations(chosen, total, n_final):
    """Each shot's duration follows its move's distance (sized for M_TARGET px/frame), clamped,
    then scaled toward the baseline total (within TOTAL_TOLERANCE_S), integer centiseconds."""
    lo = [SHOT_RANGE_S[0]] * len(chosen)
    lo[n_final] = max(lo[n_final], FINAL_MIN_S)
    want = [min(max(c["secs"], lo[i]), SHOT_RANGE_S[1]) for i, c in enumerate(chosen)]
    s = sum(want)
    target = min(max(s, total - TOTAL_TOLERANCE_S), total + TOTAL_TOLERANCE_S)
    d = list(want)
    for _ in range(20):
        f = target / sum(d)
        d = [min(max(x * f, lo[i]), SHOT_RANGE_S[1]) for i, x in enumerate(d)]
        if abs(sum(d) - target) < 1e-6:
            break
    return [round(x, 2) for x in d], round(s, 2), round(target, 2)


def timing_for(cost, frames):
    """cost is the path's time-cost (distance plus turn cost); between turns the camera moves at
    hermite' x cost / frames px/frame."""
    mean = cost / max(1, frames - 1)
    # slopes above 1 would make the middle of the shot slower than its ends (at 3 it stops), so a
    # short move enters and leaves below V_CUT at a constant speed instead
    s = min(1.0, V_CUT / mean) if mean > 0 else 1.0
    return {"s0": round(s, 4), "s1": round(s, 4), "turn": TURN_COST}, mean


def fit_speed(g, start, segs, frames, note):
    """Shrink the move toward its end state (5% steps) until its peak speed is <= V_MAX."""
    end = segs[-1]
    f = 1.0
    while True:
        sc = lambda st: {**st, **_state(  # noqa: E731
            end["cx"] + f * (st["cx"] - end["cx"]), end["cy"] + f * (st["cy"] - end["cy"]),
            end["zoom"] * (st["zoom"] / end["zoom"]) ** f)}
        st2 = sc(start)
        segs2 = [{**sc(s), "t0": s["t0"], "t1": s["t1"]} for s in segs[:-1]] + [end]
        raw = make_path(g, st2, segs2, {"s0": 1, "s1": 1, "turn": TURN_COST})
        dist = raw.distance
        timing, mean = timing_for(raw.cost, frames)
        peak = hermite_slope_max(timing["s0"], timing["s1"]) * mean
        m = measured(make_path(g, st2, segs2, timing), frames, g)
        if (m["speed_max_px_f"] <= V_MAX + 1e-6 and m["min_margin_out_px"] >= MIN_MARGIN_OUT_PX
                and m["min_spp"] >= MIN_SPP - 1e-9) or f <= SHRINK_STEP:
            break
        f = round(f - SHRINK_STEP, 3)
    note.update({"shrink": f, "distance_px": round(dist, 1), "mean_px_f": round(mean, 3),
                 "planned_peak_px_f": round(peak, 3), "planned_cut_px_f": round(timing["s0"] * mean, 3)})
    return st2, segs2, timing


def measured(path, frames, g):
    fr = path.frames(frames)
    sp = [corner_rms(a, b, g) for a, b in zip(fr, fr[1:])]
    margins = [frame_margin_out_px(cx, cy, z, g.src_w, g.src_h, OUT_W, OUT_H, g.projection, g.hfov_deg)
               for cx, cy, z in fr]
    spp = [frame_spp(cx, cy, z, g.src_w, g.src_h, OUT_W, OUT_H, g.projection, g.hfov_deg) for cx, cy, z in fr]
    zs = [z for _, _, z in fr]
    v = [motion_px(a, b, g.src_w, g.src_h, OUT_W, OUT_H) for a, b in zip(fr, fr[1:])]
    turn = [math.degrees(vec_angle(a, b)) for a, b in zip(v, v[1:])] or [0.0]
    return {"turn_max_deg_f": round(max(turn), 3),"speed_first_px_f": round(sp[0], 3), "speed_cut_px_f": round(sp[-1], 3),
            "speed_max_px_f": round(max(sp), 3), "speed_min_px_f": round(min(sp), 3),
            "min_margin_out_px": round(min(margins), 3), "min_spp": round(min(spp), 4),
            "zoom_range": [round(min(zs), 4), round(max(zs), 4)]}


def corner_rms(a, b, g):
    from modules.camera_path import corner_rms_px
    return corner_rms_px(a, b, g.src_w, g.src_h, OUT_W, OUT_H)


# ---------------------------------------------------------------------------------------------

def build(baseline, saliency, images_dir):
    base = copy.deepcopy(baseline.get("plan", baseline))
    plan = copy.deepcopy(base)
    plan["plan_version"] = "1.3"
    shots = plan["shots"]
    sal = saliency["photos"]
    names = [s["image_uri"].split("/")[-1] for s in shots]
    maps = {n: SalMap(os.path.join(images_dir, n)) for n in set(names)}
    spreads = sorted(sal[n]["spread"] for n in names)
    thr = statistics.median(spreads)
    q25 = statistics.quantiles(spreads, n=4)[0]
    cands = candidates(shots, sal, maps, thr, q25)
    chosen, score = select(cands)
    durations, wanted_total, target_total = allocate_durations(chosen, base["total_duration_s"], len(shots) - 1)
    for s, d in zip(shots, durations):
        s["duration_s"] = d
        s["transition_in"] = {"type": "cut", "duration_s": 0.0}
    plan["total_duration_s"] = round(sum(durations), 2)
    timeline = compute_timeline(parse_plan_loose(plan))
    notes = []
    for i, (s, c) in enumerate(zip(shots, chosen)):
        g = geometry(s, c["move"])
        n = timeline[i]["render_frames"]
        note = {"shot_id": s["shot_id"], "room_type": s["room_type"], "category": c["category"],
                "move": c["move"], "d": c["d"], "v": c["v"], "entry": c["entry"], "exit": c["exit"],
                "suit": c["suit"], "interest": c["interest"], "duration_s": s["duration_s"],
                "frames": [timeline[i]["render_from"], timeline[i]["render_from"] + n]}
        if i:
            note["continuity_from_prev"] = round(continuity(chosen[i - 1], c), 3)
        # the move as designed (fit_speed shrinks from it again at the final duration)
        fn, _ = MOVES[c["move"]]
        start, segs = fn(g, saliency["photos"][s["image_uri"].split("/")[-1]], c["d"], c["v"])
        start, segs, timing = fit_speed(g, start, segs, n, note)
        s["camera"] = {"type": "compound", "move": c["move"], "start": start, "segments": segs, "timing": timing}
        if g.projection == "rotate":
            s["camera"].update({"projection": "rotate", "hfov_deg": g.hfov_deg})
        note.update(measured(make_path(g, start, segs, timing), n, g))
        note["zoom_start_end"] = [start["zoom"], segs[-1]["zoom"]]
        if notes:
            note["cut_zoom_ratio"] = round(start["zoom"] / notes[-1]["zoom_start_end"][1], 4)
        note["signature"] = c["signature"]
        note["camera"] = s["camera"]
        notes.append(note)
    plan["provenance"]["planner_config_version"] = f"{PLANNER_VERSION}/v5"
    plan["provenance"].pop("plan_hash", None)
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    parse_plan(plan)  # contract checks, including every-frame bounds
    summary = {"score": round(score, 3), "spread_q25": q25, "moves": [c["move"] for c in chosen],
               "directions": [(c["d"], c["v"]) for c in chosen], "spread_median": thr,
               "total_duration_s": plan["total_duration_s"], "distance_sized_total_s": wanted_total,
               "baseline_total_s": base["total_duration_s"], "v_cut": V_CUT, "v_max": V_MAX}
    return plan, {"summary": summary, "shots": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--saliency", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    plan, notes = build(json.load(open(args.baseline)), json.load(open(args.saliency)), args.images_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(plan, open(os.path.join(args.out_dir, "v5.plan.json"), "w"), indent=1)
    json.dump(notes, open(os.path.join(args.out_dir, "v5_plan_notes.json"), "w"), indent=1)
    print(json.dumps(notes["summary"]))
    for n in notes["shots"]:
        e, x = n["entry"], n["exit"]
        print(f"{n['shot_id']} {n['category']:<8} {n['move']:<12} d{n['d']:+d} v{n['v']:+d} {n['duration_s']:>4}s "
              f"in(z{e['zoom']:+d} x{e['lateral']:+d} y{e['vertical']:+d}) out(z{x['zoom']:+d} x{x['lateral']:+d} y{x['vertical']:+d}) "
              f"cont {n.get('continuity_from_prev', '-')} zcut {n.get('cut_zoom_ratio', '-')} | shrink {n['shrink']} "
              f"turn {n['turn_max_deg_f']} speed first {n['speed_first_px_f']} "
              f"max {n['speed_max_px_f']} min {n['speed_min_px_f']} cut {n['speed_cut_px_f']} | margin {n['min_margin_out_px']} "
              f"spp {n['min_spp']} zoom {n['zoom_range']}")


if __name__ == "__main__":
    main()
