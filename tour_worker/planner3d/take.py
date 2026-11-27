"""One take through the home (user 2026-11-27: "enter the room, circle it, then see its exit and then
enter the other room and do the panning and other camera moves all in one shot"; rooms joined by a
walking route are one take, a transition only where none joins them; room changes on the beat; the
tour near 90 s).

Each room is a loop: from where the camera comes in, round the room on a ring kept off its walls,
looking at the room's middle (or its main piece) - circling it - and over its last stretch turning
onto the door it leaves by. Between two rooms the camera walks through the door (RoomMoves.bridge),
its look already along the way. A small room with one door (a bathroom) is stepped into, panned
across and stepped out of. Every loop is judged by the planner's render checks (the lens, the
void); the variants are tried by how far they keep off the walls and which way round they go.
"""
from __future__ import annotations

import math

import numpy as np

from tour_worker.planner3d.moves import (AIM_Z_MAX, BRIDGE_DPS, BRIDGE_MPS, EYE, SAMPLES, VOID_MAX,
                                         Move, _route_point, resample, smooth)

LOOP_INSETS = (0.7, 0.55, 0.9)    # m off the room's walls the loop keeps, tried in this order
LOOP_RAYS = 72                    # directions the room's ring is measured in
LOOP_MIN_ARC = math.radians(120)  # a loop goes at least this far round (it circles the room)
LOOP_MAX_ARC = math.radians(300)  # ... and at most this far
LOOP_FLOOR_ARC = math.radians(60) # ... less only in a room whose share of the time is short
LOOP_MAX_OCC = 0.4                # a walk round a room passes furniture: its cover may reach this (a held shot: MAX_OCC)
LOOP_WARN_OCC = 0.65              # ... past this a loop is used only when the room has no other, and named
LOOP_EXIT_SHARE = 0.3             # of its length, the last share turns its look onto the exit
LOOP_SAMPLES = 120
LOOP_CLEAR = 0.3                  # m: the loop's line keeps this off the walls everywhere
LOOP_FREE = 0.45                  # m clear a ring point stands in (open floor, not a squeeze)
LOOP_ORBIT_DPS = 8.0              # deg/s its look may turn while it circles (it keeps the middle in view)
LOOK_Z = EYE - 0.25               # its aim's height (as the room shots')
LOOK_MIN_M, LOOK_MAX_M = 1.5, 4.0
DOOR_IN = 0.7                     # m inside a door a loop starts / ends
EXIT_LOOK_M = 1.5                 # m past the exit door its look goes at the end
PEEK_RING_M = 0.45                # a room whose ring is mostly narrower than this is peeked into
PEEK_SWEEP = math.radians(70)     # ... and panned across this far
TAKE_ROUTE_MAX_M = 12.0           # m: a longer walk between two rooms is a transition, not a walk
LOOP_BUDGET_SLACK = 1.2           # a loop may run this much over its room's share of the time
DOOR_WALK_S = 5.0                 # s a walk through a door is budgeted at (before it is planned)
DEAD_END_M = 1.2                  # m: a room entered and left within this of one point is a dead end
TAKE_PEAK_SHARE = 0.85            # a take's piece keeps its peak turn / travel under this share of the limits
TAKE_OVER_S = 2.0                 # s a tour of takes may run past its length (the soft limit)
ROOM_WEIGHT_HERO = 1.5            # a hero room's share of the time against its size's


def _ring(space, room: str, inset: float, subject=None):
    """(centre, angles, radii): the floor round the room's main piece (subject; else the room's most
    open point) - per direction, the open floor between the piece and the wall, at its middle but
    `inset` off the wall (furniture stands in a room's middle: a ring round the room's centre had
    nowhere to go, 2026-11-27). NaN where there is no open floor that way (a bed's head against a wall)."""
    idx = space.room_keys.index(room)
    ids = [space.room_keys.index(r) for r in sorted(space.spaces.get(room, {room}))]
    own = space.room_of == idx
    free = np.isin(space.room_of, ids) & (space.dt >= LOOP_FREE)
    ys, xs = np.nonzero(own & free)
    if not len(ys):
        return None
    cells = np.c_[space.lo[0] + (xs + .5) * 0.05, space.lo[1] + (ys + .5) * 0.05]
    if subject is not None:
        centre = np.asarray(subject[:2], float)
    else:                                              # the room's most open point
        dts = np.array([space.clearance(c) for c in cells])
        centre = cells[int(np.argmax(dts))]
    ang = np.linspace(-math.pi, math.pi, LOOP_RAYS, endpoint=False)
    radii = np.full(len(ang), np.nan)
    for n, a in enumerate(ang):
        d = np.array([math.cos(a), math.sin(a)])
        start = end = None
        for r in np.arange(0.4, 6.0, 0.05):
            i, j = space.cell(centre + r * d)
            if not (0 <= i < free.shape[0] and 0 <= j < free.shape[1]):
                break
            ok = bool(free[i, j]) and bool(own[i, j])
            if ok and start is None:
                start = r
            if start is not None:
                if not ok:
                    break
                end = r
        if start is None or end is None or end - start < 0.1:
            continue
        radii[n] = max(start, min(0.5 * (start + end), end - (inset - LOOP_FREE)))
    if np.isfinite(radii).sum() < 4:
        return None
    return centre, ang, radii


def _clear_line(space, a, b) -> bool:
    n = max(2, int(np.linalg.norm(b - a) / 0.05))
    return all(space.clearance(a + (b - a) * t) >= LOOP_CLEAR for t in np.linspace(0, 1, n))


def _aim(eye, look_at):
    d = look_at[:2] - eye
    r = float(np.clip(np.linalg.norm(d), LOOK_MIN_M, LOOK_MAX_M))
    h = math.atan2(d[1], d[0])
    return h, r


def room_loop(mover, room: str, entry, exit_pt, exit_after, budget_s: float, inset: float, way: float,
              subject=None, piece=None, arc_cap: float = LOOP_MAX_ARC) -> Move | None:
    """The room's loop: from entry (a point just inside the door it comes in by; None for the tour's
    first room: it starts on the far side from its exit) round the ring `way` (+1 anticlockwise, -1
    clockwise) to exit_pt (just inside the door it leaves by; None: the tour's last room - it ends
    looking at the room), turning its look onto exit_after (past the door) over its last stretch.
    Within budget_s at the 3D pace (BRIDGE_MPS; its look at most LOOP_ORBIT_DPS)."""
    sp = mover.space
    rg = _ring(sp, room, inset, subject)
    if rg is None:                                     # no open floor round it: peeked into, from its middle
        ys, xs = np.nonzero(sp.room_of == sp.room_keys.index(room))
        if not len(ys):
            return None
        mid = np.array([sp.lo[0] + (xs.mean() + .5) * 0.05, sp.lo[1] + (ys.mean() + .5) * 0.05])
        return _peek(mover, room, entry, exit_pt, exit_after, mid)
    centre, ang, radii = rg
    look_c = centre
    if np.nanmedian(radii) - 0.4 < PEEK_RING_M and np.isfinite(radii).mean() < 0.3:
        return _peek(mover, room, entry, exit_pt, exit_after, centre)
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi   # noqa: E731
    if entry is not None:
        entry = sp.snap(entry, LOOP_CLEAR, 0.5)
        if entry is None:
            return None
    if exit_pt is not None:
        exit_pt = sp.snap(exit_pt, LOOP_CLEAR, 0.5)
        if exit_pt is None:
            return None
    th_x = math.atan2(exit_pt[1] - centre[1], exit_pt[0] - centre[0]) if exit_pt is not None else None
    if entry is None:                                  # the tour's first room: start across from the exit
        th_e = (th_x + math.pi) if th_x is not None else float(ang[int(np.nanargmax(radii))])
    else:
        th_e = math.atan2(entry[1] - centre[1], entry[0] - centre[0])
    if th_x is None:
        arc = LOOP_MIN_ARC
    else:
        arc = (way * (th_x - th_e)) % (2 * math.pi)
        if arc < LOOP_MIN_ARC:                         # (the exit beside the way in: the long way round)
            arc = LOOP_MAX_ARC
        arc = min(arc, LOOP_MAX_ARC, max(LOOP_FLOOR_ARC, arc_cap))
    ring_pts = []
    for t in np.linspace(0, arc, max(8, int(math.degrees(arc) / 5))):
        th = wrap(th_e + way * t)
        k = int(np.argmin(np.abs([wrap(a - th) for a in ang])))
        if np.isfinite(radii[k]):
            ring_pts.append(centre + radii[k] * np.array([math.cos(ang[k]), math.sin(ang[k])]))
    if len(ring_pts) < 4:
        return None
    waypts = ([np.asarray(entry, float)] if entry is not None else []) + ring_pts + \
             ([np.asarray(exit_pt, float)] if exit_pt is not None else [])
    # joined by straight lines where clear, else by the walking path round what is in the way
    poly = [waypts[0]]
    for b in waypts[1:]:
        a = poly[-1]
        if float(np.linalg.norm(b - a)) < 0.05:
            continue
        if _clear_line(sp, a, b):
            poly.append(b)
            continue
        route = sp.paths_between(np.array([a, b]), sp.dt >= LOOP_CLEAR)(0, 1)
        if route is None or len(route) < 2:
            return None
        poly += [np.asarray(q, float) for q in route[1:]]
    poly = np.array(poly)
    path = resample(smooth(sp, poly, LOOP_CLEAR), LOOP_SAMPLES)
    path[0] = poly[0]
    path[-1] = poly[-1]
    arcl = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    L = float(arcl[-1])
    u = arcl / max(L, 1e-9)
    hd, rr = [], []
    for k, e in enumerate(path):
        h, r = _aim(e, look_c)
        hd.append(h)
        rr.append(r)
    hd = np.unwrap(np.asarray(hd))
    if exit_pt is not None:                            # the last stretch: onto the way out
        ease = lambda x: x * x * (3 - 2 * x)   # noqa: E731
        w = ease(np.clip((u - (1 - LOOP_EXIT_SHARE)) / LOOP_EXIT_SHARE, 0, 1))
        hx = np.array([math.atan2(exit_after[1] - e[1], exit_after[0] - e[0]) for e in path])
        hd = hd + w * np.array([wrap(x - h) for x, h in zip(hx, hd)])
        rr = np.asarray(rr) * (1 - w) + w * np.clip(np.linalg.norm(exit_after[None] - path, axis=1), LOOK_MIN_M, LOOK_MAX_M)
    tgt = np.c_[path + np.asarray(rr)[:, None] * np.c_[np.cos(hd), np.sin(hd)], np.full(len(path), min(LOOK_Z, AIM_Z_MAX))]
    turn = float(np.degrees(np.abs(np.diff(hd)).sum()))
    dur = max(L / BRIDGE_MPS, turn / LOOP_ORBIT_DPS, 4.0)
    m = Move("loop", resample(path, SAMPLES), resample(tgt, SAMPLES), 0.0, easing="linear", piece=piece,
             why={"travel_m": round(L, 2), "turn_deg": round(turn, 1), "dur_s": round(dur, 2),
                  "arc_deg": round(math.degrees(arc), 1), "inset_m": inset, "way": way})
    if dur > budget_s * 1.15 and arc > LOOP_MIN_ARC + 1e-6:
        m.why["over_budget_s"] = round(dur - budget_s, 1)
    return m


def _peek(mover, room, entry, exit_pt, exit_after, centre) -> Move | None:
    """A small room (a bathroom): stepped into, panned across, stepped out of - entry and exit by the
    same door (or across it to the exit)."""
    sp = mover.space
    p0 = np.asarray(entry if entry is not None else centre, float)
    into = centre - p0
    if float(np.linalg.norm(into)) < 1e-6:
        return None
    into /= float(np.linalg.norm(into))
    stand = sp.snap(p0 + 0.4 * into, LOOP_CLEAR, 0.3)
    if stand is None:
        return None
    h_in = math.atan2(into[1], into[0])
    end = np.asarray(exit_pt, float) if exit_pt is not None else stand
    h_out = math.atan2(exit_after[1] - end[1], exit_after[0] - end[0]) if exit_pt is not None else h_in
    n1, n2, n3 = 25, 50, 45
    eye = np.r_[p0 + np.linspace(0, 1, n1)[:, None] * (stand - p0)[None],
                np.repeat(stand[None], n2, 0),
                stand + np.linspace(0, 1, n3)[:, None] * (end - stand)[None]]
    ease = lambda x: x * x * (3 - 2 * x)   # noqa: E731
    sweep = np.r_[np.full(n1, -PEEK_SWEEP / 2), -PEEK_SWEEP / 2 + PEEK_SWEEP * ease(np.linspace(0, 1, n2)),
                  np.full(n3, PEEK_SWEEP / 2)]
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi   # noqa: E731
    hd = h_in + sweep
    w = ease(np.r_[np.zeros(n1 + n2), np.linspace(0, 1, n3)])
    hd = np.unwrap(hd + w * np.array([wrap(h_out - h) for h in hd]))
    tgt = np.c_[eye + 2.0 * np.c_[np.cos(hd), np.sin(hd)], np.full(len(eye), min(LOOK_Z, AIM_Z_MAX))]
    L = float(np.linalg.norm(np.diff(eye, axis=0), axis=1).sum())
    turn = float(np.degrees(np.abs(np.diff(hd)).sum()))
    dur = max(L / BRIDGE_MPS, turn / BRIDGE_DPS, 4.0)
    return Move("loop", resample(eye, SAMPLES), resample(tgt, SAMPLES), 0.0, easing="linear",
                why={"travel_m": round(L, 2), "turn_deg": round(turn, 1), "dur_s": round(dur, 2), "peek": True})


def _door(mover, a: str, b: str):
    """(exit point in a, entry point in b, a point past the door) on the walking route a -> b, or
    None when no route (or only a long one) joins them."""
    sp = mover.space
    r = sp.route(a, b)
    if r is None:
        return None
    poly = r[0]
    s_ = mover._along(poly)
    if float(s_[-1]) > TAKE_ROUTE_MAX_M:
        return None
    cr = mover.crossings(r)
    leave = next((c for c in cr if c["from"] == a), None)
    into = next((c for c in reversed(cr) if c["to"] == b), None)
    if leave is None or into is None:
        return None
    x = _route_point(poly, s_, max(0.0, leave["s"] - DOOR_IN))
    e = _route_point(poly, s_, min(float(s_[-1]), into["s"] + DOOR_IN))
    past = _route_point(poly, s_, min(float(s_[-1]), leave["s"] + EXIT_LOOK_M))
    return x, e, past


def _choose(mover, room, entry, exit_pt, past, budget, sub, arc_scale=1.0):
    """The room's loop: its variants (insets x ways), each judged by the render checks; the least
    covered within LOOP_MAX_OCC nearest its time - else the least covered at all (named by the
    caller). (move, occ) or (None, None)."""
    found = []
    for inset in LOOP_INSETS:
        for way in (1.0, -1.0):
            m = room_loop(mover, room, entry, exit_pt, past, budget, inset, way, *(sub or (None, None)),
                          arc_cap=LOOP_MAX_ARC * arc_scale)
            if m is not None and m.why["dur_s"] > LOOP_BUDGET_SLACK * budget and "arc_deg" in m.why:
                cap = math.radians(m.why["arc_deg"]) * budget / m.why["dur_s"]
                m = room_loop(mover, room, entry, exit_pt, past, budget, inset, way, *(sub or (None, None)),
                              arc_cap=cap) or m
            if m is None:
                continue
            occ = 0.0
            if mover.occ is not None:
                occ = mover._cover(m)
                if mover._void(m) > VOID_MAX:
                    continue
            m.occ = occ
            found.append((occ, abs(m.why["dur_s"] - budget), m))
    if not found:
        return None, None
    ok = [f for f in found if f[0] <= LOOP_MAX_OCC]
    occ, _, m = min(ok, key=lambda f: (f[1], f[0])) if ok else min(found, key=lambda f: f[0])
    return m, occ


def plan_takes(mover, rooms: list[str], weights: dict[str, float], total_s: float, subjects: dict) -> tuple[list, list[str]]:
    """The tour's takes: [(room, Move, seamless_out)] in tour order - each room's loop, and between two
    rooms a walking route joins the walk through the door (RoomMoves.bridge, its room the next one's);
    a transition (seamless_out False) where none does. Each loop's time is its room's share of
    total_s (weights); over it by more than TAKE_OVER_S, every loop goes round less (one more pass).
    (moves, warnings)."""
    out, warns = _takes(mover, rooms, weights, total_s, subjects, 1.0)
    total = _fit_time(mover, out, total_s)
    if total > total_s + TAKE_OVER_S and out:          # at the speed limits still too long: round less
        out, warns = _takes(mover, rooms, weights, total_s, subjects, max(0.3, total_s / total))
        total = _fit_time(mover, out, total_s)
    if total > total_s + TAKE_OVER_S:
        warns.append(f"take_long:{total:.0f}s")
    return out, warns


def _fit_time(mover, out, total_s: float) -> float:
    """Each piece's time (why.dur_s): at least what keeps its peak turn and travel under
    TAKE_PEAK_SHARE of the speed limits (as the renderer plays it: steady along its arc), and the
    take as a whole no longer than total_s - a take has no cut to match speeds across, so it walks
    faster than the shots' pace when the home needs it (35 m of rooms at 0.22 m/s was 245 s,
    2026-11-27). Never slower than planned, never under 3 s. Returns the tour's length."""
    from tour_worker.planner3d.moves import MAX_MOVE_MPS, MAX_TURN_DPS
    floor = []
    for _, m, _ in out:
        d = float(m.why["dur_s"])
        pt, pm = mover.peaks(m, d)
        need = max(pt / (TAKE_PEAK_SHARE * MAX_TURN_DPS), pm / (TAKE_PEAK_SHARE * MAX_MOVE_MPS))
        floor.append(max(3.0, d * need))
    planned = sum(float(m.why["dur_s"]) for _, m, _ in out)
    squeeze = min(1.0, total_s / planned) if planned > 0 else 1.0
    total = 0.0
    for (_, m, _), lo in zip(out, floor):
        d = max(lo, float(m.why["dur_s"]) * squeeze)
        m.why["planned_dur_s"] = m.why["dur_s"]
        m.why["dur_s"] = round(d, 2)
        total += d
    return total


def _takes(mover, rooms, weights, total_s, subjects, arc_scale):
    warns: list[str] = []
    doors = [(_door(mover, a, b) if a != b else None) for a, b in zip(rooms[:-1], rooms[1:])]
    n_walks = sum(d is not None for d in doors)
    loop_s = max(10.0, total_s - DOOR_WALK_S * n_walks)
    wsum = sum(weights.get(r, 1.0) for r in rooms) or 1.0
    out: list = []
    prev_loop = None
    for i, room in enumerate(rooms):
        din = doors[i - 1] if i > 0 else None
        dout = doors[i] if i < len(doors) else None
        entry = din[1] if din is not None else None
        exit_pt, past = (dout[0], dout[2]) if dout is not None else (None, None)
        dead_end = (entry is not None and exit_pt is not None
                    and float(np.linalg.norm(np.asarray(entry) - np.asarray(exit_pt))) < DEAD_END_M)
        if dead_end:
            # left by the door it came in by: circled and left with a transition, not turned round in
            # (user 2026-11-27: four dead ends' turnarounds were ~80 s of a 90 s tour)
            exit_pt, past = None, None
        budget = loop_s * weights.get(room, 1.0) / wsum
        m, occ = _choose(mover, room, entry, exit_pt, past, budget, subjects.get(room), arc_scale)
        if m is None or occ > LOOP_WARN_OCC:
            warns.append(f"take_room_no_loop:{room}")
            prev_loop = None
            continue
        if occ > LOOP_MAX_OCC:
            warns.append(f"take_loop_close:{room}:{occ:.2f}")
        if prev_loop is not None and din is not None:
            br = mover.bridge(prev_loop[1], m)
            if br is not None:
                out[-1] = (out[-1][0], out[-1][1], True)
                out.append((room, br, True))
            else:
                warns.append(f"take_cut:{rooms[i - 1]}->{room}")
        out.append((room, m, False))
        prev_loop = None if dead_end else (room, m)
        if dead_end:
            m.why["dead_end"] = True
    return out, warns
