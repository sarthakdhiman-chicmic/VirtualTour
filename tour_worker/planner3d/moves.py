"""Camera moves for one shot of a room, generated from the room's free floor and checked against the
tour rules (user reviews 2026-11-12/13), then scored; the best one that passes wins.

Kinds of move:
  walkway  along a free path in the room (or its open-plan space), looking at the room's main piece
           of furniture: the piece's top surface stays in frame all along, seen three-quarter-on
           from a sitting-back distance; moving across the view gives parallax
  glide    along a free path, looking ahead into the room's deepest view (halls, kitchens, rooms
           without a main piece; the opening shot of a tour)
  pan      from an open spot, a slow sweep across the room's deepest view with a little drift
           (small rooms, balconies; the closing shot)
  hold     the fallback when every move runs into something near the lens: the least blocked framed
           spot with a breath of pull-back, and a warning

Rules every move is checked against:
  speed    peak turn <= MAX_TURN_DPS and travel <= MAX_MOVE_MPS as the worker draws the path
           (easing included), planned to PRE_SYNC_SHARE of them so the beat sync cannot push it over
  horizon  pitch at most PITCH_LADDER[0] below level (a small room may step down the ladder)
  zoom     one field of view for the whole tour (HFOV), no zoom moves
  framing  the piece's top surface in frame (as much of its long side as fits: CORES)
  lens     nothing near the lens (OCC_NEAR_KIND m) covers more than MAX_OCC of the frame's middle
           (rendered)
  cuts     no jump cut from the previous shot (JUMP_ANGLE / JUMP_DIST) and no repeat of an earlier
           framing of the same room
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from tour_worker.scene.camera import BezierCameraPath, intrinsics, look_at

EYE = 1.50                    # eye height (m)
LOOK_DROP = 0.30              # a level look drops this much over ~2.5 m
HFOV = 78.0                   # one field of view for the whole tour
MAX_TURN_DPS, MAX_MOVE_MPS, PRE_SYNC_SHARE = 10.0, 0.8, 0.8
TURN_BUDGET, MOVE_BUDGET = MAX_TURN_DPS * PRE_SYNC_SHARE, MAX_MOVE_MPS * PRE_SYNC_SHARE
PITCH_LADDER = (10.0, 13.0, 16.0)
JUMP_ANGLE, JUMP_DIST = 30.0, 1.5
RMIN_LADDER = (0.28, 0.22)    # camera body clearance, a little less in small rooms
MOVE_LEN = (0.6, 2.6)
CORES = (0.75, 0.6, 0.5, 0.4)
SIT_BACK = 2.4                # m from the piece at the start of a walkway move
OCC_NEAR, MAX_OCC, OCC_W, OCC_CHECK_KIND = 2.0, 0.20, 3.0, 15
OPENING_GLIDE = 1.5          # the tour opens by moving into the home
# near-lens distance per move: a walkway composes on a piece, so nothing may stand within 2 m in
# front of it; a glide or pan looks down a room, where walls 1.2 m+ away are the room itself
OCC_NEAR_KIND = {"walkway": 2.0, "glide": 1.2, "pan": 1.2}
PAN_SWEEP = 50.0
SAMPLES = 50
FRAME_W, FRAME_H = 1920, 1080


@dataclass
class Move:
    kind: str
    eye: np.ndarray             # (n, 2) planner xy, at EYE
    tgt: np.ndarray             # (n, 3) planner points
    score: float
    pitch: float = PITCH_LADDER[0]
    piece: object = None
    occ: float | None = None
    why: dict = field(default_factory=dict)

    @property
    def first(self):
        return self.eye[0], self.tgt[0]

    @property
    def last(self):
        return self.eye[-1], self.tgt[-1]


# ---- geometry helpers ----------------------------------------------------------------------------
def resample(poly: np.ndarray, n: int) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(poly[:1], n, axis=0)
    s = np.linspace(0, d[-1], n)
    return np.c_[[np.interp(s, d, poly[:, k]) for k in range(poly.shape[1])]].T


def smooth(space, poly: np.ndarray, rmin: float, sigma_m: float = 0.35) -> np.ndarray:
    """Gaussian-smoothed path (ends fixed), as smooth as the clearance allows."""
    step = 0.05
    d = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
    p = resample(poly, max(2, int(d / step) + 1))
    for sig in (sigma_m, sigma_m * 0.6, sigma_m * 0.3):
        q = ndi.gaussian_filter1d(p, sig / step, axis=0, mode="nearest")
        w = np.clip(np.r_[np.linspace(0, 1, min(len(q), 12)), np.ones(max(0, len(q) - 12))], 0, 1)
        w = np.minimum(w, w[::-1])[:, None]
        q = p * (1 - w) + q * w
        if min(space.clearance(x) for x in q) >= rmin * 0.95:
            return q
    return p


def fit_cubic(pts: np.ndarray):
    """Least-squares cubic through pts with the end points fixed (chord-length parameters)."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    u = d / d[-1] if d[-1] > 1e-9 else np.linspace(0, 1, len(pts))
    P0, P3 = pts[0], pts[-1]
    b0, b1, b2, b3 = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3
    sol, *_ = np.linalg.lstsq(np.c_[b1, b2], pts - np.outer(b0, P0) - np.outer(b3, P3), rcond=None)
    ctrl = np.array([P0, sol[0], sol[1], P3])
    fit = np.outer(b0, ctrl[0]) + np.outer(b1, ctrl[1]) + np.outer(b2, ctrl[2]) + np.outer(b3, ctrl[3])
    return ctrl, float(np.abs(fit - pts).max())


def fit_pair(eye: np.ndarray, tgt: np.ndarray, tol: float = 0.03):
    """Eye and target polylines -> 1-3 joined cubic segments each: [(eye ctrl, target ctrl)], error."""
    n = len(eye)
    for k in (1, 2, 3):
        cuts = np.linspace(0, n - 1, k + 1).astype(int)
        segs, worst = [], 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            ce, ee = fit_cubic(eye[a:b + 1])
            ct, et = fit_cubic(tgt[a:b + 1])
            worst = max(worst, ee, et / 4)            # the target can be looser
            segs.append((ce, ct))
        if worst <= tol:
            break
    return segs, worst


def eye3(eye2: np.ndarray) -> np.ndarray:
    return np.c_[eye2, np.full(len(eye2), EYE)]


def speeds_of(segments: list[dict], duration_s: float, easing: str = "ease_in_out", samples: int | None = None,
              fps: int = 30) -> tuple[float, float]:
    """(peak turn deg/s, peak travel m/s) of bezier segments as the worker draws them."""
    path = BezierCameraPath(segments, [HFOV, HFOV], easing=easing)
    n = samples or max(2, int(round(duration_s * fps)))
    poses = [path.pose(f / (n - 1)) for f in range(n)]
    eye = np.array([p[0] for p in poses])
    tgt = np.array([p[1] for p in poses])
    d = tgt - eye
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    dt = duration_s / (n - 1)
    turn = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))) / dt
    move = np.linalg.norm(np.diff(eye, axis=0), axis=1) / dt
    return float(turn.max()), float(move.max())


def trial_speeds(eye2: np.ndarray, tgt: np.ndarray, dur: float) -> tuple[float, float]:
    segs, _ = fit_pair(eye3(eye2), tgt)
    return speeds_of([{"eye": e, "target": t} for e, t in segs], dur, samples=91)


def cap_pitch(eye2: np.ndarray, tgt: np.ndarray, pitch_max: float) -> np.ndarray:
    """Targets raised where the camera would look more than pitch_max below the horizon."""
    t = np.array(tgt, float)
    h = np.linalg.norm(t[:, :2] - np.asarray(eye2)[:, :2], axis=1)
    t[:, 2] = np.maximum(t[:, 2], EYE - h * math.tan(math.radians(pitch_max)))
    return t


def _dir(e, t) -> np.ndarray:
    d = np.array([t[0] - e[0], t[1] - e[1], t[2] - EYE])
    return d / np.linalg.norm(d)


def jump_cut(a, b) -> bool:
    """Two poses (eye xy, target) too alike for a cut: < JUMP_ANGLE apart and < JUMP_DIST apart."""
    if a is None or b is None:
        return False
    (e0, t0), (e1, t1) = a, b
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(_dir(e0, t0) @ _dir(e1, t1))))))
    return ang < JUMP_ANGLE and float(np.linalg.norm(np.asarray(e0[:2]) - np.asarray(e1[:2]))) < JUMP_DIST


def repeats(first, last, earlier) -> bool:
    """The move starts or ends on a framing an earlier shot of the room already had."""
    return any(jump_cut(p, q) for p in (first, last) for f, l_ in earlier for q in (f, l_))


def in_frame(p2, target, piece, core: float) -> bool:
    """The piece's top surface (long side cut to core) inside the frame from p2 aimed at target."""
    V, K = look_at([p2[0], p2[1], EYE], target), intrinsics(HFOV, FRAME_W, FRAME_H)
    cam = piece.top_corners(core) @ V[:3, :3].T + V[:3, 3]
    if (cam[:, 2] < 0.3).any():
        return False
    px = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    py = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return px.min() > 0.03 * FRAME_W and px.max() < 0.97 * FRAME_W and py.min() > 0.03 * FRAME_H and py.max() < 0.97 * FRAME_H


def parallax(eye2: np.ndarray, tgt: np.ndarray) -> float:
    """1: the camera moves straight across its view; 0: along it."""
    mv = eye2[-1] - eye2[0]
    if np.linalg.norm(mv) < 1e-6:
        return 0.0
    mv /= np.linalg.norm(mv)
    k = len(eye2) // 2
    view = tgt[k][:2] - eye2[k]
    view /= max(1e-9, np.linalg.norm(view))
    return abs(mv[0] * view[1] - mv[1] * view[0])


# ---- the planner for one room --------------------------------------------------------------------
class RoomMoves:
    def __init__(self, space, occlusion=None):
        self.space, self.occ = space, occlusion

    def best(self, room: str, dur: float, *, earlier=(), prev=None, piece_rank: int = 0,
             opening: bool = False, closing: bool = False) -> tuple[Move | None, list[str]]:
        """The best move for a shot of `room` lasting dur s. earlier: (first, last) poses of the
        room's earlier shots; prev: the previous shot's last pose. Returns (move, warnings)."""
        sp = self.space
        pieces = sp.main_pieces(room)
        piece = pieces[piece_rank % len(pieces)] if pieces else None
        cands: list[Move] = []
        for rmin in RMIN_LADDER:
            mask = sp.space_mask(room, rmin)
            pts = sp.spread_points(mask, 36)
            if len(pts) < 2:
                continue
            paths = sp.paths_between(pts, mask)
            # a shot starts in its own room (it may then move on through an open-plan space)
            own = {k for k, p in enumerate(pts) if sp.room_at(p) == room}
            if not own:
                continue
            if piece is not None:
                for pitch in PITCH_LADDER:
                    walk = self._walkway(room, piece, pts, paths, rmin, dur, pitch, prev, earlier, own)
                    if walk:
                        cands += walk
                        break
            cands += self._glide(room, pts, paths, rmin, dur, prev, earlier, opening, own)
            cands += self._pan(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier, closing)
            if cands:
                break
        if not cands:
            return None, [f"room_no_move:{room}"]
        cands.sort(key=lambda m: -m.score)
        if self.occ is not None:                      # occlusion renders work on this room's splats
            mask = sp.space_mask(room, RMIN_LADDER[-1])
            ys, xs = np.nonzero(mask)
            cells = np.c_[sp.lo[0] + (xs + .5) * 0.05, sp.lo[1] + (ys + .5) * 0.05]
            c = cells.mean(0)
            self.occ.focus(c, float(np.linalg.norm(cells - c, axis=1).max()) + OCC_NEAR + 0.5)
        # best first; speed and occlusion checked lazily, stopping once no later candidate can win
        best, least, checked = None, None, {}
        for m in cands:
            if best is not None and m.score <= best.score - OCC_W * best.occ:
                break
            if checked.get(m.kind, 0) >= OCC_CHECK_KIND:      # every kind gets its share of checks
                continue
            if "turn_dps" not in m.why:
                turn, move = trial_speeds(m.eye, m.tgt, dur)
                m.why.update(turn_dps=round(turn, 1), move_mps=round(move, 2))
                if turn > TURN_BUDGET or move > MOVE_BUDGET:
                    continue
            checked[m.kind] = checked.get(m.kind, 0) + 1
            m.occ = self._cover(m)
            if least is None or m.occ < least.occ:
                least = m
            if m.occ > MAX_OCC:
                continue
            if best is None or m.score - OCC_W * m.occ > best.score - OCC_W * best.occ:
                best = m
        if best is None and least is None:
            return None, [f"room_no_move:{room}"]
        if best is not None:
            best.why.update(candidates=len(cands), occlusion=round(best.occ, 3))
            return best, []
        hold = self._hold(least, room)
        return hold, [f"view_partly_blocked:{room}"]

    # ---- move kinds ------------------------------------------------------------------------------
    def _path(self, paths, i, j, rmin):
        poly = paths(i, j)
        if poly is None or len(poly) < 2:
            return None, 0.0
        q = smooth(self.space, poly, rmin)
        return q, float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())

    def _walkway(self, room, piece, pts, paths, rmin, dur, pitch, prev, earlier, own) -> list[Move]:
        top = np.array([*piece.centre, piece.top])
        out = []
        keep = [k for k, p in enumerate(pts) if piece.dist(p) > 0.05]
        for a_i, a in enumerate(keep):
            for b in keep[a_i + 1:]:
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not MOVE_LEN[0] <= L <= MOVE_LEN[1]:
                    continue
                for i, j in ((a, b), (b, a)):
                    if i not in own:
                        continue
                    path, plen = self._path(paths, i, j, rmin)
                    if path is None or plen > 1.3 * L or plen / dur * 1.5 > MOVE_BUDGET:
                        continue
                    path = resample(path, SAMPLES)
                    tgt = cap_pitch(path, np.repeat(top[None], len(path), 0), pitch)
                    first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    core = next((c for c in CORES if all(in_frame(q, t, piece, c) for q, t in zip(path[::10], tgt[::10]))), None)
                    if core is None:                          # the piece leaves the frame somewhere
                        continue
                    par = parallax(path, tgt)
                    d0 = float(np.linalg.norm(path[0] - piece.centre))
                    k = len(path) // 2
                    depth = self.space.view_depth(path[k], math.atan2(*(tgt[k][:2] - path[k])[::-1]), HFOV)
                    score = (1.0 + 1.0 * par + 0.3 * plen + 1.5 * core - 1.5 * abs(d0 - SIT_BACK) / SIT_BACK
                             + 0.1 * min(depth, 5.0) - 0.15 * (pitch - PITCH_LADDER[0]))
                    out.append(Move("walkway", path, tgt, score, pitch, piece,
                                    why={"parallax": round(par, 2), "travel_m": round(plen, 2), "core": core,
                                         "start_from_piece_m": round(d0, 2), "pitch_cap": pitch}))
        return out

    def _glide(self, room, pts, paths, rmin, dur, prev, earlier, opening, own) -> list[Move]:
        out = []
        for a in sorted(own):
            for b in range(len(pts)):
                if a == b:
                    continue
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not MOVE_LEN[0] + 0.4 <= L <= MOVE_LEN[1] + 1.4:
                    continue
                path, plen = self._path(paths, a, b, rmin)
                if path is None or plen > 1.25 * L or plen / dur * 1.5 > MOVE_BUDGET:
                    continue
                path = resample(path, SAMPLES)
                d = path[-1] - path[0]
                d /= np.linalg.norm(d)
                head = math.atan2(d[1], d[0])
                reach = self.space.ray(path[-1], head)
                if reach < 1.0:                                  # walking into a wall
                    continue
                far = path[-1] + d * min(reach, 6.0)
                dist = float(np.linalg.norm(far - path[0]))
                aim = np.array([*far, EYE - min(0.45, 0.12 * dist)])
                tgt = cap_pitch(path, np.repeat(aim[None], len(path), 0), PITCH_LADDER[0])
                first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                depth = self.space.view_depth(path[0], head, HFOV)
                score = 0.3 * plen + 0.25 * min(depth, 6.0) + 0.3 * parallax(path, tgt) + (OPENING_GLIDE if opening else 0.0)
                out.append(Move("glide", path, tgt, score,
                                why={"travel_m": round(plen, 2), "view_depth_m": round(depth, 2)}))
        return out

    def _pan(self, room, pts, rmin, dur, prev, earlier, closing) -> list[Move]:
        sp = self.space
        out = []
        spots = sorted(pts, key=lambda p: -sp.clearance(p))[:12]
        heads = np.radians(np.arange(0, 360, 15))
        for p in spots:
            depths = [sp.view_depth(p, h, HFOV) for h in heads]
            hc = float(heads[int(np.argmax(depths))])
            # drift a little towards the view, where there is room (parallax)
            fwd = np.array([math.cos(hc), math.sin(hc)])
            e1 = p + 0.35 * fwd
            if sp.clearance(e1) < rmin or sp.room_at(e1) not in sp.spaces.get(room, {room}):
                e1 = p + 0.05 * fwd
            for sign in (1, -1):
                sweep = PAN_SWEEP
                for _ in range(12):                     # too fast for its length: sweep less
                    s = np.linspace(0, 1, SAMPLES)
                    eye = np.outer(1 - s, p) + np.outer(s, e1)
                    hs = hc + sign * math.radians(sweep) * (s - 0.5)
                    tgt = np.c_[eye[:, 0] + 2.5 * np.cos(hs), eye[:, 1] + 2.5 * np.sin(hs), np.full(SAMPLES, EYE - LOOK_DROP)]
                    turn, move = trial_speeds(eye, tgt, dur)
                    if turn <= TURN_BUDGET and move <= MOVE_BUDGET:
                        break
                    sweep *= 0.85
                else:
                    continue
                first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                depth = float(np.mean([sp.view_depth(e, h, HFOV) for e, h in zip(eye[::10], hs[::10])]))
                score = 0.25 * min(depth, 6.0) + 0.4 * sweep / PAN_SWEEP + (0.6 if closing else 0.0) - 0.3
                out.append(Move("pan", eye, tgt, score,
                                why={"sweep_deg": round(sweep), "view_depth_m": round(depth, 2),
                                     "turn_dps": round(turn, 1), "move_mps": round(move, 2)}))
        return out

    def _hold(self, least: Move, room: str) -> Move:
        """Every move runs into something near the lens: hold the least blocked start, with a breath
        of pull-back (never move into the blockage)."""
        p, t = least.first
        away = p - t[:2]
        away = away / max(1e-9, np.linalg.norm(away))
        e1 = p + 0.12 * away if self.space.clearance(p + 0.12 * away) >= RMIN_LADDER[-1] else p + 0.02 * away
        s = np.linspace(0, 1, SAMPLES)
        eye = np.outer(1 - s, p) + np.outer(s, e1)
        tgt = np.repeat(t[None], SAMPLES, 0)
        m = Move("hold", eye, tgt, least.score, least.pitch, least.piece, least.occ,
                 why={"held_because": "every move is blocked near the lens", "occlusion": round(least.occ, 3)})
        return m

    def _cover(self, m: Move) -> float:
        if self.occ is None:
            return 0.0
        k = len(m.eye) // 2
        worst = 0.0
        for e, t in ((m.eye[k], m.tgt[k]), (m.eye[0], m.tgt[0]), (m.eye[-1], m.tgt[-1])):
            worst = max(worst, self.occ.cover([*e, EYE], t, HFOV, m.piece, OCC_NEAR_KIND.get(m.kind, OCC_NEAR)))
            if worst > MAX_OCC:                       # already out: the other poses cannot bring it back
                break
        return worst
