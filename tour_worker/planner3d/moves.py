"""Camera moves for one shot of a room, generated from the room's free floor and checked against the
tour rules (user reviews 2026-11-12/13), then scored; the best one that passes wins.

Kinds of move (the green-house v3 tour is the standard, user 2026-11-13):
  walkway  a room shot: along a free path in the room (or its open-plan space), looking at the
           room's main piece of furniture: the piece's top surface stays in frame all along, seen
           three-quarter-on from a sitting-back distance; moving across the view gives parallax
  push     a room shot for a room with no piece to compose on but with its own working surfaces
  pull     (a kitchen's counter run, sink, hob): a slow move toward them (push) or back from them
           (pull - preferred: the view widens), aimed at them
  reveal   the opening shot from a room with no piece (a hall): along the hall toward the next
           room of the tour, the view turning from down the path onto that room
  pan      the closing shot: from an open spot of the last room, a slow sweep looking back through
           the home (its deepest view)
  glide    the fallback when none of the above exists: along a free path into the room's deepest view
  link     between two rooms a hallway joins (user 2026-11-13: "from dining pan onto the hallway,
           then move into the bedroom"): from the room being left, near where the way out is, a
           slow pan from that room onto the hallway, toward the next room
  enter    the next room's first shot after a link: along the walking route from the hallway
           through the door into the room, looking at its main piece
  hold     the fallback when every move runs into something near the lens: the least blocked framed
           spot with a breath of pull-back, and a warning
Never as room shots: pans (they sweep past what a room holds) or aims above eye level (a ceiling,
upper cabinets).

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
MAX_TURN_DPS, MAX_MOVE_MPS = 10.0, 0.8
# moves are planned to this share of the limits: the beat sync only lengthens room and hero shots
# (planner/lengths.SYNC_BOUNDS), but may shorten a close-up to 0.7x
PRE_SYNC_SHARE = {"room": 0.9, "hero": 0.9, "detail": 0.65}   # 0.9: the planner's 91-sample check vs per frame
PITCH_LADDER = (10.0, 13.0, 16.0)
JUMP_ANGLE, JUMP_DIST = 30.0, 1.5
RMIN_LADDER = (0.28, 0.22)    # camera body clearance, a little less in small rooms
MOVE_LEN = (0.6, 2.6)
CORES = (0.75, 0.6, 0.5, 0.4)
SIT_BACK = 2.4                # m from the piece at the start of a walkway move
OCC_NEAR, MAX_OCC, OCC_W, OCC_CHECK_KIND = 2.0, 0.20, 3.0, 30
OPENING_GLIDE = 1.5          # the tour opens by moving into the home
# near-lens distance per move: a walkway composes on a piece, so nothing may stand within 2 m in
# front of it; a glide or pan looks down a room, where walls 1.2 m+ away are the room itself
# (user 2026-11-16, Home Scan: shots closing in on walls the floor plan does not hold - a poorly
# captured wall is no "tall" cell - so the rendered check looks 1.8 m out for every room move)
OCC_NEAR_KIND = {"corner_sweep": 1.8, "panorama": 1.5, "descend": 1.8, "truck": 2.0, "rise": 1.8, "exit": 0.9, "walkway": 2.0, "arc": 2.0, "room_pan": 1.8, "hold_pan": 1.8, "push": 1.8, "pull": 1.8, "reveal": 1.8, "link": 1.2, "enter": 1.5, "glide": 1.8, "pan": 1.2, "detail": 0.7}
PAN_SWEEP = 50.0
AIM_Z_MAX = EYE - 0.10        # a room shot never looks up: not at a ceiling, not at upper cabinets
LOW_TOP = 1.4                 # m: an object whose top is below this is a surface a room shot may aim at
# what a room without a main piece is about: its working surfaces (low objects of these classes)
ANCHORS = {"kitchen": ("countertop", "sink", "stove", "oven", "dishwasher", "kitchen island", "cabinet"),
           "bathroom": ("bathtub", "sink", "shower", "countertop", "cabinet"),
           "ensuite": ("bathtub", "sink", "shower", "countertop", "cabinet"),
           "laundry": ("washer", "cabinet", "countertop", "sink"),
           "home_office": ("desk", "table"), "dining_room": ("table",), "living_room": ("sofa", "coffee table"),
           "bedroom": ("bed",), "primary_bedroom": ("bed",), "childrens_bedroom": ("bed",), "guest_bedroom": ("bed",)}
PUSH_LEN = (0.6, 2.2)
PUSH_END = (1.3, 3.2)         # m from the anchor where a push ends
REVEAL_SHARE = 0.35           # of the reveal's last view on the next room
CLOSING_DEPTH = 3.0           # m: a closing pan that sees less than this is no look-back
LINK_BEFORE = (1.2, 4.0)      # m along the route before the hallway where a link pan stands
LINK_AHEAD = 2.5              # m into the hallway a link pan ends looking at
LINK_SWING = (15.0, 50.0)     # deg a link pan turns
LINK_DRIFT = 0.4              # m it drifts toward the hallway
ENTER_BEFORE = (0.6, 1.8)     # m before the doorway an enter shot starts
ENTER_AFTER = (0.8, 2.6)      # m past it (straight-line) it ends
ENTER_SHARE = 0.4             # of its first view on the room it enters
# room to breathe (user 2026-11-13: "to keep the length of a shot it goes too near the walls - going
# back and wider is better"): with one field of view, closing in on a wall reads as a zoom
FRONT_MIN = 2.2               # m to the nearest wall or tall thing across the middle third of the frame,
FRONT_MIN_ENTER = 1.5         # at a room shot's start and end (an enter starts in a doorway)
AIM_MIN = 2.0                 # m from what a room shot aims at, at its end
OPEN_IN_W = 1.5               # the opening shot: score per m it closes in on its subject
WIDE_W = 0.6                  # score per m the view opens up from start to end (pulling back)
# coverage (user 2026-11-16: "each room should get up to 3 shots so all its walls get covered"):
# a room's further shots must each show this share of its walls the earlier ones did not, and the
# more new wall a candidate shows the better it scores
COVER_NEW_MIN = 0.15
COVER_W = 3.0
FRAME_MARGIN = 0.03          # the subject's box this far inside the frame's edges ...
FLOOR_MARGIN = 0.04          # ... and the floor under it this far above the frame's bottom (a strip of floor)
MIN_SCALE_PS = 0.035          # or a push's subject grows (ln of its distance ratio) this much a second
MIN_TRAVEL_PS = 0.12         # m/s: a room shot travels at least this much for its length ...
MIN_TURN_PS = 2.5            # ... or turns this much (deg/s) - else it reads as a still (user 2026-11-16, Krk)
TILT_MAX = 22.0              # deg: a push may tilt down this far at its end to frame its subject with floor
KEEP_PER_KIND = 40            # a room's arcs / pans judged further: the best by their own score
OWN_POINTS = 16              # candidate points spread over a room's own floor, besides the space's
DOOR_NEAR = 0.4             # m: a room shot never starts or ends this close to a doorway
WALL_CLEAR = 0.2             # m: nor pressed against a wall
DOOR_CLEAR = 0.45            # m: a room shot never starts or ends in a doorway (clearance under this)
EDGE_NEAR, EDGE_MAX = 0.8, 0.12   # anything within EDGE_NEAR m covering EDGE_MAX of the whole frame blocks it
VOID_MAX = 0.12              # at most this share of a view may be what the scan never captured
CEIL_PITCH = 6.0             # deg the aim drops when the scan has no ceiling (its black band stays out)
ANCHOR_HALF = 0.8           # m: the working area a push / pull must show whole
ARC_R = (2.3, 2.7, 3.1, 3.5)
ARC_SWEEP = (30.0, 40.0, 50.0, 65.0)     # an orbit: wider sweeps where the speed limit allows
HOLD_SWEEP = 25.0
NO_AIM = ("door", "curtain", "windowpane", "blind")   # never what a room shot ends on
REVEAL_SWING = 40.0           # deg: the most a reveal turns (from down its path onto the next room)
SAMPLES = 50
# scene-picked moves and the flow between rooms (user 2026-11-16: "a multitude of camera moves
# selected according to the scenes ... use the floor plan by moving toward the next room, more a
# drone shot vibe"; decided: a rise may reach RISE_Z with RISE_PITCH down, only in open or large
# rooms; doorways are cuts with matched motion)
TRUCK_LEN = (1.2, 1.7, 2.2)   # m a truck slides sideways ...
TRUCK_DIST = (2.3, 2.8, 3.3)  # ... this far out from its subject ...
TRUCK_AIM = 0.4               # ... its aim sliding this share of the way with it (parallax)
RISE_Z = 2.1                  # m: the top of a rise
RISE_PITCH = 20.0             # deg down at the top of a rise
RISE_AREA = 16.0              # m2: a room this big (or open plan) may rise
RISE_HEADROOM = 0.35          # m under the ceiling a rise stops
RISE_DRIFT = (0.3, 0.7)       # m it drifts toward what it shows
RISE_DIST = (2.6, 6.0)        # m to what it shows
EXIT_STOP = 1.0               # m before the doorway an exit ends (door frames stay off the lens)
EXIT_LEN = (1.2, 3.0)         # m an exit travels along the route
EXIT_AHEAD = 2.5              # m ahead along the route it looks (where it is going)
AFTER_EXIT = (0.5, 1.2)       # m into the next room its walk-in starts after an exit
PRIOR_W = 0.6                 # score of a kind the room's profile asks for
ROOM_REPEAT_W = 0.5           # score lost repeating a kind the room already had
# the photo tours' sequence rules (planner/cinematic.py), on this planner's score scale (x0.4)
# (user 2026-11-17: "a lot of camera rules from the 2d implementation ... would be true to this")
SAME_KIND_W = 1.0             # the previous shot's kind again            (2D: -2.5)
TWO_BACK_W = 0.4              # the kind of two shots back                (2D: -1.0)
SAME_SIDE_W = 0.8             # sideways the same way as the previous shot (2D: -2.0, sides alternate)
USED_DIR_W = 0.4              # per earlier use of the kind, same way      (2D: -1.0)
OVERUSE_W = 0.6               # per use of a kind past its second          (2D: -1.5)
DOLLY_KINDS = ("push", "pull", "rise", "descend", "enter")   # the moves led by moving in / out (2D: zoom-led)
DOLLY_CAP_W = 2.0             # past a third of the tour's shots          (2D: -5.0)
DOLLY_RUN_W = 0.8             # two in a row                              (2D: -2.0)
HERO_TRAVEL_W = 0.2           # a hero shot that travels                  (2D: +0.5)
TRAVELS = ("walkway", "truck", "arc", "rise", "descend", "corner_sweep", "exit", "enter", "push", "pull", "glide")
MIN_KIND_S = {"room_pan": 3.8, "pan": 3.8, "panorama": 4.5, "truck": 3.5, "arc": 3.5, "rise": 2.8, "descend": 2.8,
              "corner_sweep": 3.5}
# the move library mixes (user 2026-11-17: "we need like 10 15 camera moves we mix and match. not all
# camera moves need to be in every tour"): no kind more than KIND_CAP times a tour (rise / descend
# once or twice), nor more than KIND_SHARE of its shots - the flight between rooms is exempt
KIND_CAP = {"rise": 2, "descend": 1, "panorama": 2, "corner_sweep": 2, "truck": 2, "arc": 2, "walkway": 3,
            "room_pan": 1, "push": 2, "pull": 2, "glide": 2, "hold_pan": 1}
KIND_SHARE = 0.25
CAP_W = 3.0                   # past its cap
FLIGHT = ("exit", "enter", "link", "reveal")      # the moves between rooms (not capped)
SWEEP_LEN = (1.0, 1.4, 1.8)   # m a corner sweep travels toward the room's middle ...
SWEEP_TURN = (25.0, 35.0)     # ... turning this much across it
PANO_SWEEP = (45.0, 80.0)     # deg a panorama turns (the speed limit decides within)   # shorter: a simpler move
SINK_M = 0.3                  # a view may not drop this much past what it shows (the interest above it)
# what a room's first shot (the one its label plays on) must show: what makes it that room (user
# 2026-11-16: the Home Scan's kitchen opened on its dining table - "the camera shows the room its
# mentioning in the frame")
TYPE_SIGNS = {"kitchen": ("stove", "oven", "sink", "refrigerator", "dishwasher", "kitchen island", "countertop",
                          "hood", "microwave"),
              "bedroom": ("bed",), "primary_bedroom": ("bed",), "guest_bedroom": ("bed",),
              "bathroom": ("toilet", "bathtub", "shower", "sink"),
              "living_room": ("sofa", "television receiver", "fireplace"), "family_room": ("sofa", "television receiver", "fireplace"),
              "dining_room": ("table",), "home_office": ("desk", "computer", "monitor", "table")}
TYPE_WHOLE = ("bed", "sofa")    # ... framed whole, when the room has one
TOP_VOID_MAX = 0.15           # at most this share of the frame's top fifth may be uncaptured (a missing ceiling; 0.3 let the Home Scan kitchen band through)
TOP_TILT = (3.0, 6.0)         # deg a shot tilts down to clear it, before it is dropped
BAND_MARGIN = 0.05            # m: the frame's top edge stays this far under where the scan's walls end
BAND_TILT_MAX = 6.0           # deg a shot may tilt down to keep it there (no ceiling), else it goes
WHOLE_GAP = 0.4               # m: labelled pieces of one bed / sofa this close are one piece
FRAME_W, FRAME_H = 1920, 1080
# what a view is worth showing (scene labels): a fireplace or a window beats a door
FEATURES = {"fireplace": 1.0, "windowpane": 0.8, "kitchen island": 0.8, "bathtub": 0.8, "bed": 0.8,
            "stairs": 0.7, "stairway": 0.7, "countertop": 0.6, "stove": 0.6, "bookcase": 0.6, "chandelier": 0.6,
            "sofa": 0.6, "television receiver": 0.5, "painting": 0.5, "table": 0.5, "desk": 0.5, "mirror": 0.4,
            "curtain": 0.3, "door": 0.2, "sink": 0.5, "hood": 0.5, "oven": 0.4, "refrigerator": 0.3,
            "cabinet": 0.15}
# of a view's middle on the shot's own room: under this the shot loses score (OWN_W)
OWN_SHARE = {"corner_sweep": 0.5, "panorama": 0.5, "descend": 0.35, "truck": 0.5, "rise": 0.35, "walkway": 0.35, "arc": 0.35, "room_pan": 0.5, "push": 0.5, "pull": 0.5, "glide": 0.5}
RELAX = 0.8                   # the second try: room to breathe x this ...
OWN_MIN_RELAXED = 0.35        # ... and this much of its own room in view
OWN_MIN = 0.5                 # a room shot's view is at least half its own room
OWN_FREE = ("enter", "reveal", "link", "pan", "detail", "exit")   # they look in from outside it, or back over the home
COVER_OWN_MIN = 0.6           # a room's further shot shows mostly that room (not the hallway beyond)
COMPOSED = ("walkway", "arc", "room_pan", "push", "pull", "truck", "rise", "corner_sweep", "panorama", "descend")    # the kinds a room's further shots may be
OWN_W = 4.0                                                # score lost per missing share
FEATURE_W = 0.4
# close-ups (detail shots, 1.5-2.5 s): fixtures, finishes, decor
DETAILS = {"chandelier": 1.0, "fireplace": 1.0, "vase": 0.9, "flower": 0.9, "sculpture": 0.9, "painting": 0.8,
           "stove": 0.8, "sink": 0.7, "lamp": 0.7, "sconce": 0.7, "clock": 0.6, "plant": 0.6, "mirror": 0.6,
           "hood": 0.6, "bathtub": 0.6, "television receiver": 0.3}
DETAIL_SIZE = (0.12, 1.6)     # m across: a vase to a fireplace
DETAIL_DIST = (0.9, 1.15, 1.4)
DETAIL_PUSH = 0.25
DETAIL_PITCH_MAX = 30.0
DETAIL_NOT_PREV = 0.9       # m: a close-up's object this near the previous shot's last aim is what it already showed


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
    eye_z: float = EYE           # detail moves sit lower / higher than the room eye height
    subject: np.ndarray | None = None   # (8, 3) box of what the shot ends on (planner coords), floor to top
    z: np.ndarray | None = None  # (n,) eye height along the move (a rise); None: eye_z throughout
    easing: str = "ease_in_out"  # a fly-out speeds up into the cut (ease_in), the fly-in settles (ease_out)

    @property
    def zs(self) -> np.ndarray:
        return self.z if self.z is not None else np.full(len(self.eye), self.eye_z)

    @property
    def eye3(self) -> np.ndarray:
        return np.c_[self.eye, self.zs]

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


def trial_speeds(eye2: np.ndarray, tgt: np.ndarray, dur: float, z=EYE, easing: str = "ease_in_out") -> tuple[float, float]:
    segs, _ = fit_pair(np.c_[eye2, np.broadcast_to(np.asarray(z, float), (len(eye2),))], tgt)
    return speeds_of([{"eye": e, "target": t} for e, t in segs], dur, easing=easing, samples=91)


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


def piece_box(fp) -> np.ndarray:
    """A piece's box: its footprint at the floor and at its top (8, 3)."""
    c = []
    for z in (0.0, float(fp.top)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                c.append([*(fp.centre + sx * fp.half_long * fp.u + sy * fp.half_short * fp.v), z])
    return np.array(c, float)


def objects_box(objs: list[dict]) -> np.ndarray | None:
    """The box around labelled objects, down to the floor (8, 3)."""
    if not objs:
        return None
    lo = np.min([o["min"] for o in objs], 0)
    hi = np.max([o["max"] for o in objs], 0)
    return np.array([[x, y, z] for z in (0.0, float(hi[2])) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])], float)


def frames_fully(eye2, target, box: np.ndarray, eye_z: float = EYE) -> bool:
    """The whole subject in frame, with room around it: its box inside the frame by FRAME_MARGIN on
    every side, and the floor under it above the frame's bottom by FLOOR_MARGIN (user 2026-11-16:
    "when we zoom onto an object to end the scene we can see it fully - a sofa relative to the floor,
    a painting or fireplace with enough wall and a bit of floor, so we know what's where")."""
    V, K = look_at([eye2[0], eye2[1], eye_z], target), intrinsics(HFOV, FRAME_W, FRAME_H)
    cam = box @ V[:3, :3].T + V[:3, 3]
    if (cam[:, 2] < 0.3).any():
        return False
    px = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    py = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    mx, my = FRAME_MARGIN * FRAME_W, FRAME_MARGIN * FRAME_H
    if px.min() < mx or px.max() > FRAME_W - mx or py.min() < my:
        return False
    floor = py[box[:, 2] <= 1e-6]                         # the corners on the floor
    return bool(floor.max() <= (1.0 - FLOOR_MARGIN) * FRAME_H)


def _keep_best(moves: list, n: int = None) -> list:
    """The KEEP_PER_KIND best of a kind's candidates by their own score, before the rendered and
    framing checks (a room's pans or arcs run to hundreds - 659 in the Home Scan's living room,
    planning 6x slower, 2026-11-16)."""
    n = KEEP_PER_KIND if n is None else n
    return sorted(moves, key=lambda m: -m.score)[:n]


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

    def best(self, room: str, dur: float, **kw) -> tuple[Move | None, list[str]]:
        """_best; a room with no view under the full rules gets a second try with the softest ones
        eased (room to breathe x RELAX, own-room share OWN_MIN_RELAXED, framing not required) - a
        room skipped is worse than a room shown a little closer (v10: the green house's dining
        room, study and hall had no shot)."""
        mv, w = self._best(room, dur, **kw)
        if mv is None and kw.get("covered") is None and kw.get("kind", "room") != "detail":
            self.relaxed = True
            try:
                mv, w2 = self._best(room, dur, **kw)
            finally:
                self.relaxed = False
            if mv is not None:
                mv.why["relaxed"] = True
                return mv, [f"room_rules_relaxed:{room}"] + [x for x in w2 if not x.startswith("room_no")]
        return mv, w

    def _best(self, room: str, dur: float, *, earlier=(), prev=None, piece_rank: int = 0,
             opening: bool = False, closing: bool = False, kind: str = "room",
             next_room: str | None = None, enter_route=None, after_exit: bool = False,
             final: bool = False, follows_link: bool = False, covered: np.ndarray | None = None) -> tuple[Move | None, list[str]]:
        """The best move for a shot of `room` lasting dur s. earlier: (first, last) poses of the
        room's earlier shots; prev: the previous shot's last pose; next_room: the tour's next room
        (an opening reveal turns onto it). Returns (move, warnings).

        Which move, in this order (the first whose candidates pass the speed, own-room and lens
        checks wins; each kind at the roomier camera clearance first):
          close-up          a detail shot
          look-back pan     the closing shot, when the last room sees back into the home (straight
                            after a link, the enter comes first)
          enter             a room's first shot when the route to it runs through a hallway: in
                            along the route from the hallway
          reveal            the opening shot of a room with no main piece, toward the next room
          walkway           the room's main piece (a second angle: its next piece, or the same one
                            from elsewhere - never a framing an earlier shot had)
          push              no piece: toward the room's own working surfaces (ANCHORS)
          glide / pan       nothing else exists"""
        sp = self.space
        type_warn: list[str] = []
        share = PRE_SYNC_SHARE.get(kind, 0.8)
        self.turn_budget, self.move_budget = MAX_TURN_DPS * share, MAX_MOVE_MPS * share
        pieces = sp.main_pieces(room)
        piece = pieces[piece_rank % len(pieces)] if pieces else None
        anchor = self.anchor(room)
        # straight after a link the tour walks in, even into its last room; otherwise the closing
        # look-back comes first
        order = (("detail", "enter", "closing", "reveal", "walkway", "push") if follows_link or after_exit
                 else ("detail", "closing", "enter", "reveal", "walkway", "push"))
        found: dict[str, list[list[Move]]] = {k: [] for k in order}
        for rmin in RMIN_LADDER:                       # each kind at the roomier clearance first
            mask = sp.space_mask(room, rmin)
            pts = self._points(room, mask)
            if len(pts) < 2:
                continue
            paths = sp.paths_between(pts, mask)
            # a shot starts in its own room (it may then move on through an open-plan space)
            own = {k for k, p in enumerate(pts) if sp.room_at(p) == room}
            if not own:
                continue
            if kind == "detail":                      # a close-up of one of the room's details
                found["detail"].append(self._detail(room, rmin, dur, prev, earlier))
                continue
            if enter_route is not None:
                found["enter"].append(self._enter(room, piece, enter_route, rmin, dur, prev, earlier, after_exit))
            if opening and piece is None and next_room is not None:
                found["reveal"].append(self._reveal(room, next_room, pts, paths, rmin, dur, prev, own))
            if closing:
                found["closing"].append([m for m in self._pan(room, [pts[k] for k in sorted(own)], rmin, dur, prev,
                                                              earlier, True) if m.why["view_depth_m"] >= CLOSING_DEPTH])
            # the composed moves compete on score: walkways on the piece, arcs round it, pans across
            # the room's own things (user 2026-11-16: small rooms "almost no pan")
            composed: list[Move] = []
            if piece is not None:
                for pitch in PITCH_LADDER:
                    walk = self._walkway(room, piece, pts, paths, rmin, dur, pitch, prev, earlier, own)
                    if walk:
                        composed += walk
                        break
                composed += self._arc(room, piece, rmin, dur, prev, earlier)
            composed += self._room_pan(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._truck(room, rmin, dur, prev, earlier)
            composed += self._rise(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._descend(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._corner_sweep(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            composed += self._panorama(room, [pts[k] for k in sorted(own)], rmin, dur, prev, earlier)
            if composed:
                found["walkway"].append(composed)
            if anchor is not None:
                found["push"].append(self._push(room, anchor, pts, paths, rmin, dur, prev, earlier, own))
        tiers: list[list[Move]] = [t for k in order for t in found[k]]
        # nothing to compose on, or every composed move blocked: glides into the deepest views (a pan
        # is never a room shot - it sweeps past what a room holds; only the closing look-back pans)
        glides: list[Move] = []
        mask = sp.space_mask(room, RMIN_LADDER[-1])
        pts = self._points(room, mask)
        if len(pts) >= 2 and kind != "detail":
            own = {k for k, p in enumerate(pts) if sp.room_at(p) == room}
            if own:
                paths = sp.paths_between(pts, mask)
                glides = self._glide(room, pts, paths, RMIN_LADDER[-1], dur, prev, earlier, opening, own, closing)
        # room to breathe first: composed moves, then (only when none has it) the closest composed
        # calls, then glides the same way. A room's second angle is a composed move with room to
        # breathe, or none: without one the room's first shot keeps the time.
        # the aims as they will play, before anything is judged: never up, and lower when the scan
        # has no ceiling (its black band stays out of the frame)
        drop = math.tan(math.radians(CEIL_PITCH)) if "scene_ceiling_missing" in getattr(sp, "warnings", []) else 0.0
        for m in {id(m): m for t in tiers + [glides] for m in t}.values():
            if m.kind != "detail":
                if drop:
                    m.tgt[:, 2] -= drop * np.linalg.norm(m.tgt[:, :2] - m.eye, axis=1)
                m.tgt[:, 2] = np.minimum(m.tgt[:, 2], AIM_Z_MAX)
        strict = piece_rank > 0 and kind != "detail"
        roomy = lambda t: [m for m in t if self._roomy(m) and self._good(m, dur)]
        if covered is not None and kind != "detail":
            # a further shot of the room: it must show walls the earlier ones did not (glides
            # included - they face away from the piece the first shot framed), with room to breathe
            # (user 2026-11-16: glides and doorway views as further shots made "close up walls, in
            # hallways without getting into the room") - only composed moves, of this room
            def fresh(t):
                out = []
                for m in t:
                    if m.kind not in COMPOSED:
                        continue
                    k = len(m.eye) // 2
                    own_ = sp.own_share(room, m.eye[k], math.atan2(*(m.tgt[k][:2] - m.eye[k])[::-1]), HFOV)
                    if own_ < COVER_OWN_MIN:
                        continue
                    new = self.new_walls(room, m, covered)
                    if new >= COVER_NEW_MIN:
                        m.why["walls_new"] = round(new, 2)
                        m.score += COVER_W * new
                        out.append(m)
                return out
            tiers = [fresh(roomy(t)) for t in tiers]
        else:
            # every room shot keeps room to breathe and shows mostly its own room, or the room is not
            # filmed (user 2026-11-16: "close up walls" - the closest-call fallback is gone)
            def own_room(t):
                out = []
                for m in t:
                    if m.kind in OWN_FREE:
                        out.append(m)
                        continue
                    k = len(m.eye) // 2
                    o = sp.own_share(room, m.eye[k], math.atan2(*(m.tgt[k][:2] - m.eye[k])[::-1]), HFOV)
                    m.why["own_share"] = round(o, 2)
                    if o >= (OWN_MIN_RELAXED if getattr(self, "relaxed", False) else OWN_MIN):
                        out.append(m)
                return out
            tiers = ([own_room(roomy(t)) for t in tiers] if strict
                     else [own_room(roomy(t)) for t in tiers] + [own_room(roomy(glides))])
        # the room's first shot shows what makes it that room (TYPE_SIGNS), when anything does
        if piece_rank == 0 and covered is None and kind != "detail" and self.type_signs(room):
            shown = [[m for m in t if m.kind in ("link",) or self.shows_type(room, m)] for t in tiers]
            # a bedroom / living room opens on its bed / sofa whole (user 2026-11-16: the primary
            # bedroom opened on a mirror and a corner of the bed); else on what shows its kind
            if any(o["class"] in TYPE_WHOLE for o in self.type_signs(room)):
                whole = [[m for m in t if m.kind in ("link",) or self.shows_type(room, m, whole=True)] for t in shown]
                if any(whole):
                    shown = whole
            if any(shown):
                tiers = shown
            else:
                type_warn = [f"room_type_not_in_view:{room}"]
        # what the room asks for (its profile) and variety (never the same kind twice running)
        prof = self.profile(room)
        used = self.__dict__.setdefault("_kinds", {}).get(room, [])
        # moves that cannot play in the time (a pan under 3.8 s reads rushed), that sink past what
        # they show, or (the tour's last room) close by pushing in: the 2D tours' rules
        tiers = [[m for m in t if dur >= MIN_KIND_S.get(m.kind, 0.0) and not self.sinks(m)
                  and not (final and m.kind in ("push", "exit"))] for t in tiers]
        for m in {id(m): m for t in tiers for m in t}.values():
            pr = PRIOR_W * prof["prior"].get(m.kind, 0.0)
            vr = self.sequence_cost(m) + (ROOM_REPEAT_W if m.kind in used else 0.0)
            if kind == "hero" and m.kind in TRAVELS:
                pr += HERO_TRAVEL_W
            if pr or vr:
                m.score += pr - vr
                m.why["prior"] = round(pr - vr, 2)
        # one of each kind a room: a room's later shots differ in kind (user 2026-11-16: a tour of
        # pans) - a kind again only when nothing else will do
        used = self.__dict__.setdefault("_kinds", {}).get(room, [])
        if used:                                   # (any kind: a room is not two trucks either)
            again = [m for t in tiers for m in t if m.kind in used]
            tiers = [[m for m in t if m.kind not in used] for t in tiers] + [again]
        # the tour's caps are hard (v10: three rises past a cap of two): a kind at its cap only when
        # nothing else will do
        capped = [m for t in tiers for m in t if self.at_cap(m.kind)]
        if capped:
            tiers = [[m for m in t if not self.at_cap(m.kind)] for t in tiers] + [capped]
        tiers = [t for t in tiers if t]
        if not tiers:
            return None, [f"room_no_move:{room}" if kind != "detail" else f"room_no_detail:{room}"]
        for m in {id(m): m for t in tiers for m in t}.values():
            if m.kind == "detail" or "front_m" not in m.why:
                continue
            if opening:
                # the tour's first shot moves into the home, toward what it shows (user 2026-11-16:
                # "in the first shot we should pan in instead of zoom out")
                d0 = float(np.linalg.norm(m.tgt[0][:2] - m.eye[0]))
                d1 = float(np.linalg.norm(m.tgt[-1][:2] - m.eye[-1]))
                m.score += OPEN_IN_W * float(np.clip(d0 - d1, -1.0, 1.5))
                m.why["moves_in_m"] = round(d0 - d1, 2)
            else:                                                     # pulling back, the view widening
                f0, f1 = m.why["front_m"]
                m.score += WIDE_W * float(np.clip(f1 - f0, -1.0, 1.5))
        if self.occ is not None:                      # occlusion renders work on this room's splats
            mask = sp.space_mask(room, RMIN_LADDER[-1])
            ys, xs = np.nonzero(mask)
            cells = np.c_[sp.lo[0] + (xs + .5) * 0.05, sp.lo[1] + (ys + .5) * 0.05]
            c = cells.mean(0)
            self.occ.focus(c, float(np.linalg.norm(cells - c, axis=1).max()) + OCC_NEAR + 0.5)
        least, seen = None, set()
        for cands in tiers:
            for m in cands:
                if id(m) in seen:
                    continue
                seen.add(id(m))
                if m.kind != "detail":
                    b = self._feature_bonus(room, m)                      # what the view shows (labels)
                    m.score += b
                    if b:
                        m.why["features"] = round(b / FEATURE_W, 2)
            best, low = self._check(room, cands, dur)
            if low is not None and (least is None or low.occ < least.occ):
                least = low
            if best is not None:
                best.why.update(candidates=len(cands), occlusion=round(best.occ, 3))
                self._kinds.setdefault(room, []).append(best.kind)
                self.record(best)
                return best, type_warn
        if least is None:
            return None, [f"room_no_move:{room}"]
        if covered is not None:                   # a further shot is a good one or none (user 2026-11-16:
            return None, [f"coverage_no_move:{room}"]     # a held view ended on a blur at the lens)
        hold = self._hold(least, room)
        self.record(hold)
        return hold, [f"view_partly_blocked:{room}"] + type_warn

    def front(self, e, t) -> float:
        """Metres to the nearest wall or tall thing across the middle third of the view from e to t."""
        h = math.atan2(t[1] - e[1], t[0] - e[0])
        return float(self.space.rays(e, h + np.radians(np.linspace(-HFOV / 6, HFOV / 6, 7)), max_m=6.0).min())

    def _roomy(self, m: Move) -> bool:
        """Room to breathe at both ends (FRONT_MIN, an enter's start FRONT_MIN_ENTER) and the end at
        least AIM_MIN from what it aims at. Close-ups and the hallway links are exempt."""
        if m.kind in ("detail", "link"):
            return True
        if "front_m" not in m.why:
            m.why["front_m"] = (round(self.front(m.eye[0], m.tgt[0]), 2), round(self.front(m.eye[-1], m.tgt[-1]), 2))
        f0, f1 = m.why["front_m"]
        aim = float(np.linalg.norm(m.tgt[-1][:2] - m.eye[-1]))
        k = RELAX if getattr(self, "relaxed", False) else 1.0
        return f0 >= k * (FRONT_MIN_ENTER if m.kind == "enter" else FRONT_MIN) and f1 >= k * FRONT_MIN and aim >= k * AIM_MIN

    def _good(self, m: Move, dur: float) -> bool:
        """A room shot is seen to move, never starts or ends in a doorway (a walk-in may start in
        one), and ends with its subject whole and floor under it."""
        if m.kind in ("detail", "link"):
            return True
        sp = self.space
        ends = (m.eye[-1],) if m.kind == "enter" else (m.eye[0], m.eye[-1])
        if any(sp.clearance(q) < WALL_CLEAR or self.in_doorway(q) for q in ends):
            return False
        return self.moves_enough(m, dur) and (self.framed(m) or getattr(self, "relaxed", False))

    def _points(self, room: str, mask: np.ndarray) -> np.ndarray:
        """Where shots may start and end: spread over the whole space a shot may move through, and
        again over the room's own floor - spread over a whole open house, a small room's share of
        the points lands in its corners and doorways (none where a camera would stand)."""
        sp = self.space
        pts = sp.spread_points(mask, 36)
        own = sp.spread_points(mask & (sp.room_of == sp.room_keys.index(room)), OWN_POINTS, min_gap=0.4)
        if not len(own):
            return pts
        keep = [p for p in own if not len(pts) or np.linalg.norm(pts - p, axis=1).min() > 0.3]
        return np.concatenate([pts, np.array(keep).reshape(-1, 2)]) if keep else pts

    def in_doorway(self, q) -> bool:
        """q within DOOR_NEAR of a doorway: where two rooms meet through a gap narrower than
        DOOR_CLEAR x 2 (an open-plan seam is not a doorway)."""
        sp = self.space
        if "_doors" not in self.__dict__:
            from scipy import ndimage as ndi
            lab, dt = sp.room_of, sp.dt
            seam = np.zeros(lab.shape, bool)
            for ax in (0, 1):
                a, b = np.moveaxis(lab, ax, 0)[:-1], np.moveaxis(lab, ax, 0)[1:]
                m = (a >= 0) & (b >= 0) & (a != b)
                s0, s1 = np.moveaxis(seam, ax, 0)[:-1], np.moveaxis(seam, ax, 0)[1:]
                s0 |= m
                s1 |= m
            seam &= dt < DOOR_CLEAR + 0.15
            from tour_worker.planner3d.space import CELL
            self._doors = ndi.distance_transform_edt(~seam) * CELL <= DOOR_NEAR if seam.any() else seam
        i, j = sp.cell(q)
        return bool(0 <= i < self._doors.shape[0] and 0 <= j < self._doors.shape[1] and self._doors[i, j])

    def walls_seen(self, room: str, m: Move) -> np.ndarray:
        """Flat indices of the room's wall cells the move shows (five poses along it, a cell's
        tolerance for rounding)."""
        sp = self.space
        walls = sp.wall_cells(room)
        if not len(walls):
            return walls
        hit = []
        for k in np.linspace(0, len(m.eye) - 1, 5).astype(int):
            e, t = m.eye[k], m.tgt[k]
            hit.append(sp.seen_cells(e, math.atan2(t[1] - e[1], t[0] - e[0]), HFOV))
        grid = np.zeros(sp.shape, bool)
        grid.flat[np.concatenate(hit)] = True
        grid = ndi.binary_dilation(grid, iterations=1)
        return walls[grid.flat[walls]]

    def new_walls(self, room: str, m: Move, covered: np.ndarray) -> float:
        """Share of the room's walls the move shows that `covered` (flat indices) does not."""
        walls = self.space.wall_cells(room)
        if not len(walls):
            return 0.0
        seen = self.walls_seen(room, m)
        return float(len(np.setdiff1d(seen, covered, assume_unique=False))) / len(walls)

    def _check(self, room: str, cands: list[Move], dur: float) -> tuple[Move | None, Move | None]:
        """(the best candidate within the speed limits, its own room share and the lens rule;
        the least blocked one within the speed limits) - best first, lazily, stopping once no later
        candidate can win."""
        sp = self.space
        cands = sorted(cands, key=lambda m: -m.score)
        best, least, checked, passed = None, None, {}, []
        for m in cands:
            if best is not None and m.score <= best.score - OCC_W * best.occ:
                break
            if checked.get(m.kind, 0) >= OCC_CHECK_KIND:      # every kind gets its share of checks
                continue
            if m.kind in OWN_SHARE and "own_share" not in m.why:    # it films its own room
                k = len(m.eye) // 2
                share = sp.own_share(room, m.eye[k], math.atan2(*(m.tgt[k][:2] - m.eye[k])[::-1]), HFOV)
                m.why["own_share"] = round(share, 2)
                # a view mostly of another room loses score (a small room in an open plan still
                # gets its best view: there is no view of it without its neighbour)
                m.score -= OWN_W * max(0.0, OWN_SHARE[m.kind] - share)
                if best is not None and m.score <= best.score - OCC_W * best.occ:
                    continue
            if "turn_dps" not in m.why:
                turn, move = trial_speeds(m.eye, m.tgt, dur, m.zs, m.easing)
                m.why.update(turn_dps=round(turn, 1), move_mps=round(move, 2))
            if m.why["turn_dps"] > self.turn_budget or m.why["move_mps"] > self.move_budget:
                continue
            checked[m.kind] = checked.get(m.kind, 0) + 1
            m.occ = self._cover(m)
            if least is None or m.occ < least.occ:
                least = m
            if m.occ > MAX_OCC:
                continue
            passed.append(m)
            if best is None or m.score - OCC_W * m.occ > best.score - OCC_W * best.occ:
                best = m
        # what of its view the scan never captured (rendered, so only for the winners): best first,
        # down the ranking until one passes
        if self.occ is not None and best is not None:
            best = None
            for m in sorted(passed, key=lambda m: -(m.score - OCC_W * m.occ)):
                if m.kind != "detail":
                    v = self._void(m)
                    m.why["void"] = round(v, 3)
                    if v > VOID_MAX:
                        continue
                best = m
                break
        return best, least

    # ---- what a view shows ----------------------------------------------------------------------
    def features_seen(self, room: str, eye2, tgt) -> float:
        """Sum of FEATURES weights of the room's (and its open plan's) labelled features in the
        middle 70% of the frame, in front within 8 m."""
        sp = self.space
        rooms = sp.spaces.get(room, {room})
        d = np.asarray(tgt[:2]) - np.asarray(eye2[:2])
        head = math.atan2(d[1], d[0])
        half = math.radians(HFOV) * 0.35
        w = 0.0
        for o in sp.objects:
            if o.get("room") not in rooms or o["class"] not in FEATURES:
                continue
            v = np.asarray(o["centre"][:2]) - np.asarray(eye2[:2])
            dist = float(np.linalg.norm(v))
            if not 0.5 < dist < 8.0:
                continue
            a = (math.atan2(v[1], v[0]) - head + math.pi) % (2 * math.pi) - math.pi
            if abs(a) <= half:
                w += FEATURES[o["class"]]
        return min(w, 2.0)

    def _feature_bonus(self, room: str, m: Move) -> float:
        k = len(m.eye) // 2
        return FEATURE_W * max(self.features_seen(room, m.eye[0], m.tgt[0]), self.features_seen(room, m.eye[k], m.tgt[k]))

    def _detail(self, room, rmin, dur, prev, earlier) -> list[Move]:
        """Close-ups of the room's details (scene labels): from DETAIL_DIST m off, the eye a little
        above the object, a slow DETAIL_PUSH m push-in; the object well inside the frame."""
        sp = self.space
        out = []
        objs = [o for o in sp.room_objects(room, DETAILS)
                if DETAIL_SIZE[0] <= max(o["max"][0] - o["min"][0], o["max"][1] - o["min"][1]) <= DETAIL_SIZE[1]]
        if prev is not None:                          # not what the previous shot ended on
            pe, pt = prev
            objs = [o for o in objs if np.linalg.norm(np.asarray(o["centre"][:2]) - np.asarray(pt[:2])) > DETAIL_NOT_PREV]
        for o in objs:
            c = np.asarray(o["centre"], float)
            size = max(o["max"][0] - o["min"][0], o["max"][1] - o["min"][1], o["max"][2] - o["min"][2])
            z_eye = float(np.clip(c[2] + 0.35, 0.9, 1.6))
            for dist in DETAIL_DIST:
                for ang in range(0, 360, 30):
                    a = math.radians(ang)
                    p0 = c[:2] + (dist + DETAIL_PUSH) * np.array([math.cos(a), math.sin(a)])
                    p1 = c[:2] + dist * np.array([math.cos(a), math.sin(a)])
                    if sp.clearance(p0) < 0.2 or sp.clearance(p1) < 0.2 or sp.room_at(p0) not in sp.spaces.get(room, {room}):
                        continue
                    pitch = math.degrees(math.atan2(z_eye - c[2], dist))
                    if pitch > DETAIL_PITCH_MAX:
                        continue
                    s_ = np.linspace(0, 1, SAMPLES)
                    eye = np.outer(1 - s_, p0) + np.outer(s_, p1)
                    tgt = np.repeat(c[None], SAMPLES, 0)
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    fill = size / (2 * dist * math.tan(math.radians(HFOV) / 2))
                    if not 0.15 <= fill <= 0.8:
                        continue
                    score = 1.0 * DETAILS[o["class"]] - 1.5 * abs(fill - 0.4) + 0.3 * min(sp.clearance(p0), 0.6)
                    out.append(Move("detail", eye, tgt, score, pitch, None, eye_z=z_eye,
                                    why={"object": o["class"], "distance_m": dist, "fill": round(fill, 2)}))
        return out

    # ---- what a room without a main piece is about --------------------------------------------------
    def anchor(self, room: str) -> np.ndarray | None:
        """The room's working surfaces (ANCHORS by room type: a kitchen's counter run, sink, hob),
        as one aim point: their area-weighted centre at their top. Only low objects of the room
        itself (never upper cabinets, never a neighbour's); None when it has none."""
        from tour_worker.scoring.rooms import canon
        classes = ANCHORS.get(canon(self.space.room_types.get(room)) or "", ())
        objs = [o for o in self.space.room_objects(room, classes)
                if 0.3 <= o["max"][2] <= LOW_TOP and max(o["max"][0] - o["min"][0], o["max"][1] - o["min"][1]) >= 0.2]
        if not objs:
            return None
        w = np.array([(o["max"][0] - o["min"][0]) * (o["max"][1] - o["min"][1]) for o in objs]) + 1e-6
        c = np.array([o["centre"][:2] for o in objs])
        top = np.array([o["max"][2] for o in objs])
        a = np.array([*(w[:, None] * c).sum(0) / w.sum(), float((w * top).sum() / w.sum())])
        # what a push / pull ends on: the working area round the anchor (a counter run along a wall
        # is the room's structure - its middle ANCHOR_HALF m each way, down to the floor)
        box = objects_box(objs)
        if box is not None:
            box[:, 0] = np.clip(box[:, 0], a[0] - ANCHOR_HALF, a[0] + ANCHOR_HALF)
            box[:, 1] = np.clip(box[:, 1], a[1] - ANCHOR_HALF, a[1] + ANCHOR_HALF)
        self.__dict__.setdefault("_anchor_box", {})[room] = box
        return a

    def _push(self, room, anchor, pts, paths, rmin, dur, prev, earlier, own) -> list[Move]:
        """A slow push toward the room's working surfaces, aimed at them from start to end."""
        aim = np.array([anchor[0], anchor[1], min(anchor[2] + 0.15, AIM_Z_MAX)])
        out = []
        for a in sorted(own):
            for b in range(len(pts)):
                if a == b:
                    continue
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not PUSH_LEN[0] <= L <= PUSH_LEN[1]:
                    continue
                d0, d1 = float(np.linalg.norm(anchor[:2] - pts[a])), float(np.linalg.norm(anchor[:2] - pts[b]))
                if not (PUSH_END[0] <= min(d0, d1) and max(d0, d1) <= PUSH_END[1] + 0.8 and abs(d1 - d0) >= 0.3):
                    continue
                mv, to = pts[b] - pts[a], anchor[:2] - pts[a]
                toward = float(mv @ to) / (np.linalg.norm(mv) * np.linalg.norm(to))
                if abs(toward) < 0.6:                     # straight toward what it shows, or back from it
                    continue
                path, plen = self._path(paths, a, b, rmin)
                if path is None or plen > 1.3 * L or plen / dur * 1.5 > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                tgt = cap_pitch(path, np.repeat(aim[None], len(path), 0), PITCH_LADDER[0])
                first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                depth = self.space.view_depth(path[0], math.atan2(*(aim[:2] - path[0])[::-1]), HFOV)
                score = 1.0 + 0.3 * plen + 0.5 * abs(toward) - 0.4 * abs(min(d0, d1) - 2.2) + 0.05 * min(depth, 5.0)
                out.append(Move("push" if toward > 0 else "pull", path, tgt, score,
                                subject=self.__dict__.get("_anchor_box", {}).get(room),
                                why={"travel_m": round(plen, 2), "toward": round(toward, 2),
                                     "end_from_anchor_m": round(d1, 2), "anchor": np.round(anchor, 2).tolist()}))
        return out

    def _goal(self, room: str) -> np.ndarray:
        """What a reveal of `room` turns onto: its main piece (its top), else the middle of its floor."""
        pieces = self.space.main_pieces(room)
        if pieces:
            return np.array([*pieces[0].centre, pieces[0].top])
        sp = self.space
        ys, xs = np.nonzero((sp.room_of == sp.room_keys.index(room)) & (sp.dt >= 0.3))
        return np.array([sp.lo[0] + (xs.mean() + .5) * 0.05, sp.lo[1] + (ys.mean() + .5) * 0.05, EYE - 0.4])

    def _reveal(self, room, next_room, pts, paths, rmin, dur, prev, own) -> list[Move]:
        """The opening from a room with nothing to compose on (a hall): along a free path toward the
        tour's next room, the view turning evenly from down the path onto that room, which fills the
        last frame (REVEAL_SHARE of its middle)."""
        sp = self.space
        goal = self._goal(next_room)
        out = []
        for a in sorted(own):
            for b in range(len(pts)):
                if a == b:
                    continue
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not MOVE_LEN[0] + 0.4 <= L <= MOVE_LEN[1] + 1.4:
                    continue
                d0, d1 = float(np.linalg.norm(goal[:2] - pts[a])), float(np.linalg.norm(goal[:2] - pts[b]))
                if not (d1 < d0 - 0.5 and d1 >= 1.8):    # toward the next room, not into it
                    continue
                path, plen = self._path(paths, a, b, rmin)
                if path is None or plen > 1.25 * L or plen / dur * 1.5 > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                d = path[min(10, len(path) - 1)] - path[0]
                h0 = math.atan2(d[1], d[0])
                hg = math.atan2(goal[1] - path[0][1], goal[0] - path[0][0])
                dh = (h0 - hg + math.pi) % (2 * math.pi) - math.pi
                if abs(dh) > math.radians(REVEAL_SWING):   # start turned part way: an even, slow swing
                    h0 = hg + math.copysign(math.radians(REVEAL_SWING), dh)
                d = np.array([math.cos(h0), math.sin(h0)])
                far = np.array([*(path[0] + d * min(sp.ray(path[0], h0), 6.0)), EYE - LOOK_DROP])
                u = np.linspace(0, 1, len(path))
                w = (u * u * (3 - 2 * u))[:, None]
                tgt = cap_pitch(path, (1 - w) * far + w * goal, PITCH_LADDER[0])
                first = (path[0], tgt[0])
                if jump_cut(prev, first):
                    continue
                h1 = math.atan2(*(tgt[-1][:2] - path[-1])[::-1])
                share = sp.own_share(next_room, path[-1], h1, HFOV)
                if share < REVEAL_SHARE:
                    continue
                depth = sp.view_depth(path[0], h0, HFOV)
                score = 1.0 + 0.3 * plen + 1.5 * share + 0.1 * min(depth, 6.0) + 0.3 * parallax(path, tgt)
                out.append(Move("reveal", path, tgt, score,
                                why={"travel_m": round(plen, 2), "reveals": next_room, "share_next": round(share, 2),
                                     "view_depth_m": round(depth, 2)}))
        return out

    # ---- between rooms ---------------------------------------------------------------------------
    @staticmethod
    def _along(poly: np.ndarray) -> np.ndarray:
        return np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]

    def link(self, src: str, hall: str, dst: str, route, dur: float, prev=None) -> tuple[Move | None, list[str]]:
        """The link shot: the tour leaves `src` for `dst` through `hall`. From a spot of src's space
        on the walking route, LINK_BEFORE m before the hallway, a slow pan from src's side onto the
        hallway (LINK_AHEAD m into it, toward dst), drifting LINK_DRIFT m toward it."""
        sp = self.space
        self.turn_budget, self.move_budget = MAX_TURN_DPS * PRE_SYNC_SHARE["room"], MAX_MOVE_MPS * PRE_SYNC_SHARE["room"]
        poly, _ = route
        at = [sp.room_at(q) for q in poly]
        h_in = next((k for k, r in enumerate(at) if r == hall), None)
        if h_in is None:
            return None, [f"link_no_route:{hall}"]
        s_ = self._along(poly)
        mine = sp.spaces.get(src, {src})
        cands = []
        for k in range(h_in):
            back = s_[h_in] - s_[k]
            if not LINK_BEFORE[0] <= back <= LINK_BEFORE[1] or at[k] not in mine | {hall}:
                continue
            p0 = poly[k]
            if sp.clearance(p0) < RMIN_LADDER[-1]:
                continue
            k1 = int(np.searchsorted(s_, s_[k] + LINK_DRIFT))
            p1 = poly[min(k1, h_in)]
            # it ends looking down the hallway: at the farthest point of the route past the hallway's
            # start still in sight (at least LINK_AHEAD m on), toward the next room
            aim = None
            for j in range(len(poly) - 1, h_in, -1):
                if s_[j] - s_[h_in] < LINK_AHEAD:
                    break
                d = poly[j] - p1
                dist = float(np.linalg.norm(d))
                if dist > 0.5 and sp.ray(p1, math.atan2(d[1], d[0]), max_m=dist + 0.5) >= 0.95 * dist:
                    aim = np.array([*poly[j], EYE - LOOK_DROP])
                    break
            if aim is None:
                continue
            h_end = math.atan2(aim[1] - p1[1], aim[0] - p1[0])
            for sign in (1, -1):
                # start turned toward the room being left: the side whose view holds more of it
                for swing in np.arange(LINK_SWING[1], LINK_SWING[0] - 1e-6, -5.0):
                    h0 = h_end + sign * math.radians(swing)
                    u = np.linspace(0, 1, SAMPLES)
                    w = u * u * (3 - 2 * u)
                    eye = np.outer(1 - u, p0) + np.outer(u, p1)
                    hs = h0 + (h_end - h0) * w
                    rr = np.linalg.norm(aim[:2] - eye, axis=1)
                    tgt = np.c_[eye[:, 0] + rr * np.cos(hs), eye[:, 1] + rr * np.sin(hs), np.full(SAMPLES, aim[2])]
                    turn, move = trial_speeds(eye, tgt, dur)
                    if turn <= self.turn_budget and move <= self.move_budget:
                        break
                else:
                    continue
                if jump_cut(prev, (eye[0], tgt[0])):
                    continue
                src_share = sp.own_share(src, eye[0], h0, HFOV)
                hall_share = sp.own_share(hall, eye[-1], h_end, HFOV)
                score = 1.0 + 1.0 * src_share + 1.5 * hall_share + 0.3 * swing / LINK_SWING[1] \
                    + 0.1 * min(sp.view_depth(eye[-1], h_end, HFOV), 6.0)
                cands.append(Move("link", eye, tgt, score, why={
                    "from": src, "onto": hall, "toward": dst, "looks_down_m": round(float(np.linalg.norm(aim[:2] - p1)), 2), "swing_deg": float(swing), "src_share": round(src_share, 2),
                    "hall_share": round(hall_share, 2), "turn_dps": round(turn, 1), "move_mps": round(move, 2)}))
        if not cands:
            return None, [f"link_no_move:{hall}"]
        if self.occ is not None:                      # occlusion renders around where the link stands
            pts_ = np.concatenate([m.eye for m in cands] + [m.tgt[-1:, :2] for m in cands])
            c = pts_.mean(0)
            self.occ.focus(c, float(np.linalg.norm(pts_ - c, axis=1).max()) + OCC_NEAR + 0.5)
        best, least = self._check(hall, cands, dur)
        if best is None:
            return None, [f"link_blocked:{hall}"]
        best.why.update(candidates=len(cands), occlusion=round(best.occ, 3))
        return best, []

    def _enter(self, room, piece, route, rmin, dur, prev, earlier, after_exit=False) -> list[Move]:
        """In along the walking route: from ENTER_BEFORE m before the doorway (in the hallway) to a
        spot ENTER_AFTER m into the room, looking at its main piece (else the middle of its floor).
        After an exit (the last room's shot flew to its doorway): it starts AFTER_EXIT m inside,
        still heading along the route, and turns onto the piece - one flight, cut at the door."""
        if after_exit:
            return self._enter_after_exit(room, piece, route, rmin, dur, earlier)
        sp = self.space
        poly, _ = route
        at = [sp.room_at(q) for q in poly]
        d_in = next((k for k, r in enumerate(at) if r == room), None)
        if d_in is None or d_in == 0:
            return []
        door = poly[d_in]
        fp = (piece.frame or piece) if piece is not None else None
        aim = np.array([*fp.centre, fp.top]) if fp is not None else self._goal(room)
        s_ = self._along(poly)
        starts = [k for k in range(d_in) if ENTER_BEFORE[0] <= s_[d_in] - s_[k] <= ENTER_BEFORE[1]
                  and sp.clearance(poly[k]) >= rmin][::3]
        mask = (sp.dt >= rmin) & (sp.room_of >= 0)
        inside = sp.spread_points(sp.space_mask(room, rmin), 36)
        ends = [q for q in inside if sp.room_at(q) == room and ENTER_AFTER[0] <= np.linalg.norm(q - door) <= ENTER_AFTER[1]]
        if not starts or not ends:
            return []
        pts = np.array([poly[k] for k in starts] + list(ends))
        paths = sp.paths_between(pts, mask)
        out = []
        for a in range(len(starts)):
            for b in range(len(starts), len(pts)):
                path, plen = self._path(paths, a, b, rmin)
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if path is None or plen > 1.4 * L or plen / dur * 1.5 > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                tgt = cap_pitch(path, np.repeat(aim[None], len(path), 0), PITCH_LADDER[0])
                first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                share = sp.own_share(room, path[0], math.atan2(*(aim[:2] - path[0])[::-1]), HFOV)
                if share < ENTER_SHARE:                          # it sees into the room from the start
                    continue
                core = 0.0
                if fp is not None:
                    core = next((c for c in CORES if in_frame(path[-1], tgt[-1], fp, c)), 0.0)
                score = 1.0 + 0.3 * plen + 1.0 * share + 1.0 * core
                out.append(Move("enter", path, tgt, score, piece=piece, subject=piece_box(fp) if fp is not None else None,
                                why={"travel_m": round(plen, 2), "share_start": round(share, 2), "core_end": core,
                                     "through": [r for r in route[1] if r != room][-1:] or None}))
        return out

    # ---- small rooms: moves that turn (user 2026-11-16, Krk: "there is almost no pan") ------------
    def _arc(self, room, piece, rmin, dur, prev, earlier) -> list[Move]:
        """An arc around the room's piece: the camera slides along a circle round it (ARC_R m out,
        ARC_SWEEP deg), aimed at its top all along - movement and parallax where a straight walk has
        no room. Every point on free floor of the room's space; the piece's top in frame throughout."""
        sp = self.space
        fp = piece.frame or piece
        c = np.asarray(fp.centre, float)
        top = np.array([*c, fp.top])
        mine = sp.spaces.get(room, {room})
        out = []
        u = np.linspace(0, 1, SAMPLES)
        for r in ARC_R:
            for a0 in np.radians(np.arange(0, 360, 20)):
                for sweep in ARC_SWEEP:
                    for sign in (1, -1):
                        a = a0 + sign * math.radians(sweep) * u
                        eye = c + r * np.c_[np.cos(a), np.sin(a)]
                        if any(sp.clearance(q) < rmin or sp.room_at(q) not in mine for q in eye[::7]):
                            continue
                        tgt = cap_pitch(eye, np.repeat(top[None], SAMPLES, 0), PITCH_LADDER[0])
                        first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                        if jump_cut(prev, first) or repeats(first, last, earlier):
                            continue
                        core = next((k for k in CORES if all(in_frame(q, t, fp, k) for q, t in zip(eye[::10], tgt[::10]))), None)
                        if core is None:
                            continue
                        score = 1.0 + 1.5 * core + 0.5 * sweep / ARC_SWEEP[-1] - 0.3 * abs(r - 2.7)
                        out.append(Move("arc", eye, tgt, score, PITCH_LADDER[0], piece, subject=piece_box(fp),
                                        why={"radius_m": r, "sweep_deg": sweep, "core": core}))
        return _keep_best(out)

    def _subjects(self, room: str) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """What a pan of the room may show: its pieces and its low features - (aim point, box, name)."""
        sp = self.space
        out = []
        for pc in sp.main_pieces(room):
            fp = pc.frame or pc
            out.append((np.array([*fp.centre, fp.top]), piece_box(fp), pc.cls or "piece"))
        for o in sp.room_objects(room, FEATURES):
            if o["max"][2] <= LOW_TOP and o["class"] not in NO_AIM:
                out.append((np.array([*o["centre"][:2], max(o["centre"][2], EYE - 0.6)]), objects_box([o]), o["class"]))
        return out

    def _room_pan(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A pan across the room's own things: from an open spot, the view turns from one subject to
        another (20-55 deg apart, both at least AIM_MIN away), drifting a little toward them."""
        sp = self.space
        subs = self._subjects(room)
        if len(subs) < 2:
            return []
        out = []
        u = np.linspace(0, 1, SAMPLES)
        w = (u * u * (3 - 2 * u))[:, None]
        spots = sorted(pts, key=lambda p: -sp.clearance(p))[:10]
        for p in spots:
            for i, (ta, _, na) in enumerate(subs):
                for j, (tb, box_b, nb) in enumerate(subs):
                    if i == j:
                        continue
                    da, db = ta[:2] - p, tb[:2] - p
                    if np.linalg.norm(da) < AIM_MIN or np.linalg.norm(db) < AIM_MIN:
                        continue
                    ang = math.degrees(abs((math.atan2(db[1], db[0]) - math.atan2(da[1], da[0]) + math.pi) % (2 * math.pi) - math.pi))
                    if not 20.0 <= ang <= 55.0:
                        continue
                    mid = (ta[:2] + tb[:2]) / 2 - p
                    e1 = p + 0.3 * mid / max(1e-9, np.linalg.norm(mid))
                    if sp.clearance(e1) < rmin or sp.room_at(e1) != room:
                        e1 = p
                    eye = np.outer(1 - u, p) + np.outer(u, e1)
                    tgt = cap_pitch(eye, (1 - w) * ta + w * tb, PITCH_LADDER[0])
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    score = 0.9 + 0.4 * ang / 55.0 + 0.05 * min(sp.view_depth(p, math.atan2(mid[1], mid[0]), HFOV), 5.0)
                    out.append(Move("room_pan", eye, tgt, score, subject=box_b,
                                    why={"from": na, "to": nb, "sweep_deg": round(ang)}))
        return _keep_best(out)

    # ---- the scene picks the move ----------------------------------------------------------------
    def type_signs(self, room: str) -> list[dict]:
        from tour_worker.scoring.rooms import canon
        cls = TYPE_SIGNS.get(canon(self.space.room_types.get(room)) or "", ())
        return self.space.room_objects(room, cls) if cls else []

    def sees(self, eye2, tgt, o: dict, ez: float = EYE) -> bool:
        """o's centre inside the frame (its middle 80%) and in sight (no wall between)."""
        sp = self.space
        c = np.asarray(o["centre"], float)
        v = c[:2] - np.asarray(eye2[:2])
        dist = float(np.linalg.norm(v))
        if not 0.5 < dist < 9.0:
            return False
        V, K = look_at([eye2[0], eye2[1], ez], tgt), intrinsics(HFOV, FRAME_W, FRAME_H)
        q = V[:3, :3] @ c + V[:3, 3]
        if q[2] < 0.3:
            return False
        px, py = K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]
        if not (0.1 * FRAME_W <= px <= 0.9 * FRAME_W and 0.05 * FRAME_H <= py <= 0.95 * FRAME_H):
            return False
        return sp.ray(np.asarray(eye2[:2]), math.atan2(v[1], v[0]), max_m=dist + 0.3) >= dist - 0.6

    def whole_boxes(self, room: str) -> list:
        """(8, 3) boxes of the room's beds / sofas, each its labelled pieces merged (WHOLE_GAP)."""
        cache = self.__dict__.setdefault("_whole", {})
        if room in cache:
            return cache[room]
        out = []
        for cls in TYPE_WHOLE:
            groups = [[o] for o in self.space.room_objects(room, (cls,))]
            merged = True
            while merged and len(groups) > 1:
                merged = False
                for a in range(len(groups)):
                    for b in range(a + 1, len(groups)):
                        la = np.min([o["min"] for o in groups[a]], 0); ha = np.max([o["max"] for o in groups[a]], 0)
                        lb = np.min([o["min"] for o in groups[b]], 0); hb = np.max([o["max"] for o in groups[b]], 0)
                        gap = np.maximum(0.0, np.maximum(la[:2] - hb[:2], lb[:2] - ha[:2])).max()
                        if gap <= WHOLE_GAP:
                            groups[a] += groups.pop(b)
                            merged = True
                            break
                    if merged:
                        break
            out += [objects_box(g) for g in groups]
        cache[room] = [b for b in out if b is not None]
        return cache[room]

    def shows_type(self, room: str, m: Move, whole: bool = False) -> bool:
        """The move's middle or last frame shows something that makes the room what it is; whole:
        a bed or sofa (TYPE_WHOLE) among them is in the last frame whole, with floor under it."""
        signs = self.type_signs(room)
        zs = m.zs
        if whole:
            # the whole bed / sofa: its labelled pieces merged (user 2026-11-17: the primary bedroom
            # passed on a fragment labelled "bed" while the frame showed a corner of the bed)
            hit = any(frames_fully(m.eye[-1], m.tgt[-1], box, float(zs[-1])) for box in self.whole_boxes(room))
            m.why["frames_type"] = hit
            return hit
        hit = any(self.sees(m.eye[k], m.tgt[k], o, float(zs[k])) for k in (len(m.eye) // 2, len(m.eye) - 1) for o in signs)
        m.why["shows_type"] = hit
        return hit

    # ---- the tour's sequence (the photo tours' rules, planner/cinematic.py) ---------------------
    @staticmethod
    def side(m: Move) -> int:
        """Which way the view travels across the frame: +1 / -1 (sideways travel, else the turn), 0."""
        k = len(m.eye) // 2
        h = m.tgt[k][:2] - m.eye[k]
        v = m.eye[-1] - m.eye[0]
        if np.linalg.norm(v) >= 0.3 and parallax(m.eye, m.tgt) >= 0.35:
            c = h[0] * v[1] - h[1] * v[0]
        else:
            d0, d1 = m.tgt[0][:2] - m.eye[0], m.tgt[-1][:2] - m.eye[-1]
            c = d0[0] * d1[1] - d0[1] * d1[0]
            if abs(math.degrees(math.atan2(c, float(d0 @ d1)))) < 8.0:
                return 0
        return 1 if c > 0 else -1

    def record(self, m: Move) -> None:
        """A shot the tour plays: its kind and side, for the next shots' sequence costs."""
        self.__dict__.setdefault("_history", []).append((m.kind, self.side(m)))
        self._last_kind = m.kind

    def sequence_cost(self, m: Move) -> float:
        """Score lost to the tour so far (2D: cinematic.py's beam search): the previous kind again,
        two shots back, the same side as the previous shot (sides alternate), each earlier use of
        the kind the same way, a kind past its second use, and moving in / out past a third of the
        tour or twice running."""
        hist = self.__dict__.get("_history", [])
        if not hist:
            return 0.0
        sd = self.side(m)
        cost = 0.0
        if hist[-1][0] == m.kind:
            cost += SAME_KIND_W
        if len(hist) >= 2 and hist[-2][0] == m.kind:
            cost += TWO_BACK_W
        if sd and hist[-1][1] == sd:
            cost += SAME_SIDE_W
        cost += USED_DIR_W * sum(1 for k, d in hist if k == m.kind and d == sd)
        cost += OVERUSE_W * max(0, sum(1 for k, _ in hist if k == m.kind) - 1)
        if m.kind not in FLIGHT:                  # the library mixes: caps per kind and share
            uses = sum(1 for k, _ in hist if k == m.kind)
            n_tour = max(getattr(self, "tour_shots", 9), 1)
            if uses >= KIND_CAP.get(m.kind, 2) or uses + 1 > max(1, math.ceil(KIND_SHARE * n_tour)):
                cost += CAP_W
        if m.kind in DOLLY_KINDS:
            cap = math.ceil(max(getattr(self, "tour_shots", 9), 1) / 3)
            if sum(1 for k, _ in hist if k in DOLLY_KINDS) >= cap:
                cost += DOLLY_CAP_W
            if hist[-1][0] in DOLLY_KINDS:
                cost += DOLLY_RUN_W
        m.why["sequence_cost"] = round(cost, 2)
        return cost

    def at_cap(self, kind: str) -> bool:
        if kind in FLIGHT:
            return False
        hist = self.__dict__.get("_history", [])
        uses = sum(1 for k, _ in hist if k == kind)
        n_tour = max(getattr(self, "tour_shots", 9), 1)
        return uses >= KIND_CAP.get(kind, 2) or uses + 1 > max(1, math.ceil(KIND_SHARE * n_tour))

    @staticmethod
    def sinks(m: Move) -> bool:
        """The view drops past what it shows (2D: never sink below the interest): its aim ends
        SINK_M lower than it starts while the subject's top is above the end aim. A rise looks down
        on purpose; a push may tilt onto its counter."""
        if m.kind in ("rise", "descend", "push", "pull", "detail", "link") or m.subject is None:
            return False
        return bool(m.tgt[-1][2] < m.tgt[0][2] - SINK_M and float(m.subject[:, 2].max()) > m.tgt[-1][2])

    def profile(self, room: str) -> dict:
        """What the room is like, and the kinds of move that suits (prior per kind, 0..1)."""
        cache = self.__dict__.setdefault("_profiles", {})
        if room in cache:
            return cache[room]
        sp = self.space
        area = float(sp.room_area(room))
        open_plan = len(sp.spaces.get(room, {room})) > 1
        pieces = [p.cls for p in sp.main_pieces(room)]
        subjects = self._subjects(room)
        prior = {}
        if open_plan or area >= RISE_AREA:
            prior["rise"] = 1.0                             # space to open up from above
        # the photo tours' scene fit (planner/cinematic.py): a push only when one subject dominates
        # the room's features; a truck / pan when they spread wide; a bathroom pulls back
        dom, spread = self._dominance_spread(room)
        prior["push"] = float(np.clip(2.0 * (dom - 0.3), -0.5, 1.0))
        if self.anchor(room) is not None or spread >= 0.5:
            prior["truck"] = max(1.0 if self.anchor(room) is not None else 0.0, min(1.0, spread))
        prior["room_pan"] = -0.5 + 0.5 * min(1.0, spread)  # a pan only where nothing else will do
        if any(c in ("table", "kitchen island", "bed", "coffee table", "pool table") for c in pieces):
            prior["arc"] = 1.0                              # a piece with floor round it: circle it
        prior["corner_sweep"] = 0.6                         # shows two or three walls in one move
        if area < 14.0 or spread >= 0.6:
            prior["panorama"] = 0.5                         # a small room, or things all round it
        if (open_plan or area >= RISE_AREA) and any(c in ("table", "bed", "kitchen island", "sofa") for c in pieces):
            prior["descend"] = 0.8                          # come down onto the table / bed
        from tour_worker.scoring.rooms import canon
        if canon(sp.room_types.get(room)) in ("bathroom", "primary_bathroom", "ensuite", "powder_room"):
            prior["pull"], prior["push"] = 1.0, min(prior["push"], 0.0)
        elif area < 12.0:
            prior["pull"] = 0.5                             # small: move out, not across
        cache[room] = {"area": round(area, 1), "open_plan": open_plan, "pieces": pieces,
                       "subjects": len(subjects), "dominance": round(dom, 2), "spread": round(spread, 2),
                       "prior": prior}
        return cache[room]

    def _dominance_spread(self, room: str) -> tuple[float, float]:
        """(the largest labelled feature's share of the room's feature weight x size; how widely
        the features spread - their extent across the room's longer side, 0..1). The photo tours
        read these off the photo (scene_context.py dominance / h_extent); here off the labels."""
        sp = self.space
        objs = [o for o in sp.room_objects(room, FEATURES)]
        if not objs:
            return 0.0, 0.0
        w = np.array([FEATURES[o["class"]] * max(0.05, float(np.prod(np.asarray(o["max"][:2]) - np.asarray(o["min"][:2]))))
                      for o in objs])
        dom = float(w.max() / w.sum())
        c = np.array([o["centre"][:2] for o in objs], float)
        mask = sp.room_of == sp.room_keys.index(room)
        ys, xs = np.nonzero(mask)
        side = max(np.ptp(xs), np.ptp(ys), 1) * 0.05
        ext = float(max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))) if len(c) > 1 else 0.0
        return dom, min(1.0, ext / side)

    def _truck(self, room, rmin, dur, prev, earlier) -> list[Move]:
        """A truck: the camera slides sideways past one of the room's subjects (a counter run, a
        bed, shelves), TRUCK_DIST m out, its aim sliding TRUCK_AIM of the way with it - parallax
        across the whole frame, the subject held."""
        sp = self.space
        subs = list(self._subjects(room))
        a = self.anchor(room)
        if a is not None:
            subs.append((np.array([a[0], a[1], min(a[2] + 0.15, AIM_Z_MAX)]),
                         self.__dict__.get("_anchor_box", {}).get(room), "working surface"))
        mine = sp.spaces.get(room, {room})
        u = np.linspace(0, 1, SAMPLES)
        w = (u * u * (3 - 2 * u))[:, None]
        out = []
        for ta, box, name in subs:
            for dist in TRUCK_DIST:
                for a0 in np.radians(np.arange(0, 360, 20)):
                    n = np.array([math.cos(a0), math.sin(a0)])
                    c = ta[:2] + dist * n
                    if sp.clearance(c) < rmin or sp.room_at(c) not in mine:
                        continue
                    if sp.ray(c, math.atan2(-n[1], -n[0]), max_m=dist + 0.3) < dist - 0.6:
                        continue                              # it is behind a wall from here
                    side = np.array([-n[1], n[0]])
                    for L in TRUCK_LEN:
                        for sg in (1, -1):
                            p0, p1 = c - sg * side * L / 2, c + sg * side * L / 2
                            eye = (1 - w) * p0 + w * p1
                            if any(sp.clearance(q) < rmin or sp.room_at(q) not in mine for q in eye[::7]):
                                continue
                            shift = (w - 0.5) * L * TRUCK_AIM * sg
                            tgt = np.c_[ta[0] + shift[:, 0] * side[0], ta[1] + shift[:, 0] * side[1], np.full(SAMPLES, ta[2])]
                            tgt = cap_pitch(eye, tgt, PITCH_LADDER[0])
                            first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                            if jump_cut(prev, first) or repeats(first, last, earlier):
                                continue
                            score = 1.2 + 0.4 * L - 0.3 * abs(dist - 2.8)
                            out.append(Move("truck", eye, tgt, score, subject=box,
                                            why={"of": name, "slide_m": L, "out_m": dist}))
        return _keep_best(out)

    def _rise(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A rise: in an open or large room, from an open spot the camera climbs from eye height to
        RISE_Z (under the ceiling), drifting toward the room's main things and tilting down to
        RISE_PITCH - the space opens up below, as from a drone."""
        sp = self.space
        prof = self.profile(room)
        if "rise" not in prof["prior"]:
            return []
        top = RISE_Z
        if "scene_ceiling_missing" not in getattr(sp, "warnings", []):
            top = min(RISE_Z, float(sp.room_height) - RISE_HEADROOM)
        if top < EYE + 0.35:
            return []
        subs = self._subjects(room)
        if not subs:
            g = self._goal(room)
            subs = [(g, None, "floor")]
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        z = EYE + (top - EYE) * w
        out = []
        for p in sorted(pts, key=lambda q: -sp.clearance(q))[:10]:
            for ta, box, name in subs:
                v = ta[:2] - p
                d = float(np.linalg.norm(v))
                if not RISE_DIST[0] <= d <= RISE_DIST[1]:
                    continue
                if sp.ray(p, math.atan2(v[1], v[0]), max_m=d + 0.3) < d - 0.6:
                    continue
                for drift in RISE_DRIFT:
                    p1 = p + drift * v / d
                    if sp.clearance(p1) < rmin or sp.room_at(p1) not in sp.spaces.get(room, {room}):
                        continue
                    eye = np.outer(1 - w, p) + np.outer(w, p1)
                    h = np.linalg.norm(ta[:2] - eye, axis=1)
                    # level-ish at the start (the usual look), RISE_PITCH down at the top
                    pitch = np.radians(PITCH_LADDER[0] * 0.6 + (RISE_PITCH - PITCH_LADDER[0] * 0.6) * w)
                    look = np.minimum(np.tan(pitch), (z - 0.2) / np.maximum(h, 0.1))
                    tgt = np.c_[np.repeat(ta[None, :2], SAMPLES, 0), z - h * look]
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    depth = sp.view_depth(p, math.atan2(v[1], v[0]), HFOV)
                    score = 1.1 + 0.1 * min(depth, 6.0) + 0.2 * drift
                    out.append(Move("rise", eye, tgt, score, subject=box, z=z,
                                    why={"to_m": round(top, 2), "of": name, "drift_m": drift, "pitch_end": RISE_PITCH}))
        return _keep_best(out, 20)

    def exit(self, room: str, nxt: str, route, dur: float, prev=None, earlier=()) -> tuple[Move | None, list[str]]:
        """The room's last shot heads for the next room: along the walking route to EXIT_STOP m before
        where the room ends (its doorway, or the open-plan neighbour's floor), looking ahead where it
        is going - the next shot picks the flight up inside the next room."""
        sp = self.space
        self.turn_budget, self.move_budget = MAX_TURN_DPS * PRE_SYNC_SHARE["room"], MAX_MOVE_MPS * PRE_SYNC_SHARE["room"]
        poly, _ = route
        mine = sp.spaces.get(room, {room}) if nxt not in sp.spaces.get(room, {room}) else {room}
        at = [sp.room_at(q) for q in poly]
        k_out = next((k for k, r in enumerate(at) if k > 0 and r not in mine), None)
        if k_out is None:
            return None, [f"exit_no_route:{room}"]
        s_ = self._along(poly)
        ends = [k for k in range(k_out) if EXIT_STOP <= s_[k_out] - s_[k] <= EXIT_STOP + 0.6]
        cands = []
        for rmin in RMIN_LADDER:
            for ke in ends[::2]:
                if sp.clearance(poly[ke]) < rmin or self.in_doorway(poly[ke]):
                    continue
                for ks in range(ke):
                    L = s_[ke] - s_[ks]
                    if not EXIT_LEN[0] <= L <= EXIT_LEN[1] or (ks % 3) or at[ks] not in mine:
                        continue
                    if L / dur * 2.1 > self.move_budget:          # eased in: its peak ~2x its mean speed
                        continue
                    seg = poly[ks:ke + 1]
                    if min(sp.clearance(q) for q in seg) < rmin * 0.9:
                        continue
                    path = resample(smooth(sp, seg, rmin), SAMPLES)
                    sp_ = self._along(path)
                    # look ahead along the route, EXIT_AHEAD m on (into the next room at the end)
                    s0 = s_[ks]
                    look = [np.interp(min(s0 + d + EXIT_AHEAD, s_[-1]), s_, poly[:, j]) for d in sp_ for j in (0, 1)]
                    look = np.array(look).reshape(-1, 2)
                    tgt = np.c_[look, np.full(SAMPLES, EYE - LOOK_DROP)]
                    first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    score = 1.0 + 0.3 * L + 0.1 * min(sp.view_depth(path[-1], math.atan2(*(tgt[-1][:2] - path[-1])[::-1]), HFOV), 6.0)
                    cands.append(Move("exit", path, tgt, score, easing="ease_in", why={"toward": nxt, "travel_m": round(L, 2),
                                                                      "stops_before_m": round(float(s_[k_out] - s_[ke]), 2)}))
            if cands:
                break
        cands = [m for m in cands if self.moves_enough(m, dur) and sp.clearance(m.eye[0]) >= WALL_CLEAR]
        if not cands:
            return None, [f"exit_no_move:{room}"]
        if self.occ is not None:
            pts_ = np.concatenate([m.eye for m in cands])
            c = pts_.mean(0)
            self.occ.focus(c, float(np.linalg.norm(pts_ - c, axis=1).max()) + OCC_NEAR + EXIT_AHEAD)
        best, _ = self._check(room, cands, dur)
        if best is None:
            return None, [f"exit_blocked:{room}"]
        best.why.update(candidates=len(cands), occlusion=round(best.occ, 3))
        self.record(best)
        return best, []

    def _enter_after_exit(self, room, piece, route, rmin, dur, earlier) -> list[Move]:
        sp = self.space
        poly, _ = route
        at = [sp.room_at(q) for q in poly]
        d_in = next((k for k, r in enumerate(at) if r == room), None)
        if d_in is None or d_in == 0:
            return []
        s_ = self._along(poly)
        fp = (piece.frame or piece) if piece is not None else None
        aim = np.array([*fp.centre, fp.top]) if fp is not None else self._goal(room)
        inside = [k for k in range(d_in, len(poly)) if AFTER_EXIT[0] <= s_[k] - s_[d_in] <= AFTER_EXIT[1]
                  and at[k] == room and sp.clearance(poly[k]) >= rmin and not self.in_doorway(poly[k])]
        ends = [q for q in sp.spread_points(sp.space_mask(room, rmin), 36)
                if sp.room_at(q) == room and ENTER_AFTER[0] + 1.0 <= np.linalg.norm(q - poly[d_in]) <= ENTER_AFTER[1] + 1.2]
        if not inside or not ends:
            return []
        mask = (sp.dt >= rmin) & (sp.room_of >= 0)
        starts = inside[::2]
        pts = np.array([poly[k] for k in starts] + list(ends))
        paths = sp.paths_between(pts, mask)
        u = np.linspace(0, 1, SAMPLES)
        w = (u * u * (3 - 2 * u))[:, None]
        out = []
        for a, ks in enumerate(starts):
            ahead = np.array([np.interp(min(s_[ks] + EXIT_AHEAD, s_[-1]), s_, poly[:, j]) for j in (0, 1)])
            t0 = np.array([*ahead, EYE - LOOK_DROP])
            for b in range(len(starts), len(pts)):
                path, plen = self._path(paths, a, b, rmin)
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if path is None or plen > 1.4 * L or plen < 0.8 or plen / dur * 1.5 > self.move_budget:
                    continue
                path = resample(path, SAMPLES)
                h0 = path[min(8, SAMPLES - 1)] - path[0]
                if float(h0 @ (ahead - poly[ks])) <= 0:          # it keeps flying the same way
                    continue
                tgt = cap_pitch(path, (1 - w) * t0 + w * aim, PITCH_LADDER[0])
                if repeats((path[0], tgt[0]), (path[-1], tgt[-1]), earlier):
                    continue
                core = next((c for c in CORES if fp is not None and in_frame(path[-1], tgt[-1], fp, c)), 0.0)
                score = 1.5 + 0.3 * plen + 1.0 * core
                out.append(Move("enter", path, tgt, score, piece=piece, subject=piece_box(fp) if fp is not None else None,
                                easing="ease_out",
                                why={"travel_m": round(plen, 2), "after_exit": True, "core_end": core}))
        return out

    def _room_centre(self, room: str) -> np.ndarray:
        sp = self.space
        ys, xs = np.nonzero(sp.room_of == sp.room_keys.index(room))
        return np.array([sp.lo[0] + (xs.mean() + .5) * 0.05, sp.lo[1] + (ys.mean() + .5) * 0.05])

    def _look(self, eye: np.ndarray, heads: np.ndarray) -> np.ndarray:
        """Targets for headings from eye (one per sample): as far as the room goes (2-6 m), a level look."""
        sp = self.space
        out = np.zeros((len(eye), 3))
        for k, (e, h) in enumerate(zip(eye, heads)):
            r = float(np.clip(sp.ray(e, h, max_m=6.0), 2.0, 6.0))
            out[k] = (e[0] + r * math.cos(h), e[1] + r * math.sin(h), EYE - LOOK_DROP)
        return out

    def _corner_sweep(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A corner sweep: from near a corner the camera dollies toward the room's middle while its
        view turns across it (SWEEP_TURN) - two or three walls in one move (coverage)."""
        sp = self.space
        c = self._room_centre(room)
        mine = sp.spaces.get(room, {room})
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        out = []
        for p in sorted(pts, key=lambda q: -float(np.linalg.norm(q - c)))[:12]:
            to = c - p
            dist = float(np.linalg.norm(to))
            if dist < 2.0:
                continue
            hc = math.atan2(to[1], to[0])
            for L in SWEEP_LEN:
                if L > dist - 0.8:
                    continue
                p1 = p + L * to / dist
                eye = np.outer(1 - w, p) + np.outer(w, p1)
                if any(sp.clearance(q) < rmin or sp.room_at(q) not in mine for q in eye[::7]):
                    continue
                for turn in SWEEP_TURN:
                    for side in (1, -1):
                        heads = hc + side * math.radians(turn) * (0.5 - w)
                        tgt = self._look(eye, heads)
                        first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                        if jump_cut(prev, first) or repeats(first, last, earlier):
                            continue
                        score = 1.0 + 0.25 * L + 0.3 * turn / SWEEP_TURN[-1]
                        out.append(Move("corner_sweep", eye, tgt, score, why={"travel_m": L, "turn_deg": turn}))
        return _keep_best(out, 20)

    def _panorama(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A panorama: from an open spot near the room's edge, a slow pan across the room
        (PANO_SWEEP, as wide as the turn limit allows in the time), drifting 0.2 m in."""
        sp = self.space
        c = self._room_centre(room)
        sweep = min(PANO_SWEEP[1], 0.6 * self.turn_budget * dur)
        if sweep < PANO_SWEEP[0]:
            return []
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        out = []
        for p in sorted(pts, key=lambda q: -float(np.linalg.norm(q - c)))[:8]:
            to = c - p
            dist = float(np.linalg.norm(to))
            if dist < 1.5 or sp.clearance(p) < max(rmin, 0.45):
                continue
            hc = math.atan2(to[1], to[0])
            p1 = p + 0.2 * to / dist
            eye = np.outer(1 - w, p) + np.outer(w, p1)
            for side in (1, -1):
                heads = hc + side * math.radians(sweep) * (w - 0.5)
                tgt = self._look(eye, heads)
                first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                out.append(Move("panorama", eye, tgt, 1.0 + 0.4 * sweep / PANO_SWEEP[1], why={"sweep_deg": round(sweep)}))
        return _keep_best(out, 12)

    def _descend(self, room, pts, rmin, dur, prev, earlier) -> list[Move]:
        """A descend: the rise played the other way - from RISE_Z down to eye height while drifting
        toward a low subject (a dining table, a bed), the tilt easing from RISE_PITCH to level. It
        comes down ONTO what it shows, never past it."""
        sp = self.space
        if "descend" not in self.profile(room)["prior"]:
            return []
        top = RISE_Z
        if "scene_ceiling_missing" not in getattr(sp, "warnings", []):
            top = min(RISE_Z, float(sp.room_height) - RISE_HEADROOM)
        if top < EYE + 0.35:
            return []
        low = [(np.array([*(pc.frame or pc).centre, (pc.frame or pc).top]), piece_box(pc.frame or pc), pc.cls)
               for pc in sp.main_pieces(room) if (pc.frame or pc).top <= 1.0]
        if not low:
            return []
        u = np.linspace(0, 1, SAMPLES)
        w = u * u * (3 - 2 * u)
        z = top + (EYE - top) * w
        out = []
        for p in sorted(pts, key=lambda q: -sp.clearance(q))[:10]:
            for ta, box, name in low:
                v = ta[:2] - p
                d = float(np.linalg.norm(v))
                if not RISE_DIST[0] <= d <= RISE_DIST[1]:
                    continue
                for drift in RISE_DRIFT:
                    p1 = p + drift * v / d
                    if sp.clearance(p1) < rmin or sp.room_at(p1) not in sp.spaces.get(room, {room}):
                        continue
                    eye = np.outer(1 - w, p) + np.outer(w, p1)
                    h = np.linalg.norm(ta[:2] - eye, axis=1)
                    pitch = np.radians(RISE_PITCH + (PITCH_LADDER[0] - RISE_PITCH) * w)
                    look = np.minimum(np.tan(pitch), (z - 0.2) / np.maximum(h, 0.1))
                    tgt = np.c_[np.repeat(ta[None, :2], SAMPLES, 0), z - h * look]
                    first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    out.append(Move("descend", eye, tgt, 1.1 + 0.2 * drift, subject=box, z=z,
                                    why={"from_m": round(top, 2), "onto": name, "drift_m": drift}))
        return _keep_best(out, 12)

    def moves_enough(self, m: Move, dur: float) -> bool:
        """A room shot is seen to move: it travels MIN_TRAVEL_PS m/s, or turns MIN_TURN_PS deg/s."""
        travel = float(np.linalg.norm(np.diff(m.eye3, axis=0), axis=1).sum())
        d0, d1 = _dir(m.eye[0], m.tgt[0]), _dir(m.eye[-1], m.tgt[-1])
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(d0 @ d1)))))
        m.why.setdefault("travel_m", round(travel, 2))
        m.why["turn_deg"] = round(turn, 1)
        if travel >= MIN_TRAVEL_PS * dur or turn >= MIN_TURN_PS * dur:
            return True
        if m.kind in ("push", "pull"):             # what a push shows moving: its subject growing
            a0 = float(np.linalg.norm(m.tgt[0][:2] - m.eye[0]))
            a1 = float(np.linalg.norm(m.tgt[-1][:2] - m.eye[-1]))
            scale = abs(math.log(max(a0, 0.1) / max(a1, 0.1)))
            m.why["scale"] = round(math.exp(scale), 2)
            return scale >= MIN_SCALE_PS * dur
        return False

    def framed(self, m: Move) -> bool:
        """The subject a room shot ends on is whole in its last frame, with floor under it. A push
        or pull that would cut the floor off tilts down as it goes (up to TILT_MAX at its end) -
        a push in that settles onto the counter - rather than being dropped."""
        if m.subject is None:
            return True
        ze = float(m.zs[-1])
        if frames_fully(m.eye[-1], m.tgt[-1], m.subject, ze):
            return True
        if m.kind not in ("push", "pull"):
            return False
        h = float(np.linalg.norm(m.tgt[-1][:2] - m.eye[-1][:2]))
        p0 = math.degrees(math.atan2(ze - m.tgt[-1][2], h))
        w = np.linspace(0.0, 1.0, len(m.tgt))
        w = w * w * (3 - 2 * w)                            # eased: the tilt grows as the move closes
        for p in np.arange(math.ceil(p0) + 1.0, TILT_MAX + 0.5, 1.0):
            dz = float(m.tgt[-1][2] - (ze - h * math.tan(math.radians(p))))
            end = m.tgt[-1].copy()
            end[2] -= dz
            if frames_fully(m.eye[-1], end, m.subject, ze):
                m.tgt = m.tgt.copy()
                m.tgt[:, 2] -= dz * w
                m.why["tilt_deg"] = round(float(p), 1)
                return True
        return False

    # ---- move kinds ------------------------------------------------------------------------------
    def _path(self, paths, i, j, rmin):
        """The smoothed walking path from point i to j and its length - cached on the path finder
        (each kind of move, and each step of the pitch ladder, asks for the same pairs)."""
        cache = paths.__dict__.setdefault("smoothed", {})
        key = (i, j, rmin)
        if key not in cache:
            poly = paths(i, j)
            if poly is None or len(poly) < 2:
                cache[key] = (None, 0.0)
            else:
                q = smooth(self.space, poly, rmin)
                cache[key] = (q, float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()))
        q, plen = cache[key]
        return (None if q is None else q.copy()), plen

    def _walkway(self, room, piece, pts, paths, rmin, dur, pitch, prev, earlier, own) -> list[Move]:
        fp = piece.frame or piece                     # what is framed: the labelled object, else the piece
        top = np.array([*fp.centre, fp.top])
        out = []
        keep = [k for k, p in enumerate(pts) if piece.dist(p) > 0.05]
        for a_i, a in enumerate(keep):
            for b in keep[a_i + 1:]:
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not MOVE_LEN[0] <= L <= MOVE_LEN[1]:
                    continue
                for i, j in ((a, b), (b, a)):
                    # a walkway's subject is the room's own piece: it may start anywhere in the open
                    # plan (a dining table filmed from the living side is a dining shot)
                    path, plen = self._path(paths, i, j, rmin)
                    if path is None or plen > 1.3 * L or plen / dur * 1.5 > self.move_budget:
                        continue
                    path = resample(path, SAMPLES)
                    tgt = cap_pitch(path, np.repeat(top[None], len(path), 0), pitch)
                    first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    core = next((c for c in CORES if all(in_frame(q, t, fp, c) for q, t in zip(path[::10], tgt[::10]))), None)
                    if core is None:                          # the piece leaves the frame somewhere
                        continue
                    par = parallax(path, tgt)
                    d0 = float(np.linalg.norm(path[0] - fp.centre))
                    k = len(path) // 2
                    depth = self.space.view_depth(path[k], math.atan2(*(tgt[k][:2] - path[k])[::-1]), HFOV)
                    score = (1.0 + 1.0 * par + 0.3 * plen + 1.5 * core - 1.5 * abs(d0 - SIT_BACK) / SIT_BACK
                             + 0.1 * min(depth, 5.0) - 0.15 * (pitch - PITCH_LADDER[0]))
                    out.append(Move("walkway", path, tgt, score, pitch, piece, subject=piece_box(fp),
                                    why={"parallax": round(par, 2), "travel_m": round(plen, 2), "core": core,
                                         "start_from_piece_m": round(d0, 2), "pitch_cap": pitch}))
        return out

    def _glide(self, room, pts, paths, rmin, dur, prev, earlier, opening, own, closing=False) -> list[Move]:
        out = []
        for a in sorted(own):
            for b in range(len(pts)):
                if a == b:
                    continue
                L = float(np.linalg.norm(pts[b] - pts[a]))
                if not MOVE_LEN[0] + 0.4 <= L <= MOVE_LEN[1] + 1.4:
                    continue
                path, plen = self._path(paths, a, b, rmin)
                if path is None or plen > 1.25 * L or plen / dur * 1.5 > self.move_budget:
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
                aims = [("depth", np.array([*far, EYE - min(0.45, 0.12 * dist)]), None)]
                for o in self.space.room_objects(room, FEATURES):      # or push toward a feature ahead
                    if o["max"][2] > LOW_TOP or o["class"] in NO_AIM:  # never upper cabinets, a door, a curtain
                        continue
                    v = np.asarray(o["centre"][:2]) - path[-1]
                    if np.linalg.norm(v) > 0.8 and float(v @ d) / np.linalg.norm(v) > 0.5:
                        aims.append((o["class"], np.array([*o["centre"][:2], max(o["centre"][2], EYE - 0.6)]),
                                     objects_box([o])))
                for why_aim, aim, box in aims:
                    tgt = cap_pitch(path, np.repeat(aim[None], len(path), 0), PITCH_LADDER[0])
                    first, last = (path[0], tgt[0]), (path[-1], tgt[-1])
                    if jump_cut(prev, first) or repeats(first, last, earlier):
                        continue
                    depth = self.space.view_depth(path[0], math.atan2(*(aim[:2] - path[0])[::-1]), HFOV)
                    score = (0.3 * plen + 0.25 * min(depth, 6.0) + 0.3 * parallax(path, tgt)
                             + (OPENING_GLIDE if opening else 0.0) + (0.25 * min(depth, 6.0) if opening or closing else 0.0))
                    out.append(Move("glide", path, tgt, score, subject=box,
                                    why={"travel_m": round(plen, 2), "view_depth_m": round(depth, 2), "aim": why_aim}))
        return out

    def _pan(self, room, pts, rmin, dur, prev, earlier, closing, opening=False) -> list[Move]:
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
                    if turn <= self.turn_budget and move <= self.move_budget:
                        break
                    sweep *= 0.85
                else:
                    continue
                first, last = (eye[0], tgt[0]), (eye[-1], tgt[-1])
                if jump_cut(prev, first) or repeats(first, last, earlier):
                    continue
                depth = float(np.mean([sp.view_depth(e, h, HFOV) for e, h in zip(eye[::10], hs[::10])]))
                score = (0.25 * min(depth, 6.0) + 0.4 * sweep / PAN_SWEEP + (0.6 if closing else 0.0) - 0.3
                         + (0.25 * min(depth, 6.0) if opening or closing else 0.0))
                out.append(Move("pan", eye, tgt, score,
                                why={"sweep_deg": round(sweep), "view_depth_m": round(depth, 2),
                                     "turn_dps": round(turn, 1), "move_mps": round(move, 2)}))
        return out

    def _hold(self, least: Move, room: str) -> Move:
        """Every move runs into something near the lens: hold the least blocked start, with a breath
        of pull-back (never move into the blockage)."""
        # not a still (user 2026-11-16, Krk: a 9 s hold read as a frozen frame): from the held spot
        # the view turns HOLD_SWEEP deg, toward the side that shows more of the room
        p, t = least.first
        sp = self.space
        h0 = math.atan2(t[1] - p[1], t[0] - p[0])
        r = float(np.linalg.norm(t[:2] - p))
        side = max((1, -1), key=lambda sg: sp.own_share(room, p, h0 + sg * math.radians(HOLD_SWEEP), HFOV))
        s = np.linspace(0, 1, SAMPLES)
        hs = h0 + side * math.radians(HOLD_SWEEP) * (s * s * (3 - 2 * s))
        eye = np.repeat(np.asarray(p)[None], SAMPLES, 0)
        tgt = np.c_[p[0] + r * np.cos(hs), p[1] + r * np.sin(hs), np.full(SAMPLES, t[2])]
        m = Move("hold_pan", eye, tgt, least.score, least.pitch, least.piece, least.occ,
                 why={"held_because": "every move is blocked near the lens", "occlusion": round(least.occ, 3),
                      "sweep_deg": HOLD_SWEEP})
        return m

    def band_deficit(self, m: Move) -> float:
        """Degrees the view's top edge rises past where the scan's walls end (Space.wall_top), at
        its start, middle and end, across the frame - 0 when the scene has its ceiling."""
        sp = self.space
        top_map = getattr(sp, "wall_top", None)
        if top_map is None or "scene_ceiling_missing" not in getattr(sp, "warnings", []):
            return 0.0
        half_v = math.atan(math.tan(math.radians(HFOV) / 2) * FRAME_H / FRAME_W)
        offs = np.radians(np.array([-0.35, 0.0, 0.35]) * HFOV)
        e3 = m.eye3
        worst = 0.0
        for k in (0, len(e3) // 2, len(e3) - 1):
            e, t = e3[k], m.tgt[k]
            d = t - e
            hd = float(np.linalg.norm(d[:2]))
            head, pitch = math.atan2(d[1], d[0]), math.atan2(-d[2], max(hd, 1e-6))
            dist = sp.rays(e[:2], head + offs)
            for a, D in zip(offs, dist):
                hit = e[:2] + (D + 0.05) * np.array([math.cos(head + a), math.sin(head + a)])
                i, j = sp.cell(hit)
                if not (0 <= i < top_map.shape[0] and 0 <= j < top_map.shape[1]) or top_map[i, j] <= 0:
                    continue
                elev = math.atan(math.tan(half_v - pitch) * math.cos(a))
                need = math.atan2(float(top_map[i, j]) - BAND_MARGIN - e[2], max(float(D), 0.3))
                worst = max(worst, math.degrees(elev - need))
        return worst

    def _void(self, m: Move) -> float:
        """The worst share of the view the scan never captured (start, middle, end). A black band
        across the top (a missing ceiling, TOP_VOID_MAX of its top fifth) tilts the move down
        TOP_TILT deg; if that does not clear it (or loses the subject's floor) the move fails."""
        e3 = m.eye3
        ks = (0, len(e3) // 2, len(e3) - 1)
        # no ceiling: the frame's top edge stays under where the scan's walls end (geometric, the
        # thinned void render misses a thin band); a few degrees down, else the move goes
        deficit = self.band_deficit(m)
        if deficit > 0:
            if deficit > BAND_TILT_MAX or m.kind in ("detail", "link"):
                m.why["band_deficit_deg"] = round(deficit, 1)
                return 1.0
            old = m.tgt
            h = np.linalg.norm(m.tgt[:, :2] - m.eye, axis=1)
            m.tgt = m.tgt.copy()
            m.tgt[:, 2] -= h * math.tan(math.radians(deficit + 0.5))
            if not self.framed(m) or self.band_deficit(m) > 0:
                m.tgt = old
                m.why["band_deficit_deg"] = round(deficit, 1)
                return 1.0
            m.why["tilt_for_band"] = round(deficit + 0.5, 1)

        def worst(tgt):
            r = [self.occ.void(e3[k], tgt[k], HFOV, top=True) for k in ks]
            return max(a for a, _ in r), max(b for _, b in r)
        whole, top = worst(m.tgt)
        m.why["void_top"] = round(top, 3)
        if top <= TOP_VOID_MAX or m.kind in ("detail", "link"):
            return whole
        h = np.linalg.norm(m.tgt[:, :2] - m.eye, axis=1)
        for deg in TOP_TILT:
            tgt = m.tgt.copy()
            tgt[:, 2] -= h * math.tan(math.radians(deg))
            w2, t2 = worst(tgt)
            if t2 <= TOP_VOID_MAX:
                old = m.tgt
                m.tgt = tgt
                if self.framed(m):
                    m.why.update(void_top=round(t2, 3), tilt_for_void=deg)
                    return w2
                m.tgt = old
        return 1.0

    def _cover(self, m: Move) -> float:
        if self.occ is None:
            return 0.0
        e3 = m.eye3
        worst = 0.0
        for k in (len(e3) // 2, 0, len(e3) - 1):
            e, t = e3[k], m.tgt[k]
            worst = max(worst, self.occ.cover(e, t, HFOV, m.piece, OCC_NEAR_KIND.get(m.kind, OCC_NEAR)))
            if worst > MAX_OCC:                       # already out: the other poses cannot bring it back
                break
            if m.kind not in ("link", "detail", "exit"):     # (an exit flies past door frames)
                # a door frame or a wall at the frame's edge, right at the lens (user 2026-11-16, Krk:
                # the bedroom shot ended with the door frame over a quarter of the frame)
                edge = self.occ.cover(e, t, HFOV, m.piece, EDGE_NEAR, region="full")
                if edge > EDGE_MAX:
                    m.why["edge_cover"] = round(edge, 3)
                    return 1.0
        return worst
