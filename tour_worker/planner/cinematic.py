"""Cinematic camera engine (plan 1.3): the approved "v8" tour moves, planned in the worker (tour_plan job).

Ported from the tour worker's experiment planners (tools/tour_v6, tools/tour_v3, tools/tour_v2
at worker commit d2f2e5a), where they were tuned against reviewed renders. The worker now only
renders what this module plans.

Reveals start on part of the room and end on all of it. Push-ins (2026-10-16, review: "camera
motions are still a bit lacking" - 15 reveals in a row repeat one gesture) run a reveal
backwards: they start on the whole room and end on its most interesting part (scored by the END
window's saliency, so they never zoom at nothing). pull_back / slider / crane / pan can push;
the first exterior and the final shot always reveal. Selection alternates reveal and push (never
three alike running) as it alternates left / right travel. Variants, each mirrored (d):

  tilt_up      far low corner -> across and up to eye level -> whole room       (rotate)
  tilt_down    the same mirrored vertically                                     (rotate)
  pan          the same at one height                                           (rotate)
  pull_back    on the subject -> straight back with a slight drift -> whole room (rotate)
  slider       tight on one side -> glides across the room while easing out to the
               whole room; travel grows with coverage                           (crop)
  arc          (not selected; replaced by slider 2026-10-14) the aim circles once while
               closing in on the centre and pulling back (compound)             (rotate)
  crane        low -> eye level with a slight pull-back; lateral travel grows
               with coverage                                                    (crop)
  rise         exteriors: low -> as high as the photo allows, zooming out       (crop)
  aerial_approach  drone exteriors: starts on the whole view, slightly banked, and
               closes in on the frame centre while levelling out, like a drone
               coming in to the house                                           (crop + roll)
  aerial_orbit (not selected) the reverse: banked on the house, pulling back
  gimbal_track sideways glide at a constant zoom, level - a gimbal operator walking past
               the room, no zoom                                                (crop)
  vertical_glide  narrow but tall spaces (a bathroom vanity and mirror, a closet, a staircase):
               glides up (or down, to interest that sits low) the room to its interest while
               easing out a little - the vertical counterpart of the slider    (crop)
  gimbal_orbit (not selected) lateral sweep banking 1.8 deg into the turn; review 2026-10-19:
               the bank / tilt did not read ("i don't get the tilt and bank part")
  pedestal     (not selected) constant-zoom rise; review 2026-10-19: worse than the older
               rise / crane on the opening exteriors and the entry

Scene context (2026-10-19, planner/scene_context.py, from the OneFormer scene summary): each
variant's fit for the photo's content is added to its score (a tall door -> the crane reveal
rising to it, interest spread wide -> track / pan, one clear subject -> a push-in); no move
may end lower than it starts while the interest is above where it ends (a push that sank onto
the entry steps, review 2026-10-19); zoom-led moves (pull-backs, cranes, push-ins) are at most
about a third of the tour and never two in a row (review: "decrease the simple zoom in and zoom
out effects").

The one exception to "always zoom out" is aerials: the house is far away in them, so the camera
goes to it. Saliency cannot find the house in a drone shot (it peaks on roads, roof edges and the
frame border; measured on a close and a far aerial, 2026-10-14), so the approach targets the frame
centre, where listing photographers put the house.

"rotate" turns a virtual camera in place (exact homography of one photo, no depth); the worker
pre-warps those shots. A pure rotation cannot move through the room: dolly/orbit need depth.

Selection (small beam search): fit for the shot's category (aerial, exterior, bath, final, or
the photo's coverage class from scoring/motion.py), how interesting the opening frame is (the
coverage saliency grid), travel direction alternating shot to shot, no variant twice running,
and only variants that fit the shot's allocated length. Every frame of every chosen path is
checked: inside the photo by >= 1 output px, source density >= the floor, peak speed <= the cap
(the move is shrunk toward its end, the whole room, until it is).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from tour_worker.render.camera_path import (
    DEFAULT_HFOV_DEG,
    CompoundPath,
    KeyframePath,
    base_window,
    corner_rms_px,
    frame_margin_out_px,
    frame_spp,
    motion_px,
    roll_fit_scale,
    vec_angle,
)
from .scene_context import SceneContext
from .schemas import CameraParams, CameraPoint, Keyframe, Segment, Timing

# ---- limits ---------------------------------------------------------------------------------
MIN_MARGIN_OUT_PX = 1.0  # contract: window inside the photo at every frame
PLAN_MARGIN_OUT_PX = 2.0  # where keyframes are placed, so frames between have room
MIN_SPP = 0.75  # source px per output px, every frame
LOWRES_EXTRA_ZOOM = 1.25  # photos already below MIN_SPP at full view: at most 25% more zoom
ENFORCE_SHRINK_STEP = 0.10
REF_DIAG = math.hypot(1920, 1080)
V_MAX_REF = 3.3  # px/frame at 1080p: about the reviewed v3 rotate pace
V_CUT_REF = 1.6  # compound: px/frame at the first and last frame
TURN_COST = 60.0  # compound timing.turn (px of time per radian of turn)
SPEED_SHRINK_STEP = 0.05
HFOV_DEG = DEFAULT_HFOV_DEG
BEAM = 300

# ---- variant table --------------------------------------------------------------------------
SUIT = {
    "aerial": {"aerial_approach": 5.0},
    # exteriors keep the older moves (review 2026-10-19: the opening shots read better with them)
    "exterior": {"rise": 4.0, "tilt_up": 1.5, "pull_back": 1.0},
    "wide": {
        "gimbal_track": 2.5,
        "tilt_up": 3.0,
        "crane": 2.75,
        "slider": 2.75,
        "pan": 2.5,
        "tilt_down": 2.0,
        "pull_back": 1.0,
    },
    "medium": {
        "gimbal_track": 2.5,
        "vertical_glide": 1.5,
        "tilt_up": 2.5,
        "pan": 2.5,
        "crane": 2.25,
        "slider": 2.25,
        "tilt_down": 2.0,
        "pull_back": 1.5,
    },
    "tight": {"pull_back": 3.0, "vertical_glide": 2.75, "crane": 2.5, "gimbal_track": 2.25,
              "pan": 2.0, "tilt_up": 2.0, "tilt_down": 1.5},
    "bath": {"pull_back": 3.0, "vertical_glide": 3.0, "gimbal_track": 2.0, "pan": 2.0,
             "tilt_up": 1.5},
    "final": {"pull_back": 2.5, "tilt_up": 2.5, "gimbal_track": 2.0, "pan": 2.0,
              "tilt_down": 1.5},
}
HERO_BONUS = 0.5
PUSH_VARIANTS = {"pull_back", "slider", "crane", "pan"}
NO_PUSH_CATEGORIES = {"aerial", "exterior", "final"}   # open on the house, end on the whole room
# no bonus for alternating reveal / push for variety's sake (user 2026-10-16: each shot gets the
# move that fits its scene, not a rotation); a scene-fit score replaces this (camera plan step 3)
MODE_ALTERNATE_BONUS = 0.0
MODE_RUN_PENALTY = 0.0
MIN_TRAVEL_PX = 180.0           # at 1920 wide: a move travelling less reads as a still
STATIC_PENALTY = 1.5
TRAVELLING = {
    "tilt_up",
    "tilt_down",
    "pan",
    "arc",
    "crane",
    "rise",
    "aerial_orbit",
    "aerial_approach",
    "slider",
    "gimbal_track",
    "gimbal_orbit",
    "pedestal",
    "vertical_glide",
}
CROP_VARIANTS = {"crane", "rise", "aerial_orbit", "aerial_approach", "slider", "gimbal_track",
                 "gimbal_orbit", "pedestal", "vertical_glide"}
ZOOM_LED = {"pull_back", "crane"}   # + every push-in: the move IS the zoom
ZOOM_SHARE = 1 / 3                  # at most this share of the tour's shots zoom-led
ZOOM_OVER_PENALTY = 5.0          # above any fit bonus: the share is a cap, not a preference
ZOOM_RUN_PENALTY = 2.0
ZOOM_IN_VARIANTS = {"aerial_approach"}  # every other variant ends wider than it starts
COMPOUND_VARIANTS = {"arc"}
# shortest shot each variant still reads well in (shorter shots get a simpler move)
# (shorter, content-length shots 2026-10-19: normal shots run 2.8-3.6 s)
MIN_DURATION_S = {
    "arc": 5.5,
    "slider": 3.5,
    "aerial_orbit": 5.0,
    "aerial_approach": 3.0,
    "tilt_up": 3.8,
    "tilt_down": 3.8,
    "pan": 3.8,
    "crane": 2.8,
    "rise": 2.8,
    "pull_back": 2.5,
    "gimbal_track": 2.5,
    "gimbal_orbit": 3.2,
    "pedestal": 2.5,
    "vertical_glide": 2.5,
}
FALLBACK_VARIANT = "pull_back"
FALLBACK_BY_CATEGORY = {"aerial": "aerial_approach"}  # an aerial never pulls back

TILT = {"z0": 1.32, "t1": 0.62, "z1": 1.24, "z2": 1.012}
PULL = {"z0": 1.175, "z1": 1.02, "drift": 0.08}   # drift 0.03 read as standing still (~100 px)
ARC = {"z": (1.16, 1.14, 1.11, 1.07, 1.012), "r": (1.0, 0.95, 0.8, 0.5), "overlap": 0.18}
CRANE = {"z0": (1.10, 1.28), "z1": (1.035, 1.06), "rise": 0.16, "drift": (0.025, 0.11)}
AERIAL = {
    "z": (1.2, 1.1, 1.03),
    "t1": 0.55,
    "roll": (2.5, 1.0, 0.0),
    "toward_peak": 0.6,
    "drift": 0.03,
}
# aerial_approach: whole view (banked) -> halfway -> close on the centre, level; a slight drift
# slider: tight on side -d -> glides to side +d while easing out to near-full view (crop, so it
# reads as the camera moving sideways, like a slider or gimbal walk). Start zoom and travel grow
# with coverage like the crane's: lateral room in a crop is +-1/(2z) of the width.
SLIDER = {"z0": (1.16, 1.26), "z1": (1.03, 1.05), "t1": 0.5}
APPROACH = {"z": (1.02, 1.11, 1.24), "t1": 0.5, "roll": (2.0, 0.8, 0.0), "drift": 0.025}
RISE = {"z0": 1.10, "z1": 1.03, "x_toward_peak": 0.3}
TRACK = {"z": (1.15, 1.22), "roll": (0.0, 0.0, 0.0), "toward_interest": 0.5}   # level
ORBIT = {"z": ((1.20, 1.26), (1.16, 1.21), (1.10, 1.13)), "roll": (1.8, 0.9, 0.0),
         "lift": 0.02, "toward_interest": 0.4}
PEDESTAL = {"z": (1.16, 1.24), "x_toward_interest": 0.5}
VGLIDE = {"z0": (1.18, 1.26), "z1": (1.04, 1.07), "x_toward_interest": 0.4}


# ---- inputs ---------------------------------------------------------------------------------
@dataclass
class ShotInput:
    """What the engine needs per shot (built by planner.py from the allocation and scoring)."""

    shot_id: str
    src_w: int
    src_h: int
    room_type: str | None
    narrative_stage: str
    role: str
    duration_s: float
    peak_x: float = 0.5
    peak_y: float = 0.5
    spread: float = 0.4
    coverage_class: str = "medium"
    grid: list[list[float]] = field(default_factory=list)
    ctx: SceneContext | None = None     # what the photo shows (planner/scene_context.py)


@dataclass
class CinematicResult:
    cameras: list[CameraParams]
    warnings: list[str]
    notes: list[dict]


# ---- geometry -------------------------------------------------------------------------------
class Geometry:
    def __init__(self, src_w, src_h, out_w, out_h, projection="crop", hfov_deg=HFOV_DEG):
        self.src_w, self.src_h, self.out_w, self.out_h = src_w, src_h, out_w, out_h
        self.projection, self.hfov_deg = projection, hfov_deg
        self.bw, self.bh = base_window(src_w, src_h, out_w, out_h)
        full = self.bw * src_w / out_w
        self.min_spp = min(MIN_SPP, full / LOWRES_EXTRA_ZOOM)
        self.zoom_cap = full / self.min_spp

    def half(self, z, margin_px=PLAN_MARGIN_OUT_PX):
        hw, hh = self.bw / (2 * z), self.bh / (2 * z)
        return hw + margin_px * (self.bw / z) / self.out_w, hh + margin_px * (
            self.bh / z
        ) / self.out_h

    def clamp(self, cx, cy, z):
        hw, hh = self.half(z)
        return min(max(cx, hw), 1 - hw), min(max(cy, hh), 1 - hh)

    def margin(self, cx, cy, z):
        return frame_margin_out_px(
            cx,
            cy,
            z,
            self.src_w,
            self.src_h,
            self.out_w,
            self.out_h,
            self.projection,
            self.hfov_deg,
        )

    def spp(self, cx, cy, z):
        return frame_spp(
            cx,
            cy,
            z,
            self.src_w,
            self.src_h,
            self.out_w,
            self.out_h,
            self.projection,
            self.hfov_deg,
        )

    def fit(self, cx, cy, z):
        """Crop clamp, then (rotate reaches less far) the furthest point on the line to the centre
        that keeps the plan margin."""
        cx, cy = self.clamp(cx, cy, z)
        if self.margin(cx, cy, z) >= PLAN_MARGIN_OUT_PX:
            return cx, cy
        lo, hi = 0.0, 1.0
        for _ in range(40):
            m = (lo + hi) / 2
            if self.margin(0.5 + (cx - 0.5) * m, 0.5 + (cy - 0.5) * m, z) >= PLAN_MARGIN_OUT_PX:
                lo = m
            else:
                hi = m
        return 0.5 + (cx - 0.5) * lo, 0.5 + (cy - 0.5) * lo

    def width(self, z):
        return self.bw / z

    def height(self, z):
        return self.bh / z


def geometry_for(shot: ShotInput, variant: str, out_w: int, out_h: int) -> Geometry:
    proj = "crop" if variant in CROP_VARIANTS else "rotate"
    return Geometry(shot.src_w, shot.src_h, out_w, out_h, proj, HFOV_DEG)


def kf(t, cx, cy, z):
    return {"t": round(t, 4), "cx": round(cx, 6), "cy": round(cy, 6), "zoom": round(z, 6)}


def _state(cx, cy, z):
    return {"cx": round(cx, 6), "cy": round(cy, 6), "zoom": round(z, 6)}


def _seg(t0, t1, st):
    return {"t0": t0, "t1": t1, **st}


# ---- move builders --------------------------------------------------------------------------
def _tilt_reveal(g, peak_y, d):
    p = TILT
    cx0, cy0 = g.fit(0.5 - d * 0.5, 1.0, p["z0"])
    cx1, cy1 = g.fit(0.5 + d * 0.5, min(peak_y, 0.5), p["z1"])
    edge_x = cx1 + d * g.width(p["z1"]) / 2
    cx2, cy2 = g.fit(edge_x - d * g.width(p["z2"]) / 2, 0.5, p["z2"])
    return [kf(0, cx0, cy0, p["z0"]), kf(p["t1"], cx1, cy1, p["z1"]), kf(1, cx2, cy2, p["z2"])]


def _crane_coverage(shot: ShotInput) -> float:
    """0 = small-room crane (v7, gentle drift) .. 1 = widest rooms (slides ~16% of the width)."""
    if shot.coverage_class == "wide":
        return 0.5 + 0.5 * max(0.0, min(1.0, (shot.spread - 0.42) / 0.08))
    return 0.25 if shot.coverage_class == "medium" else 0.0


def variant_keyframes(variant: str, g: Geometry, shot: ShotInput, d: int, out_w: int, out_h: int):
    if variant == "tilt_up":
        return _tilt_reveal(g, shot.peak_y, d)
    if variant == "tilt_down":
        return [dict(k, cy=round(1 - k["cy"], 6)) for k in _tilt_reveal(g, 1 - shot.peak_y, d)]
    if variant == "pan":
        kfs = _tilt_reveal(g, shot.peak_y, d)
        kfs[0] = dict(kfs[0], cy=kfs[1]["cy"])
        return kfs
    if variant == "pull_back":
        p = PULL
        cx0, cy0 = g.clamp(shot.peak_x, shot.peak_y, p["z0"])
        cx0, cy0 = g.fit(cx0 - d * p["drift"], cy0, p["z0"])
        cx1, cy1 = g.clamp(0.5, 0.5, p["z1"])
        return [kf(0, cx0, cy0, p["z0"]), kf(1, cx1, cy1, p["z1"])]
    if variant == "crane":
        c, f = CRANE, _crane_coverage(shot)
        z0 = c["z0"][0] + f * (c["z0"][1] - c["z0"][0])
        drift = c["drift"][0] + f * (c["drift"][1] - c["drift"][0])
        z1 = c["z1"][0] + f * (c["z1"][1] - c["z1"][0])
        cx0, cy0 = g.fit(0.5 - d * drift, 0.5 + c["rise"] / 2 + 0.04, z0)
        cx1, cy1 = g.fit(0.5 + d * drift, 0.5 - c["rise"] / 2 + 0.04, z1)
        return [kf(0, cx0, cy0, z0), kf(1, cx1, cy1, z1)]
    if variant == "aerial_orbit":
        a = AERIAL
        z0 = min(a["z"][0], g.zoom_cap / roll_fit_scale(math.radians(a["roll"][0]), out_w, out_h))
        zs = (z0, min(a["z"][1], z0), a["z"][2])
        px = 0.5 + a["toward_peak"] * (shot.peak_x - 0.5)
        py = 0.5 + a["toward_peak"] * (shot.peak_y - 0.5)
        pts = [
            (px - d * a["drift"], py),
            ((px + 0.5) / 2, (py + 0.5) / 2),
            (0.5 + d * a["drift"], 0.5),
        ]
        out = []
        for k, ((cx, cy), z, r) in enumerate(zip(pts, zs, a["roll"], strict=False)):
            cx, cy = g.fit(cx, cy, z)
            point = kf((0, a["t1"], 1)[k], cx, cy, z)
            point["roll_deg"] = round(-d * r, 4)
            out.append(point)
        return out
    if variant == "slider":
        c, f = SLIDER, _crane_coverage(shot)
        z0 = c["z0"][0] + f * (c["z0"][1] - c["z0"][0])
        z2 = c["z1"][0] + f * (c["z1"][1] - c["z1"][0])
        z1 = math.sqrt(z0 * z2)
        cy = 0.5 + 0.25 * (shot.peak_y - 0.5)  # at the room's interesting height, gently
        cx0, cy0 = g.fit(0.5 - d * 0.5, cy, z0)  # as far to side -d as the zoom allows
        cx2, cy2 = g.fit(0.5 + d * 0.5, 0.5, z2)  # as far to side +d as the end zoom allows
        cx1, cy1 = g.fit((cx0 + cx2) / 2, (cy0 + cy2) / 2, z1)
        return [kf(0, cx0, cy0, z0), kf(c["t1"], cx1, cy1, z1), kf(1, cx2, cy2, z2)]
    if variant == "aerial_approach":
        a = APPROACH
        z_end = min(a["z"][2], g.zoom_cap)
        zs = (a["z"][0], min(a["z"][1], z_end), z_end)
        pts = [(0.5 - d * a["drift"], 0.5), (0.5 - d * a["drift"] / 2, 0.5), (0.5, 0.5)]
        out = []
        for k, ((cx, cy), z, r) in enumerate(zip(pts, zs, a["roll"], strict=True)):
            cx, cy = g.fit(cx, cy, z)
            point = kf((0, a["t1"], 1)[k], cx, cy, z)
            point["roll_deg"] = round(-d * r, 4)
            out.append(point)
        return out
    if variant == "vertical_glide":
        c, f = VGLIDE, _crane_coverage(shot)
        z0 = c["z0"][0] + f * (c["z0"][1] - c["z0"][0])
        z2 = c["z1"][0] + f * (c["z1"][1] - c["z1"][0])
        z1 = math.sqrt(z0 * z2)
        ix, iy = _interest(shot)
        cx = 0.5 + c["x_toward_interest"] * (ix - 0.5)
        top = shot.ctx.top_y if shot.ctx else iy
        if iy <= 0.6:                                   # rise from the floor to the interest
            p0 = g.fit(cx, 1.0, z0)
            p2 = g.fit(0.5 + 0.5 * (cx - 0.5), min(iy, (top + iy) / 2 + 0.1), z2)
        else:                                           # interest sits low: come down to it
            p0 = g.fit(cx, 0.0, z0)
            p2 = g.fit(0.5 + 0.5 * (cx - 0.5), iy, z2)
        p1 = g.fit((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2, z1)
        return [kf(0, *p0, z0), kf(0.5, *p1, z1), kf(1, *p2, z2)]
    if variant in ("gimbal_track", "gimbal_orbit", "pedestal"):
        return _gimbal_keyframes(variant, g, shot, d)
    if variant == "rise":
        r = RISE
        cx = 0.5 + r["x_toward_peak"] * (shot.peak_x - 0.5)
        cx0, cy0 = g.fit(cx, 1.0, r["z0"])
        cx1, cy1 = g.fit(cx, 0.0, r["z1"])
        return [kf(0, cx0, cy0, r["z0"]), kf(1, cx1, cy1, r["z1"])]
    raise ValueError(variant)


def _interest(shot: ShotInput) -> tuple[float, float]:
    return (shot.ctx.interest_x, shot.ctx.interest_y) if shot.ctx else (shot.peak_x, shot.peak_y)


def _gimbal_keyframes(variant: str, g: Geometry, shot: ShotInput, d: int) -> list[dict]:
    f = _crane_coverage(shot)
    ix, iy = _interest(shot)
    if variant == "gimbal_track":
        c = TRACK
        z = c["z"][0] + f * (c["z"][1] - c["z"][0])
        cy = 0.5 + c["toward_interest"] * (iy - 0.5)
        pts = [g.fit(0.5 - d * 0.5, cy, z), None, g.fit(0.5 + d * 0.5, cy, z)]
        pts[1] = ((pts[0][0] + pts[2][0]) / 2, cy)
        zs = (z, z, z)
        rolls = c["roll"]
    elif variant == "gimbal_orbit":
        c = ORBIT
        zs = tuple(a + f * (b - a) for a, b in c["z"])
        cy = 0.5 + c["toward_interest"] * (iy - 0.5)
        mx = 0.5 + c["toward_interest"] * (ix - 0.5)
        pts = [g.fit(0.5 - d * 0.5, cy, zs[0]), g.fit(mx, cy - c["lift"], zs[1]),
               g.fit(0.5 + d * 0.5, cy, zs[2])]
        rolls = c["roll"]
    else:                                                   # pedestal: rise to the interest
        c = PEDESTAL
        z = c["z"][0] + f * (c["z"][1] - c["z"][0])
        cx = 0.5 + c["x_toward_interest"] * (ix - 0.5)
        top = shot.ctx.top_y if shot.ctx else iy
        target = min(iy, (top + iy) / 2 + g.height(z) / 4)
        p0, p2 = g.fit(cx, 1.0, z), g.fit(cx, target, z)
        return [kf(0, *p0, z), kf(1, *p2, z)]
    # a banked frame is scaled up by roll_fit_scale: keep zoom x that inside the photo's
    # resolution (1536 px listing photos leave only ~1.25x), keeping the zoom non-increasing
    capped, prev_z = [], math.inf
    for z, r in zip(zs, rolls, strict=True):
        z = min(z, prev_z, g.zoom_cap / roll_fit_scale(math.radians(r), g.out_w, g.out_h) - 1e-3)
        capped.append(z)
        prev_z = z
    out = []
    for k, ((cx, cy_), z, r) in enumerate(zip(pts, capped, rolls, strict=True)):
        cx, cy_ = g.fit(cx, cy_, z)
        point = kf((0, 0.5, 1)[k], cx, cy_, z)
        if r:
            point["roll_deg"] = round(-d * r, 4)
        out.append(point)
    return out


def arc_camera(g: Geometry, d: int):
    """Spiral (compound): the aim circles once (side -> top -> other side -> bottom) while the
    circle closes on the centre and the camera pulls back to the whole room."""
    z, r, ov = ARC["z"], ARC["r"], ARC["overlap"]
    pts = []
    for k, ang in enumerate((180, 90, 0, -90)):
        rx = g.fit(1.0, 0.5, z[k])[0] - 0.5
        ry = g.fit(0.5, 1.0, z[k])[1] - 0.5
        a = math.radians(ang)
        ox = math.cos(a) if k < 3 else 0.0
        pts.append(
            _state(*g.fit(0.5 + d * r[k] * rx * ox, 0.5 - r[k] * ry * math.sin(a), z[k]), z[k])
        )
    end = _state(*g.fit(0.5, 0.5, z[-1]), z[-1])
    win = [(0, 0.25 + ov), (0.25 - ov, 0.5 + ov), (0.5 - ov, 0.75 + ov), (0.75 - ov, 1)]
    segs = [
        _seg(round(max(0.0, a), 3), round(min(1.0, b), 3), p)
        for (a, b), p in zip(win, pts[1:] + [end], strict=False)
    ]
    return pts[0], segs


# ---- measurement and fitting ----------------------------------------------------------------
def _speed_scale(out_w, out_h):
    return math.hypot(out_w, out_h) / REF_DIAG


def _kf_frames(kfs, n):
    return KeyframePath(kfs).frames(n)


def _stats(frames, g):
    sp = [
        corner_rms_px(a, b, g.src_w, g.src_h, g.out_w, g.out_h)
        for a, b in zip(frames, frames[1:], strict=False)
    ] or [0.0]
    v = [
        motion_px(a, b, g.src_w, g.src_h, g.out_w, g.out_h)
        for a, b in zip(frames, frames[1:], strict=False)
    ]
    turn = [math.degrees(vec_angle(a, b)) for a, b in zip(v, v[1:], strict=False)] or [0.0]
    return {
        "speed_first_px_f": round(sp[0], 3),
        "speed_max_px_f": round(max(sp), 3),
        "speed_min_px_f": round(min(sp), 3),
        "speed_cut_px_f": round(sp[-1], 3),
        "turn_max_deg_f": round(max(turn), 3),
    }


def enforce(keyframes, n, g):
    """Fit every keyframe inside the projection's reach, cap zoom at the density floor (on every
    frame for rotate), then shrink toward the centre until every frame keeps the 1 px margin."""
    keyframes = copy.deepcopy(keyframes)
    for k in keyframes:
        k["cx"], k["cy"] = (round(v, 6) for v in g.fit(k["cx"], k["cy"], k["zoom"]))
    for k in keyframes:
        if k["zoom"] > g.zoom_cap:
            k["zoom"] = round(g.zoom_cap, 6)
            k["cx"], k["cy"] = (round(v, 6) for v in g.fit(k["cx"], k["cy"], k["zoom"]))
    for _ in range(40):
        if min(g.spp(cx, cy, z) for cx, cy, z in _kf_frames(keyframes, n)) >= g.min_spp - 1e-9:
            break
        for k in keyframes:
            if k["zoom"] > 1.05:
                k["zoom"] = round(k["zoom"] * 0.99, 6)
                k["cx"], k["cy"] = (round(v, 6) for v in g.fit(k["cx"], k["cy"], k["zoom"]))
    scale, base = 1.0, copy.deepcopy(keyframes)
    while min(g.margin(cx, cy, z) for cx, cy, z in _kf_frames(keyframes, n)) < MIN_MARGIN_OUT_PX:
        scale = round(scale - ENFORCE_SHRINK_STEP, 3)
        if scale <= 0:
            raise ValueError("path cannot be made to fit")
        keyframes = [
            dict(
                k,
                cx=round(0.5 + (b["cx"] - 0.5) * scale, 6),
                cy=round(0.5 + (b["cy"] - 0.5) * scale, 6),
            )
            for k, b in zip(keyframes, base, strict=False)
        ]
    return keyframes


def cap_speed(kfs, n, g, vmax, step, anchor=-1):
    """Shrink the move toward its anchor keyframe (the whole room: a reveal's end, a push-in's
    start) until no frame moves faster than vmax."""
    end = kfs[anchor]
    f, out = 1.0, kfs
    while _stats(_kf_frames(out, n), g)["speed_max_px_f"] > vmax and f > step:
        f = round(f - step, 3)
        out = [
            dict(
                k,
                cx=round(end["cx"] + f * (k["cx"] - end["cx"]), 6),
                cy=round(end["cy"] + f * (k["cy"] - end["cy"]), 6),
                zoom=round(end["zoom"] * (k["zoom"] / end["zoom"]) ** f, 6),
            )
            for k in kfs
        ]
        out = enforce(out, n, g)
    return out, f


def _make_compound(g, start, segs, timing):
    return CompoundPath(start, segs, timing, g.src_w, g.src_h, g.out_w, g.out_h)


def fit_compound(g, start, segs, n, vmax, v_cut, step):
    """Shrink a compound move toward its end until its peak speed is <= vmax and every frame
    keeps the margin and density floor. Timing: enters and leaves at v_cut (or slower)."""
    end = segs[-1]
    f = 1.0
    while True:

        def sc(st, f=f):
            return {
                **st,
                **_state(
                    end["cx"] + f * (st["cx"] - end["cx"]),
                    end["cy"] + f * (st["cy"] - end["cy"]),
                    end["zoom"] * (st["zoom"] / end["zoom"]) ** f,
                ),
            }

        st2 = sc(start)
        segs2 = [{**sc(s), "t0": s["t0"], "t1": s["t1"]} for s in segs[:-1]] + [end]
        raw = _make_compound(g, st2, segs2, {"s0": 1, "s1": 1, "turn": TURN_COST})
        mean = raw.cost / max(1, n - 1)
        s = min(1.0, v_cut / mean) if mean > 0 else 1.0
        timing = {"s0": round(s, 4), "s1": round(s, 4), "turn": TURN_COST}
        frames = _make_compound(g, st2, segs2, timing).frames(n)
        st = _stats(frames, g)
        ok = (
            st["speed_max_px_f"] <= vmax + 1e-6
            and min(g.margin(*fr) for fr in frames) >= MIN_MARGIN_OUT_PX
            and min(g.spp(*fr) for fr in frames) >= g.min_spp - 1e-9
        )
        if ok or f <= step:
            return st2, segs2, timing, ok
        f = round(f - step, 3)


def _rolled_density_ok(kfs, n, g):
    kp = KeyframePath(kfs)
    return (
        min(
            g.spp(cx, cy, z) / roll_fit_scale(math.radians(r), g.out_w, g.out_h)
            for (cx, cy, z), r in zip(kp.frames(n), kp.rolls(n), strict=False)
        )
        >= g.min_spp - 1e-6
    )


# ---- selection ------------------------------------------------------------------------------
def category(shot: ShotInput, is_last: bool) -> str:
    rt = (shot.room_type or "").lower()
    if rt == "aerial":
        return "aerial"
    if is_last:
        return "final"
    if shot.narrative_stage == "exterior_open" or rt in ("exterior", "front_exterior", "facade"):
        return "exterior"
    if "bath" in rt or rt in ("ensuite", "powder_room"):
        return "bath"
    return shot.coverage_class if shot.coverage_class in ("wide", "medium", "tight") else "medium"


def window_interest(grid, g, st) -> float:
    """Mean saliency of the grid cells whose centres fall in the window / the photo's mean."""
    if not grid:
        return 1.0
    rows, cols = len(grid), len(grid[0])
    ww, wh = g.width(st["zoom"]), g.height(st["zoom"])
    x0, x1 = st["cx"] - ww / 2, st["cx"] + ww / 2
    y0, y1 = st["cy"] - wh / 2, st["cy"] + wh / 2
    total = sum(sum(r) for r in grid) / (rows * cols) or 1e-9
    cells = [
        grid[i][j]
        for i in range(rows)
        for j in range(cols)
        if x0 <= (j + 0.5) / cols <= x1 and y0 <= (i + 0.5) / rows <= y1
    ]
    return (sum(cells) / len(cells)) / total if cells else 1.0


def reverse(kfs: list[dict]) -> list[dict]:
    """The same path run backwards (a reveal becomes a push-in)."""
    return [dict(k, t=round(1.0 - k["t"], 6)) for k in reversed(kfs)]


def travel_px(kfs: list[dict], g: Geometry) -> float:
    """Rough on-screen travel of a keyframe path (pan + zoom), output px."""
    tot = 0.0
    for a, b in zip(kfs, kfs[1:], strict=False):
        z = math.sqrt(a["zoom"] * b["zoom"])
        tot += math.hypot((b["cx"] - a["cx"]) * g.out_w / g.width(z),
                          (b["cy"] - a["cy"]) * g.out_h / g.height(z))
        tot += abs(math.log(b["zoom"] / a["zoom"])) * g.out_w / 2
    return tot


def lateral_travel(kfs) -> int:
    dx = kfs[1]["cx"] - kfs[0]["cx"] if len(kfs) > 2 else kfs[-1]["cx"] - kfs[0]["cx"]
    return 1 if dx > 1e-4 else -1 if dx < -1e-4 else 0


def context_fit(variant: str, mode: str, shot: ShotInput) -> float:
    """How well a move suits what the photo shows (0 without a scene context)."""
    c = shot.ctx
    if c is None:
        return 0.0
    wide = min(1.0, c.h_extent / 0.5)
    fit = 0.0
    if mode == "push":
        fit += 2.0 * (c.dominance - 0.3)       # one clear subject earns a push-in
    if variant == "gimbal_track":
        fit += 2.0 * wide
    elif variant == "gimbal_orbit":
        fit += 1.5 * c.richness + 1.0 * c.open_floor
    elif variant == "crane" and mode == "reveal":   # rises to a door / from the steps
        fit += (2.0 if c.rises_to_interest else 0.0) + min(1.0, c.v_extent / 0.4)
        fit -= 1.5 if c.interest_y > 0.6 else 0.0
    elif variant == "tilt_up":
        fit += 1.0 if (c.high_interest or c.rises_to_interest) else 0.0
    elif variant == "tilt_down":
        fit -= 0.0 if c.interest_y > 0.6 else 1.5
    elif variant in ("pan", "slider"):
        fit += 1.0 * wide
    elif variant == "vertical_glide":
        # interest that runs about as tall as it is wide or taller - a narrow space with its
        # content stacked up (a vanity under a mirror, a shower, shelves): Turkey bathroom 1.07,
        # a wide guest bedroom 0.38
        ratio = c.v_extent / max(c.h_extent, 0.1)
        fit += 3.0 * min(1.0, max(0.0, ratio - 0.5)) + (0.5 if c.tall_subject else 0.0) - 1.0
    return round(fit, 4)


def descends_from_interest(kfs: list[dict], shot: ShotInput, cat: str = "") -> bool:
    """The view ends lower than it starts while the interest is above where it ends (a push
    sinking onto floor / paving). Aerials are exempt: a drone coming down toward the house is the
    point of the shot (user 2026-10-19: "aerial shots might need to go lower")."""
    if not kfs or cat == "aerial":
        return False
    _, iy = _interest(shot)
    return kfs[-1]["cy"] - kfs[0]["cy"] > 0.03 and iy < kfs[-1]["cy"] - 0.02


def candidates(shot: ShotInput, cat: str, out_w: int, out_h: int) -> list[dict]:
    out = []
    for variant, suit in SUIT[cat].items():
        if shot.duration_s + 1e-6 < MIN_DURATION_S[variant] and variant != FALLBACK_VARIANT:
            continue
        g = geometry_for(shot, variant, out_w, out_h)
        if shot.role == "hero" and variant in TRAVELLING:
            suit += HERO_BONUS
        for d in (1, -1):
            if variant in COMPOUND_VARIANTS:
                opening, _ = arc_camera(g, d)
                kfs, travel, moves = None, d, MIN_TRAVEL_PX
            else:
                kfs = variant_keyframes(variant, g, shot, d, out_w, out_h)
                opening, travel, moves = kfs[0], lateral_travel(kfs), travel_px(kfs, g)
            interest = 1.5 * max(-1.0, min(1.0, window_interest(shot.grid, g, opening) - 1))
            base = {"variant": variant, "d": d, "category": cat,
                    "static": moves < MIN_TRAVEL_PX * out_w / 1920}
            if not descends_from_interest(kfs, shot, cat):
                out.append({**base, "suit": suit + context_fit(variant, "reveal", shot),
                            "keyframes": kfs, "interest": round(interest, 4), "travel": travel,
                            "mode": "reveal"})
            if kfs and variant in PUSH_VARIANTS and cat not in NO_PUSH_CATEGORIES:
                pk = reverse(kfs)          # whole room -> its interesting part (the END window)
                if descends_from_interest(pk, shot, cat):
                    continue
                end_interest = 1.5 * max(-1.0, min(1.0, window_interest(shot.grid, g, pk[-1]) - 1))
                out.append({**base, "suit": suit + context_fit(variant, "push", shot),
                            "keyframes": pk, "interest": round(end_interest, 4),
                            "travel": lateral_travel(pk), "mode": "push"})
    if not out:                                   # every move sank: the plain reveal still fits
        fb = FALLBACK_BY_CATEGORY.get(cat, FALLBACK_VARIANT)
        g = geometry_for(shot, fb, out_w, out_h)
        kfs = variant_keyframes(fb, g, shot, 1, out_w, out_h)
        out.append({"variant": fb, "d": 1, "category": cat, "static": False,
                    "suit": 0.0, "keyframes": kfs, "interest": 0.0, "travel": lateral_travel(kfs),
                    "mode": "reveal"})
    return out


def zoom_led(c: dict) -> bool:
    return c["variant"] in ZOOM_LED or c.get("mode") == "push"


def select(cands: list[list[dict]]) -> list[dict]:
    zoom_cap = max(1, math.ceil(ZOOM_SHARE * len(cands)))
    beams = [(0.0, [], {})]
    for i, cs in enumerate(cands):
        nxt = []
        for score, seq, uses in beams:
            for ci, c in enumerate(cs):
                s = score + c["suit"] + c["interest"] - (STATIC_PENALTY if c["static"] else 0.0)
                if seq:
                    prev = cands[i - 1][seq[-1]]
                    if c["mode"] != prev["mode"]:
                        s += MODE_ALTERNATE_BONUS
                    elif len(seq) > 1 and cands[i - 2][seq[-2]]["mode"] == c["mode"]:
                        s -= MODE_RUN_PENALTY
                    if c["travel"] and c["travel"] == prev["travel"]:
                        s -= 2.0
                    if c["variant"] == prev["variant"]:
                        s -= 2.5
                    if len(seq) > 1 and cands[i - 2][seq[-2]]["variant"] == c["variant"]:
                        s -= 1.0
                if zoom_led(c):
                    if seq and zoom_led(cands[i - 1][seq[-1]]):
                        s -= ZOOM_RUN_PENALTY
                    if uses.get("zoom", 0) + 1 > zoom_cap:
                        s -= ZOOM_OVER_PENALTY
                s -= 1.0 * uses.get((c["variant"], c["d"]), 0)
                s -= 1.5 * max(0, uses.get(c["variant"], 0) - 2)
                u = dict(uses)
                u[(c["variant"], c["d"])] = u.get((c["variant"], c["d"]), 0) + 1
                u[(c["variant"], c["mode"])] = u.get((c["variant"], c["mode"]), 0) + 1
                u[c["variant"]] = u.get(c["variant"], 0) + 1
                if zoom_led(c):
                    u["zoom"] = u.get("zoom", 0) + 1
                nxt.append((s, seq + [ci], u))
        nxt.sort(key=lambda b: (-b[0], b[1]))
        beams = nxt[:BEAM]
    return [cands[i][ci] for i, ci in enumerate(beams[0][1])]


# ---- camera for one chosen candidate --------------------------------------------------------
def _camera(
    c: dict, shot: ShotInput, n: int, out_w: int, out_h: int, speed: float, note: dict
) -> CameraParams:
    g = geometry_for(shot, c["variant"], out_w, out_h)
    k = _speed_scale(out_w, out_h)
    push = c.get("mode") == "push"
    move = ("push_in" if c["variant"] == "pull_back" else f"push_{c['variant']}") if push \
        else f"reveal_{c['variant']}"
    proj = {"projection": "rotate", "hfov_deg": HFOV_DEG} if g.projection == "rotate" else {}
    if c["variant"] in COMPOUND_VARIANTS:
        start, segs = arc_camera(g, c["d"])
        start, segs, timing, ok = fit_compound(
            g, start, segs, n, V_MAX_REF * k, V_CUT_REF * k, SPEED_SHRINK_STEP
        )
        if speed < 1:
            ref = _stats(_make_compound(g, start, segs, timing).frames(n), g)["speed_max_px_f"]
            start, segs = arc_camera(g, c["d"])
            start, segs, timing, ok = fit_compound(
                g, start, segs, n, speed * ref, V_CUT_REF * k, 0.01
            )
        if not ok:
            raise ValueError("compound path cannot be made to fit")
        note.update(_stats(_make_compound(g, start, segs, timing).frames(n), g))
        return CameraParams(
            type="compound",
            move=move,
            start=CameraPoint(**start),
            segments=[Segment(**s) for s in segs],
            timing=Timing(**timing),
            **proj,
        )
    anchor = 0 if push else -1
    kfs = enforce(c["keyframes"], n, g)
    kfs, f = cap_speed(kfs, n, g, V_MAX_REF * k, SPEED_SHRINK_STEP, anchor)
    if speed < 1:
        ref = _stats(_kf_frames(kfs, n), g)["speed_max_px_f"]
        kfs, f2 = cap_speed(kfs, n, g, speed * ref, 0.01, anchor)
        f *= f2
    if any(p.get("roll_deg") for p in kfs) and not _rolled_density_ok(kfs, n, g):
        raise ValueError("rolled frames fall below the density floor")
    note.update(_stats(_kf_frames(kfs, n), g))
    note["move_scale"] = round(f, 4)
    return CameraParams(
        type="keyframe_path", move=move, keyframes=[Keyframe(**p) for p in kfs], **proj
    )


def plan_cameras(
    shots: list[ShotInput], frames: list[int], out_w: int, out_h: int, speed: float = 1.0
) -> CinematicResult:
    """One camera per shot. frames[i] is the shot's rendered frame count (worker timeline)."""
    cats = [category(s, i == len(shots) - 1) for i, s in enumerate(shots)]
    cands = [candidates(s, c, out_w, out_h) for s, c in zip(shots, cats, strict=False)]
    chosen = select(cands)
    cameras, warnings, notes = [], [], []
    for shot, c, n in zip(shots, chosen, frames, strict=False):
        note = {
            "shot_id": shot.shot_id,
            "category": c["category"],
            "variant": c["variant"],
            "d": c["d"],
            "travel": c["travel"],
            "mode": c.get("mode", "reveal"),
            "duration_s": shot.duration_s,
            "frames": n,
        }
        try:
            cam = _camera(c, shot, n, out_w, out_h, speed, note)
        except ValueError as e:
            # never fail a tour on one shot: a gentle pull-back always fits (aerials approach)
            warnings.append(f"cinematic_fallback:{shot.shot_id}:{c['variant']}")
            note["fallback_reason"] = str(e)
            fb_variant = FALLBACK_BY_CATEGORY.get(c["category"], FALLBACK_VARIANT)
            g = geometry_for(shot, fb_variant, out_w, out_h)
            note["mode"] = "reveal"
            fb = {
                "variant": fb_variant,
                "d": c["d"],
                "mode": "reveal",
                "keyframes": variant_keyframes(fb_variant, g, shot, c["d"], out_w, out_h),
            }
            note["variant"] = fb_variant
            cam = _camera(fb, shot, n, out_w, out_h, speed, note)
        cameras.append(cam)
        notes.append(note)
    return CinematicResult(cameras=cameras, warnings=warnings, notes=notes)
