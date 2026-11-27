"""Cut continuity (client review 2026-11-20, items 1-9): how each shot enters and leaves the frame,
and whether every cut keeps the camera moving as one continuous move.

Every shot's entry and exit are its on-screen motion, in percent of the frame width per second
(%fw/s), the same unit for photo and 3D tours:
  x  lateral: + the camera moves (or turns) right - the picture flows left
  y  vertical: + the camera moves (or tilts) up
  f  forward: + the camera moves in (or zooms in) - the picture's edges flow outward; measured as
     the speed of the frame's side edge
Photo shots (render/frames.camera_frames): the crop window's travel over its own width, and the
zoom's log rate x 50. 3D shots (scene/camera.BezierCameraPath): the camera's velocity in its own
frame, seen at a reference depth REF_DEPTH_M, plus its turn rate over the field of view. Measured
over the MEASURE_S at each end of the shot's visible part (a fade's overlap excluded).

Each cut (shot k -> k+1, a hard cut) is checked against LIMITS:
  speed     incoming within 25% of outgoing (max / min <= SPEED_RATIO); both ends still is a
            "stopped" cut (the camera halts at the cut)
  direction the lateral direction turns by at most MAX_TURN_DEG; forward never meets backward
            (a push-in into a pull-out). Either is allowed at a change of part of the home,
            recorded on the incoming shot (Shot.cut_reason; reported as the row's "reset")
  beat      the cut lands within BEAT_TOL_FRAMES of the beat the sync chose for it
            (plan.music.cut_beats_s; plans without it are not judged on this)
Fades (daylight <-> twilight, a drone clip into the house) and video clips are not cuts on motion:
reported, not judged. report() is information; violations() lists what breaks the limits.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

MEASURE_S = 0.2            # each end of a shot is measured over this long
REF_DEPTH_M = 3.0          # 3D: translation seen at this distance (a room's middle distance)
SPEED_RATIO = 1.25         # incoming within 25% of outgoing (item 4)
MAX_TURN_DEG = 45.0        # lateral direction change across a cut (item 5)
STILL = 0.5                # %fw/s: under this a shot end is still
SIGNIFICANT = 0.25         # a component counts when it carries this share of the end's speed
BEAT_TOL_FRAMES = 1        # the cut within this many frames of its beat (after the cut lead)


@dataclass
class Motion:
    x: float
    y: float
    f: float

    @property
    def speed(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.f * self.f)

    @property
    def lateral(self) -> float:
        return math.hypot(self.x, self.y)

    def direction(self) -> str:
        """The dominant direction(s), e.g. "right", "forward+left", "still"."""
        if self.speed < STILL:
            return "still"
        parts = []
        if abs(self.f) >= SIGNIFICANT * self.speed:
            parts.append("forward" if self.f > 0 else "back")
        if self.lateral >= SIGNIFICANT * self.speed:
            a = math.degrees(math.atan2(self.y, self.x)) % 360
            parts.append(("right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right")[
                int(((a + 22.5) % 360) // 45)])
        return "+".join(parts) or "still"

    def rounded(self) -> dict:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "f": round(self.f, 2), "speed": round(self.speed, 2),
                "direction": self.direction()}


# --------------------------------------------------------------------------------------------
# measuring one shot
# --------------------------------------------------------------------------------------------

def _photo_motion(frames: list, a: int, b: int, fps: float, src_w: int, src_h: int, out_w: int, out_h: int) -> Motion:
    """Mean on-screen motion between frames a and b (inclusive) of a photo shot."""
    from tour_worker.render.camera_path import base_window
    bw, _ = base_window(src_w, src_h, out_w, out_h)
    if b <= a:
        return Motion(0.0, 0.0, 0.0)
    (cx0, cy0, z0), _ = frames[a]
    (cx1, cy1, z1), _ = frames[b]
    dt = (b - a) / fps
    zm = math.sqrt(z0 * z1)
    win_w = bw / zm                                    # the window's width, share of the photo width
    x = (cx1 - cx0) / win_w / dt * 100.0
    y = -(cy1 - cy0) * src_h / (win_w * src_w) / dt * 100.0     # photo y runs down; + is up
    f = math.log(z1 / z0) / dt * 50.0
    return Motion(x, y, f)


def _scene_motion(path, n: int, a: int, b: int, fps: float, up, path_s=None) -> Motion:
    """Mean on-screen motion between frames a and b of a 3D shot (bezier_path)."""
    if b <= a:
        return Motion(0.0, 0.0, 0.0)
    e0, t0, h0 = path.pose_at_frame(a, n, fps, path_s)
    e1, t1, h1 = path.pose_at_frame(b, n, fps, path_s)
    dt = (b - a) / fps
    upv = np.asarray(up, np.float64)
    fwd = (t0 + t1) / 2 - (e0 + e1) / 2
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.cross(fwd, upv)
    right /= max(np.linalg.norm(right), 1e-9)
    cam_up = np.cross(right, fwd)
    v = (e1 - e0) / dt                                  # m/s
    d0 = (t0 - e0) / max(np.linalg.norm(t0 - e0), 1e-9)
    d1 = (t1 - e1) / max(np.linalg.norm(t1 - e1), 1e-9)
    w = (d1 - d0) / dt                                  # change of the view direction, rad/s (small angles)
    hfov = math.radians((h0 + h1) / 2)
    half_w = REF_DEPTH_M * math.tan(hfov / 2)           # half the frame width at the reference depth, m
    x = (float(v @ right) / (2 * half_w) + float(w @ right) / hfov) * 100.0
    y = (float(v @ cam_up) / (2 * half_w) + float(w @ cam_up) / hfov) * 100.0
    zoom = -math.log(math.tan(math.radians(h1) / 2) / math.tan(math.radians(h0) / 2)) / dt
    f = (float(v @ fwd) / REF_DEPTH_M + zoom) * 50.0
    return Motion(x, y, f)


def shot_ends(plan, k: int, timeline: list[dict]) -> tuple[Motion | None, Motion | None]:
    """(entry, exit) motion of shot k over its visible part; None for a clip or an unknown camera."""
    shot, tl = plan.shots[k], timeline[k]
    fps = plan.output.fps
    n = tl["render_frames"]
    first, last = tl["head"], n - 1 - tl["tail"]
    m = max(1, min(int(round(MEASURE_S * fps)), (last - first) // 2))
    cam = shot.camera
    if cam.type in ("keyframe_path", "compound", "ken_burns"):
        from tour_worker.render.frames import camera_frames
        frames = camera_frames(shot, n, plan.output.width, plan.output.height, fps)
        args = (fps, shot.source.width, shot.source.height, plan.output.width, plan.output.height)
        return (_photo_motion(frames, first, first + m, *args), _photo_motion(frames, last - m, last, *args))
    if cam.type == "bezier_path":
        from tour_worker.scene.camera import BezierCameraPath
        path = BezierCameraPath([s.model_dump() for s in cam.segments], cam.hfov_deg,
                                easing=getattr(cam.easing, "value", cam.easing), time_warp=cam.time_warp,
                                speed_profile=getattr(cam, "speed_profile", None))
        up = getattr(getattr(plan, "scene", None), "up_axis", None) or (0.0, 0.0, 1.0)
        ps = getattr(cam, "path_s", None)
        return (_scene_motion(path, n, first, first + m, fps, up, ps), _scene_motion(path, n, last - m, last, fps, up, ps))
    return None, None


# --------------------------------------------------------------------------------------------
# the cuts
# --------------------------------------------------------------------------------------------

def _turn_deg(a: Motion, b: Motion) -> float | None:
    """Lateral direction change, degrees; None when either end has no lateral motion to speak of."""
    if a.lateral < max(STILL, SIGNIFICANT * a.speed) or b.lateral < max(STILL, SIGNIFICANT * b.speed):
        return None
    d = abs(math.degrees(math.atan2(b.y, b.x) - math.atan2(a.y, a.x))) % 360
    return min(d, 360 - d)


def _axial_reversal(a: Motion, b: Motion) -> bool:
    big = lambda m: abs(m.f) >= max(STILL, SIGNIFICANT * m.speed)   # noqa: E731
    return big(a) and big(b) and (a.f > 0) != (b.f > 0)


SEAM_EYE_M, SEAM_AIM_M = 0.03, 0.06     # a join this close in place and aim is no cut (one take)


def _seamless(a, b, fps: float) -> bool:
    """Shot a plays its whole camera path and b's starts where it ends - same place, same aim: one
    take through the home (planner3d/plan.py _stitch), not a cut."""
    ca, cb = a.camera, b.camera
    if getattr(ca, "type", None) != "bezier_path" or getattr(cb, "type", None) != "bezier_path":
        return False
    if not ca.segments or not cb.segments:
        return False
    path_s = getattr(ca, "path_s", None)
    if path_s is not None and abs(float(path_s) - float(a.duration_s)) > 1.5 / fps:
        return False                          # (a is cut inside its path: b starts elsewhere)
    import numpy as np
    e0, t0 = np.asarray(ca.segments[-1].eye[-1], float), np.asarray(ca.segments[-1].target[-1], float)
    e1, t1 = np.asarray(cb.segments[0].eye[0], float), np.asarray(cb.segments[0].target[0], float)
    return float(np.linalg.norm(e1 - e0)) <= SEAM_EYE_M and float(np.linalg.norm(t1 - t0)) <= SEAM_AIM_M


def report(plan) -> list[dict]:
    """One row per cut: outgoing exit and incoming entry motion, their speed ratio, the direction
    change, the beat offset, and what (if anything) breaks the limits."""
    from tour_worker.render.timeline import compute_timeline
    timeline = compute_timeline(plan)
    fps = plan.output.fps
    ends = [shot_ends(plan, k, timeline) for k in range(len(plan.shots))]
    music = getattr(plan, "music", None)
    beats = list(getattr(music, "cut_beats_s", None) or []) if music is not None else []
    lead = int(getattr(music, "cut_lead_frames", 0) or 0) if music is not None else 0
    rows = []
    for k in range(len(plan.shots) - 1):
        a, b = plan.shots[k], plan.shots[k + 1]
        out_m, in_m = ends[k][1], ends[k + 1][0]
        cut_frame = timeline[k + 1]["nominal_start"]
        row = {"cut": k + 1, "from": a.shot_id, "to": b.shot_id, "at_s": round(cut_frame / fps, 3),
               "transition": b.transition_in.type, "from_room": a.room_key, "to_room": b.room_key,
               "out": out_m.rounded() if out_m else None, "in": in_m.rounded() if in_m else None,
               "speed_ratio": None, "turn_deg": None, "axial_reversal": False, "beat_offset_frames": None,
               "judged": b.transition_in.type == "cut" and out_m is not None and in_m is not None,
               "breaks": []}
        if _seamless(a, b, fps):             # one take: the camera carries straight on - no cut to judge
            row.update(judged=False, seamless=True)
        if out_m is not None and in_m is not None:
            lo, hi = sorted((out_m.speed, in_m.speed))
            row["speed_ratio"] = round(hi / lo, 2) if lo >= STILL else None
            t = _turn_deg(out_m, in_m)
            row["turn_deg"] = None if t is None else round(t, 1)
            row["axial_reversal"] = _axial_reversal(out_m, in_m)
        if k < len(beats) and beats[k] is not None:
            row["beat_offset_frames"] = round(cut_frame + lead - beats[k] * fps, 2)
        if row["judged"]:
            if out_m.speed < STILL or in_m.speed < STILL:
                row["breaks"].append("stopped")
            elif row["speed_ratio"] is not None and row["speed_ratio"] > SPEED_RATIO + 1e-6:
                row["breaks"].append("speed_jump")
            reason = getattr(b, "cut_reason", None)
            if reason:                       # a reversal the plan allows at this cut, and why
                row["reset"] = reason
            else:
                if row["turn_deg"] is not None and row["turn_deg"] > MAX_TURN_DEG + 1e-6:
                    row["breaks"].append("direction_turn")
                if row["axial_reversal"]:
                    row["breaks"].append("forward_back")
            if row["beat_offset_frames"] is not None and abs(row["beat_offset_frames"]) > BEAT_TOL_FRAMES + 1e-6:
                row["breaks"].append("off_beat")
        rows.append(row)
    return rows


def violations(rows: list[dict]) -> list[str]:
    """cut_<rule>:<cut>:<from>-><to> for every judged cut outside the limits."""
    return [f"cut_{r}:{row['cut']}:{row['from']}->{row['to']}" for row in rows for r in row["breaks"]]


def summary(rows: list[dict]) -> dict:
    judged = [r for r in rows if r["judged"]]
    count = lambda rule: sum(rule in r["breaks"] for r in judged)   # noqa: E731
    return {"cuts": len(rows), "judged": len(judged), "within_limits": sum(not r["breaks"] for r in judged),
            "seamless": sum(bool(r.get("seamless")) for r in rows),
            "stopped": count("stopped"), "speed_jump": count("speed_jump"), "direction_turn": count("direction_turn"),
            "forward_back": count("forward_back"), "off_beat": count("off_beat"),
            "limits": {"speed_ratio": SPEED_RATIO, "max_turn_deg": MAX_TURN_DEG, "still_pct_fw_s": STILL,
                       "beat_tol_frames": BEAT_TOL_FRAMES, "ref_depth_m": REF_DEPTH_M, "measure_s": MEASURE_S}}


__all__ = ["Motion", "report", "violations", "summary", "shot_ends", "asdict"]
