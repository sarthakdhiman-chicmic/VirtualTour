import hashlib
import json

from modules.camera_path import PROJECTIONS, CompoundPath, KeyframePath, validate_compound, base_window, frame_margin_out_px, validate_keyframes
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

class OutputConfig(BaseModel):
    width: int
    height: int
    fps: int
    codec: str
    model_config = ConfigDict(extra='ignore')

class Size(BaseModel):
    width: int
    height: int
    model_config = ConfigDict(extra='ignore')

class CameraTransform(BaseModel):
    cx: float
    cy: float
    zoom: float
    # gimbal roll; the renderer fits the rolled frame inside the (cx, cy, zoom) window
    roll_deg: float = 0.0
    model_config = ConfigDict(extra='ignore')

class Keyframe(BaseModel):
    t: float
    cx: float
    cy: float
    zoom: float
    roll_deg: Optional[float] = None   # addendum 1.3.3, crop projection only
    model_config = ConfigDict(extra='ignore')

class Segment(BaseModel):
    t0: float
    t1: float
    cx: float
    cy: float
    zoom: float
    model_config = ConfigDict(extra='ignore')

class Timing(BaseModel):
    s0: float
    s1: float
    turn: Optional[float] = None
    model_config = ConfigDict(extra='ignore')

class Camera(BaseModel):
    type: str
    easing: str = "ease_in_out"
    preset: Optional[str] = None
    # ken_burns
    start: Optional[CameraTransform] = None
    end: Optional[CameraTransform] = None
    # keyframe_path (plan 1.3)
    move: Optional[str] = None
    keyframes: Optional[List[Keyframe]] = None
    # keyframe_path, addendum 1.3.1 (optional): "crop" (default) or "rotate"; hfov_deg is the
    # photo's assumed horizontal field of view for "rotate" (modules/camera_path.py)
    projection: Optional[str] = None
    hfov_deg: Optional[float] = None
    # compound (addendum 1.3.2): start + segments + timing (modules/camera_path.py)
    segments: Optional[List[Segment]] = None
    timing: Optional[Timing] = None
    # beat sync (plan 1.5): [output t, move t] points, monotone, (0,0) .. (1,1): the move planned
    # for the shot's normal length, retimed to its synced length (modules/camera_path.time_warp_at)
    time_warp: Optional[List[List[float]]] = None
    # beat sync: zoom punches on musical hits (modules/camera_path.punch_scale)
    punches: Optional[List["Punch"]] = None
    model_config = ConfigDict(extra='ignore')

class Punch(BaseModel):
    """A zoom-in pulse on a musical hit: zoom x (1 + amount * pulse), the pulse rising over
    attack_s to 1 at at_s (s into the shot), easing back to 0 over decay_s."""
    at_s: float
    amount: float
    attack_s: float = 0.0
    decay_s: float
    model_config = ConfigDict(extra='ignore')

class Transition(BaseModel):
    type: str
    duration_s: float
    model_config = ConfigDict(extra='ignore')

MOTION_EFFECTS = ("fabric_sway", "foliage_sway", "water_ripple", "calm_water", "canopy_flutter", "fire_flicker")
MAX_MOTION_AMPLITUDE_PX = 48.0   # validation bound; the backend sets strength on screen (up to 48 source px on large photos)

class MotionRegion(BaseModel):
    """Plan 1.4 perceptual motion inside an object mask (modules/perceptual)."""
    id: str
    effect: str
    mask_uri: str
    bbox: List[float]                 # [x1, y1, x2, y2] normalised to the photo
    amplitude_px: float               # peak displacement, source px
    freq_hz: float
    anchor: str                       # "top" | "bottom"
    seed: int
    label: Optional[str] = None
    phase: Optional[float] = None     # sway phase (rad), set by the planner; random when None
    model_config = ConfigDict(extra='ignore')

class Shot(BaseModel):
    shot_id: str
    order: int
    asset_id: str
    image_uri: str
    source: Size
    room_key: str
    room_type: Optional[str] = None   # the backend allows untagged rooms
    narrative_stage: str
    role: str
    duration_s: float
    camera: Camera
    transition_in: Transition
    motion_regions: Optional[List[MotionRegion]] = None   # plan 1.4
    model_config = ConfigDict(extra='ignore')

class Provenance(BaseModel):
    scoring_config_version: str
    planner_config_version: str
    plan_hash: str
    model_config = ConfigDict(extra='ignore')

class Music(BaseModel):
    """Plan 1.5: the tour's music (backend planner/music_sync.py). The track is queued `loops`
    times, each copy cut at loop_s (the bar where its beat stops), trimmed to end_s, faded out."""
    track_id: str
    uri: str
    sha256: Optional[str] = None
    end_s: float
    loops: int = 1
    loop_s: Optional[float] = None
    track_duration_s: Optional[float] = None
    fade_out_s: float = 1.5
    synced: bool = False
    cut_lead_frames: int = 0               # cuts sit this many frames before their beat
    hooks_required: bool = False           # every hook moment got a cut (tools/sync_quality.py)
    hooks: Optional[List[dict]] = None     # [{t, kind: cut|accent|missed, reason}]
    model_config = ConfigDict(extra='ignore')

class Plan(BaseModel):
    plan_version: str
    listing_id: str
    mode: str
    fallback_reason: Optional[str] = None
    preset: str
    output: OutputConfig
    total_duration_s: float
    shots: List[Shot]
    warnings: List[Any] = Field(default_factory=list)
    provenance: Provenance
    music: Optional[Music] = None
    model_config = ConfigDict(extra='ignore')

def compute_plan_hash(plan_dict: dict) -> str:
    """sha256 of the canonical JSON of the plan without provenance.plan_hash."""
    dict_copy = dict(plan_dict)
    dict_copy["provenance"] = dict(dict_copy.get("provenance", {}))
    dict_copy["provenance"].pop("plan_hash", None)
    canonical_json = json.dumps(dict_copy, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

def validate_plan_hash(original_dict: dict):
    if "provenance" not in original_dict or "plan_hash" not in original_dict["provenance"]:
        return
    expected = original_dict["provenance"]["plan_hash"]
    computed = compute_plan_hash(original_dict)
    if computed != expected:
        raise ValueError(f"plan_hash mismatch: expected {computed}, got {expected}")

CAMERA_TYPES = {"1.0": {"ken_burns"}, "1.1": {"ken_burns"}, "1.2": {"ken_burns"},
                "1.3": {"ken_burns", "keyframe_path", "compound"},
                "1.4": {"ken_burns", "keyframe_path", "compound"},
                "1.5": {"ken_burns", "keyframe_path", "compound"}}   # 1.5 = 1.4 + a music block

def check_camera_type(plan: Plan):
    allowed = CAMERA_TYPES.get(plan.plan_version, {"ken_burns"})
    for shot in plan.shots:
        if shot.camera.type not in allowed:
            raise ValueError(f"unsupported_camera_type: {shot.camera.type}")
        if shot.camera.type == "ken_burns" and (shot.camera.start is None or shot.camera.end is None):
            raise ValueError(f"invalid_plan: ken_burns camera in shot {shot.shot_id} needs start and end")
        if shot.camera.type == "keyframe_path":
            if not shot.camera.move:
                raise ValueError(f"invalid_plan: keyframe_path camera in shot {shot.shot_id} needs a move name")
            validate_keyframes([k.model_dump() for k in (shot.camera.keyframes or [])])
        if shot.camera.type == "compound":
            if not shot.camera.move:
                raise ValueError(f"invalid_plan: compound camera in shot {shot.shot_id} needs a move name")
            validate_compound(shot.camera.start.model_dump() if shot.camera.start else None,
                              [s.model_dump() for s in (shot.camera.segments or [])],
                              shot.camera.timing.model_dump() if shot.camera.timing else None)
        if shot.camera.type in ("keyframe_path", "compound"):
            if shot.camera.projection is not None and shot.camera.projection not in PROJECTIONS:
                raise ValueError(f"unsupported_projection: {shot.camera.projection}")
            if (shot.camera.type == "keyframe_path" and shot.camera.projection == "rotate"
                    and any(k.roll_deg for k in shot.camera.keyframes or [])):
                raise ValueError(f"invalid_plan: roll_deg in shot {shot.shot_id} is crop-only")
            if shot.camera.hfov_deg is not None and not 20 <= shot.camera.hfov_deg <= 120:
                raise ValueError(f"invalid_plan: hfov_deg in shot {shot.shot_id} must be 20-120")

def _projection_kwargs(camera) -> dict:
    kw = {"projection": camera.projection or "crop"}
    if camera.hfov_deg is not None:
        kw["hfov_deg"] = camera.hfov_deg
    return kw

def compound_path(shot, out_w: int, out_h: int) -> CompoundPath:
    c = shot.camera
    return CompoundPath(c.start.model_dump(), [s.model_dump() for s in c.segments], c.timing.model_dump(),
                        shot.source.width, shot.source.height, out_w, out_h)

def check_time_warps(plan: Plan):
    for shot in plan.shots:
        w = shot.camera.time_warp
        if w is None:
            continue
        ok = (len(w) >= 2 and all(len(p) == 2 for p in w) and w[0] == [0.0, 0.0] and w[-1] == [1.0, 1.0]
              and all(b[0] > a[0] and b[1] > a[1] for a, b in zip(w, w[1:])))
        if not ok:
            raise ValueError(f"invalid_plan: time_warp in shot {shot.shot_id} must rise from (0,0) to (1,1)")


MAX_PUNCH = 0.06          # zoom-in share; the backend plans 0.02-0.025

def check_punches(plan: Plan):
    for shot in plan.shots:
        for p in shot.camera.punches or []:
            if not (0 <= p.at_s <= shot.duration_s and 0 < p.amount <= MAX_PUNCH
                    and 0 <= p.attack_s <= 0.5 and 0.05 <= p.decay_s <= 2.0):
                raise ValueError(f"invalid_plan: punch in shot {shot.shot_id} out of range")


def check_path_bounds(plan: Plan):
    """keyframe_path: the window must stay inside the image at every rendered frame (the planner
    adds its own 1 px margin on top). Frames are the shot's render span, as in modules.timeline."""
    from modules.timeline import compute_timeline
    timeline = compute_timeline(plan)
    for shot, tl in zip(plan.shots, timeline):
        if shot.camera.type == "compound":
            path = compound_path(shot, plan.output.width, plan.output.height)
        elif shot.camera.type == "keyframe_path":
            path = KeyframePath([k.model_dump() for k in shot.camera.keyframes])
        else:
            continue
        for cx, cy, z in path.frames(tl["render_frames"]):
            m = frame_margin_out_px(cx, cy, z, shot.source.width, shot.source.height,
                                    plan.output.width, plan.output.height, **_projection_kwargs(shot.camera))
            if m < -0.01:
                raise ValueError(f"Camera window out of bounds in shot {shot.shot_id}")

def check_bounds(plan: Plan):
    out_ratio = plan.output.width / plan.output.height
    for shot in plan.shots:
        src_ratio = shot.source.width / shot.source.height
        if src_ratio > out_ratio:
            rect_w = out_ratio / src_ratio
            rect_h = 1.0
        else:
            rect_w = 1.0
            rect_h = src_ratio / out_ratio

        if shot.camera.type == "compound":
            points = [shot.camera.start] + list(shot.camera.segments or [])
        elif shot.camera.type == "keyframe_path":
            points = shot.camera.keyframes or []
        else:
            points = [p for p in (shot.camera.start, shot.camera.end) if p is not None]
        for t in points:
            w = rect_w / t.zoom
            h = rect_h / t.zoom
            left = t.cx - w / 2
            right = t.cx + w / 2
            top = t.cy - h / 2
            bottom = t.cy + h / 2
            eps = 1e-4
            if left < -eps or right > 1+eps or top < -eps or bottom > 1+eps:
                raise ValueError(f"Camera window out of bounds in shot {shot.shot_id}")

def check_easing(plan: Plan):
    valid_easings = {"linear", "ease_in", "ease_out", "ease_in_out"}
    for shot in plan.shots:
        if shot.camera.type in ("keyframe_path", "compound"):
            continue  # fixed global time profile (modules/camera_path.py)
        if shot.camera.easing not in valid_easings:
            raise ValueError(f"unsupported_easing: {shot.camera.easing}")
        if plan.plan_version in ("1.0", "1.1") and shot.camera.easing != "ease_in_out":
            raise ValueError(f"invalid_plan: non-default easing '{shot.camera.easing}' not allowed in plan_version {plan.plan_version}")

def check_transitions(plan: Plan):
    if plan.plan_version not in ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5"):
        raise ValueError(f"unsupported plan version: {plan.plan_version}")
    
    if not plan.shots:
        return

    first_trans = plan.shots[0].transition_in
    if first_trans.type != "cut" or first_trans.duration_s != 0.0:
        raise ValueError("First shot must have cut transition with 0.0 duration")

    for shot in plan.shots:
        if plan.plan_version == "1.0":
            if shot.transition_in.type != "cut":
                raise ValueError("plan_version 1.0 only supports cut transitions")
        elif plan.plan_version in ("1.1", "1.2", "1.3", "1.4", "1.5"):
            t = shot.transition_in
            if t.type not in ("cut", "fade"):
                raise ValueError(f"Unsupported transition type: {t.type}")
            if t.type == "fade" and t.duration_s > shot.duration_s * 0.5:
                raise ValueError("fade duration cannot exceed 50% of shot duration")

def check_motion_regions(plan: Plan):
    for shot in plan.shots:
        if not shot.motion_regions:
            continue
        if plan.plan_version not in ("1.4", "1.5"):
            raise ValueError(f"invalid_plan: motion_regions in shot {shot.shot_id} need plan_version 1.4 or later")
        for r in shot.motion_regions:
            if r.effect not in MOTION_EFFECTS:
                raise ValueError(f"unsupported_motion_effect: {r.effect}")
            if r.anchor not in ("top", "bottom", "none"):
                raise ValueError(f"invalid_plan: motion region {r.id} anchor must be top, bottom or none")
            if not 0 <= r.amplitude_px <= MAX_MOTION_AMPLITUDE_PX or not 0 < r.freq_hz <= 2:
                raise ValueError(f"invalid_plan: motion region {r.id} amplitude/frequency out of range")
            if len(r.bbox) != 4 or not (0 <= r.bbox[0] < r.bbox[2] <= 1 and 0 <= r.bbox[1] < r.bbox[3] <= 1):
                raise ValueError(f"invalid_plan: motion region {r.id} bbox must be normalised x1<x2, y1<y2")
            if not r.mask_uri.startswith(("s3://", "http://", "https://", "local://")) and "/" not in r.mask_uri:
                raise ValueError(f"invalid_plan: motion region {r.id} mask_uri")

def parse_plan(data: dict) -> Plan:
    validate_plan_hash(data)
    plan = Plan(**data)
    check_camera_type(plan)
    check_bounds(plan)
    check_easing(plan)
    check_transitions(plan)
    check_time_warps(plan)
    check_punches(plan)
    check_path_bounds(plan)
    check_motion_regions(plan)
    return plan
