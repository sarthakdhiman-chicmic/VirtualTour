import hashlib
import json

from modules.camera_path import KeyframePath, base_window, margin_out_px, validate_keyframes
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
    model_config = ConfigDict(extra='ignore')

class Transition(BaseModel):
    type: str
    duration_s: float
    model_config = ConfigDict(extra='ignore')

class Shot(BaseModel):
    shot_id: str
    order: int
    asset_id: str
    image_uri: str
    source: Size
    room_key: str
    room_type: str
    narrative_stage: str
    role: str
    duration_s: float
    camera: Camera
    transition_in: Transition
    model_config = ConfigDict(extra='ignore')

class Provenance(BaseModel):
    scoring_config_version: str
    planner_config_version: str
    plan_hash: str
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
                "1.3": {"ken_burns", "keyframe_path"}}

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

def check_path_bounds(plan: Plan):
    """keyframe_path: the window must stay inside the image at every rendered frame (the planner
    adds its own 1 px margin on top). Frames are the shot's render span, as in modules.timeline."""
    from modules.timeline import compute_timeline
    timeline = compute_timeline(plan)
    for shot, tl in zip(plan.shots, timeline):
        if shot.camera.type != "keyframe_path":
            continue
        path = KeyframePath([k.model_dump() for k in shot.camera.keyframes])
        for cx, cy, z in path.frames(tl["render_frames"]):
            m = margin_out_px(cx, cy, z, shot.source.width, shot.source.height,
                              plan.output.width, plan.output.height)
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

        if shot.camera.type == "keyframe_path":
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
        if shot.camera.type == "keyframe_path":
            continue  # fixed global time profile (modules/camera_path.py)
        if shot.camera.easing not in valid_easings:
            raise ValueError(f"unsupported_easing: {shot.camera.easing}")
        if plan.plan_version in ("1.0", "1.1") and shot.camera.easing != "ease_in_out":
            raise ValueError(f"invalid_plan: non-default easing '{shot.camera.easing}' not allowed in plan_version {plan.plan_version}")

def check_transitions(plan: Plan):
    if plan.plan_version not in ("1.0", "1.1", "1.2", "1.3"):
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
        elif plan.plan_version in ("1.1", "1.2", "1.3"):
            t = shot.transition_in
            if t.type not in ("cut", "fade"):
                raise ValueError(f"Unsupported transition type: {t.type}")
            if t.type == "fade" and t.duration_s > shot.duration_s * 0.5:
                raise ValueError("fade duration cannot exceed 50% of shot duration")

def parse_plan(data: dict) -> Plan:
    validate_plan_hash(data)
    plan = Plan(**data)
    check_camera_type(plan)
    check_bounds(plan)
    check_easing(plan)
    check_transitions(plan)
    check_path_bounds(plan)
    return plan
