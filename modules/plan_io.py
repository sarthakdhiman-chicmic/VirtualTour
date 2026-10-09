import hashlib
import json
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
    model_config = ConfigDict(extra='ignore')

class Camera(BaseModel):
    type: str
    easing: str
    start: CameraTransform
    end: CameraTransform
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

def validate_plan_hash(original_dict: dict):
    if "provenance" not in original_dict or "plan_hash" not in original_dict["provenance"]:
        return
    expected = original_dict["provenance"]["plan_hash"]
    
    dict_copy = dict(original_dict)
    dict_copy["provenance"] = dict(dict_copy["provenance"])
    del dict_copy["provenance"]["plan_hash"]
    
    canonical_json = json.dumps(dict_copy, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
    computed = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    if computed != expected:
        raise ValueError(f"plan_hash mismatch: expected {computed}, got {expected}")

def check_camera_type(plan: Plan):
    for shot in plan.shots:
        if shot.camera.type != "ken_burns":
            raise ValueError(f"unsupported_camera_type: {shot.camera.type}")

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

        for t in [shot.camera.start, shot.camera.end]:
            w = rect_w / t.zoom
            h = rect_h / t.zoom
            left = t.cx - w / 2
            right = t.cx + w / 2
            top = t.cy - h / 2
            bottom = t.cy + h / 2
            eps = 1e-4
            if left < -eps or right > 1+eps or top < -eps or bottom > 1+eps:
                raise ValueError(f"Camera window out of bounds in shot {shot.shot_id}")

def parse_plan(data: dict) -> Plan:
    validate_plan_hash(data)
    plan = Plan(**data)
    check_bounds(plan)
    check_camera_type(plan)
    return plan
