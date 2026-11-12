"""scene.json: what an operator marks on a splat scene in a viewer - its up axis, scale, floor
and, per room, the camera anchors to film it from. Optional (plan 1.6 tours run without it);
when given it is validated strictly: unknown keys, wrong types or impossible values raise
SceneInvalid, nothing is guessed.

    {"version": 1, "splat_sha256": "<hex of the model file>", "up_axis": [0, 0, 1],
     "metres_per_unit": 1.0, "floor_height": 0.0,
     "capture_poses": [{"position": [x, y, z], "forward": [x, y, z]}],          (optional)
     "rooms": [{"room_key": "kitchen_1", "room_type": "kitchen", "label": "Kitchen",
                "anchors": [{"position": [x, y, z], "look_at": [x, y, z],
                             "hfov_deg": 70, "hero": true}]}]}
"""
from __future__ import annotations

import json
import math

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from tour_worker.planner.captions import ROOM_LABELS
from tour_worker.scene import SceneInvalid

SCENE_JSON_VERSION = 1
ROOM_TYPES = frozenset(ROOM_LABELS) | {"other"}     # the backend's TourRoomType values
MAX_LABEL_CHARS = 40
MAX_ROOMS = 64
MAX_ANCHORS = 32
MAX_CAPTURE_POSES = 10_000


def _vec3(v: list[float], nonzero: bool = False) -> list[float]:
    if len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise ValueError("must be 3 finite numbers")
    if nonzero and math.hypot(*v) < 1e-9:
        raise ValueError("must not be the zero vector")
    return v


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=False)


class CapturePose(_Strict):
    position: list[float]
    forward: list[float]

    @field_validator("position")
    @classmethod
    def _p(cls, v):
        return _vec3(v)

    @field_validator("forward")
    @classmethod
    def _f(cls, v):
        return _vec3(v, nonzero=True)


class Anchor(_Strict):
    position: list[float]
    look_at: list[float]
    hfov_deg: float = 70.0
    hero: bool = False

    @field_validator("position", "look_at")
    @classmethod
    def _v(cls, v):
        return _vec3(v)

    @field_validator("hfov_deg")
    @classmethod
    def _h(cls, v):
        if not 20.0 <= v <= 120.0:
            raise ValueError("must be within 20-120")
        return v

    @model_validator(mode="after")
    def _looks_somewhere(self):
        if math.dist(self.position, self.look_at) < 1e-6:
            raise ValueError("look_at must differ from position")
        return self


class Room(_Strict):
    room_key: str
    room_type: str
    label: str
    anchors: list[Anchor]

    @field_validator("room_key")
    @classmethod
    def _k(cls, v):
        if not v.strip() or len(v) > 64:
            raise ValueError("must be 1-64 characters")
        return v

    @field_validator("room_type")
    @classmethod
    def _t(cls, v):
        if v not in ROOM_TYPES:
            raise ValueError(f"unknown room type {v!r}")
        return v

    @field_validator("label")
    @classmethod
    def _l(cls, v):
        if not v.strip() or len(v) > MAX_LABEL_CHARS:
            raise ValueError(f"must be 1-{MAX_LABEL_CHARS} characters")
        return v

    @field_validator("anchors")
    @classmethod
    def _a(cls, v):
        if not 1 <= len(v) <= MAX_ANCHORS:
            raise ValueError(f"needs 1-{MAX_ANCHORS} anchors")
        return v


class SceneJson(_Strict):
    version: int
    splat_sha256: str
    up_axis: list[float]
    metres_per_unit: float
    floor_height: float
    capture_poses: list[CapturePose] | None = None
    rooms: list[Room]

    @field_validator("version")
    @classmethod
    def _ver(cls, v):
        if v != SCENE_JSON_VERSION:
            raise ValueError(f"unsupported version {v}")
        return v

    @field_validator("splat_sha256")
    @classmethod
    def _sha(cls, v):
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("must be 64 lowercase hex characters")
        return v

    @field_validator("up_axis")
    @classmethod
    def _up(cls, v):
        return _vec3(v, nonzero=True)

    @field_validator("metres_per_unit")
    @classmethod
    def _scale(cls, v):
        if not (math.isfinite(v) and v > 0):
            raise ValueError("must be a positive number")
        return v

    @field_validator("floor_height")
    @classmethod
    def _floor(cls, v):
        if not math.isfinite(v):
            raise ValueError("must be finite")
        return v

    @field_validator("capture_poses")
    @classmethod
    def _poses(cls, v):
        if v is not None and len(v) > MAX_CAPTURE_POSES:
            raise ValueError(f"at most {MAX_CAPTURE_POSES}")
        return v

    @field_validator("rooms")
    @classmethod
    def _rooms(cls, v):
        if not 1 <= len(v) <= MAX_ROOMS:
            raise ValueError(f"needs 1-{MAX_ROOMS} rooms")
        keys = [r.room_key for r in v]
        if len(set(keys)) != len(keys):
            raise ValueError("room_key values must be unique")
        return v


def parse_scene_json(data: bytes | str | dict, model_sha256: str | None = None) -> SceneJson:
    """The validated scene.json. model_sha256: the scene file's hash; scene.json must name it.
    Raises SceneInvalid with the reason."""
    try:
        if isinstance(data, (bytes, str)):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise SceneInvalid("scene.json must be a JSON object")
        sj = SceneJson.model_validate(data)
    except SceneInvalid:
        raise
    except (ValueError, ValidationError) as e:          # JSONDecodeError is a ValueError
        raise SceneInvalid(f"scene.json: {e}") from None
    if model_sha256 is not None and sj.splat_sha256 != model_sha256:
        raise SceneInvalid("scene.json was made for another scene file (splat_sha256 does not match)")
    return sj
