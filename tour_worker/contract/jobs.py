"""Request and result contracts of the worker's jobs (the JSON RunPod passes as `input`, and what the
worker returns). The plan itself is validated by tour_worker.contract.plan.parse_plan, inside the
render job: its errors keep their own codes (invalid_plan, unsupported_camera_type, ...)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

JOB_TYPES = ("tour_render", "tour_segment")


class _Request(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_type: str
    job_id: Optional[str] = None


class TourRenderRequest(_Request):
    """Render a plan into an MP4: <output_prefix>master.mp4."""
    job_type: Literal["tour_render"]
    plan: dict[str, Any]                      # the plan JSON as the backend sent it (hashed as-is)
    output_prefix: str = Field(min_length=1)  # where master.mp4 goes, e.g. s3://bucket/tours/<id>/
    image_urls: dict[str, str] = Field(default_factory=dict)   # optional: plan URI -> fetch URL


class SegmentImage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    photo_id: Any
    uri: str


class TourSegmentRequest(_Request):
    """OneFormer masks for a tour's photos, before the backend plans."""
    job_type: Literal["tour_segment"]
    images: list[SegmentImage] = Field(min_length=1)
    output_prefix: str = Field(min_length=1)  # masks go to <output_prefix><photo_id>/<object_id>.png


class JobError(BaseModel):
    code: str
    message: str


class JobFailure(BaseModel):
    """Every failed job, whatever its type."""
    status: Literal["failed"]
    error: JobError


class TourRenderResult(BaseModel):
    status: Literal["completed"]
    video_uri: str
    duration_s: float
    width: int
    height: int
    fps: int
    codec: str
    render_seconds: float
    plan_hash: Optional[str]
    warnings: list[Any]
    error: None = None


class SegmentRegion(BaseModel):
    model_config = ConfigDict(extra="allow")
    object_id: str
    class_name: str
    confidence: float
    bbox: list[int]                           # [x1, y1, x2, y2] in `space` pixels
    space: list[int]                          # [w, h] the masks were made at
    mask_url: str


class TourSegmentResult(BaseModel):
    status: Literal["completed"]
    results: dict[str, Optional[list[SegmentRegion]]]   # photo_id -> regions (None: photo failed)
    scenes: dict[str, Optional[dict[str, Any]]]         # photo_id -> scene summary (None: failed)
    warnings: list[str]
    model: str                                          # "<hf id>@<revision[:10]>"
    seconds: float
