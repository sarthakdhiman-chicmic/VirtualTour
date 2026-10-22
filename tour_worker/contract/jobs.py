"""Request and result contracts of the worker's jobs (the JSON RunPod passes as `input`, and what the
worker returns). The plan itself is validated by tour_worker.contract.plan.parse_plan, inside the
render job (and on the tour_plan job's output): its errors keep their own codes (invalid_plan,
unsupported_camera_type, ...)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

JOB_TYPES = ("tour_render", "tour_segment", "tour_plan")


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
    """OneFormer masks for a tour's photos, before tour_plan plans."""
    job_type: Literal["tour_segment"]
    images: list[SegmentImage] = Field(min_length=1)
    output_prefix: str = Field(min_length=1)  # masks go to <output_prefix><photo_id>/<object_id>.png


class PlanPhoto(BaseModel):
    model_config = ConfigDict(extra="ignore")
    photo_id: str = Field(min_length=1)       # the planner's id (scoring, segments, scenes keys)
    uri: str = Field(min_length=1)            # the canonical URI: the plan's image_uri, as hashed
    room_type: Optional[str] = None           # as chosen by the user at upload ("Living Room")


class MusicTrackInput(BaseModel):
    """A music-library track (plan 1.5 beat sync): its analysis is made once, offline
    (tour_worker.music). s3_uri becomes the plan's music.uri."""
    model_config = ConfigDict(extra="ignore")
    track_id: str
    s3_uri: str
    sha256: str
    analysis: dict[str, Any]


class PlanOptions(BaseModel):
    """What the caller chooses about the tour; everything else is PlannerConfig's default."""
    model_config = ConfigDict(extra="forbid")
    preset: str = "property_showcase"
    budget_blocked: bool = False              # True: a slideshow (fallback_reason budget_exceeded)
    output_width: int = Field(1920, gt=0)
    output_height: int = Field(1080, gt=0)
    max_total_s: float = Field(120.0, gt=0)   # the tour's ceiling (the backend caps it at 120)


class TourPlanRequest(_Request):
    """Score a tour's photos and plan its shots (the plan tour_render renders)."""
    job_type: Literal["tour_plan"]
    listing_id: str = Field(min_length=1)     # seeds the motion regions; in the plan
    photos: list[PlanPhoto] = Field(min_length=1)
    image_urls: dict[str, str] = Field(default_factory=dict)   # optional: photo uri -> fetch URL
    segments: Optional[dict[str, Optional[list[dict[str, Any]]]]] = None   # tour_segment results
    scenes: Optional[dict[str, Optional[dict[str, Any]]]] = None           # tour_segment scenes
    music_track: Optional[MusicTrackInput] = None
    options: PlanOptions = Field(default_factory=PlanOptions)

    @model_validator(mode="after")
    def _unique_photo_ids(self):
        ids = [p.photo_id for p in self.photos]
        if len(set(ids)) != len(ids):
            raise ValueError("photos: photo_id must be unique")
        return self


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


class ScoringRoom(BaseModel):
    room_key: str
    room_type: Optional[str]
    narrative_order: int
    hero_photo_id: str
    photo_ids: list[str]                      # best first


class ScoringSummary(BaseModel):
    """What the backend keeps of the photo scoring (the full report stays in the worker)."""
    config_version: str
    photos: int                               # photos scored
    rooms: list[ScoringRoom]                  # most important first
    sequence: list[str]                       # room_keys in narrative order
    warnings: list[str]


class TourPlanResult(BaseModel):
    status: Literal["completed"]
    plan: dict[str, Any]                      # validated by contract.plan.parse_plan
    plan_hash: str                            # == plan["provenance"]["plan_hash"]
    scoring: ScoringSummary
    warnings: list[str]                       # photos left out (photo_fetch_failed / photo_unreadable)
    seconds: float


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
    fireplace_model: Optional[str] = None               # the fireplace pass's model (None: off)
    seconds: float
