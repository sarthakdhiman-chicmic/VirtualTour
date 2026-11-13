"""Request and result contracts of the worker's jobs (the JSON RunPod passes as `input`, and what the
worker returns). The plan itself is validated by tour_worker.contract.plan.parse_plan, inside the
render job (and on the tour_plan job's output): its errors keep their own codes (invalid_plan,
unsupported_camera_type, ...)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

JOB_TYPES = ("tour_render", "tour_segment", "tour_plan", "scene_label")


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
    known_version: Optional[str] = None    # the caller's cached result is of this segment_version


class TourSegmentRequest(_Request):
    """Mask2Former masks for a tour's photos, before tour_plan plans."""
    job_type: Literal["tour_segment"]
    images: list[SegmentImage] = Field(min_length=1)
    output_prefix: str = Field(min_length=1)  # masks go to <output_prefix><photo_id>/<object_id>.png


class PlanPhoto(BaseModel):
    model_config = ConfigDict(extra="ignore")
    photo_id: str = Field(min_length=1)       # the planner's id (scoring, segments, scenes keys)
    uri: str = Field(min_length=1)            # the canonical URI: the plan's image_uri, as hashed
    room_type: Optional[str] = None           # as chosen by the user at upload ("Living Room")
    staged: bool = False                      # virtually staged: the "Virtually Staged" label is drawn


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
    max_bytes: Optional[int] = Field(None, gt=0)   # a platform's file-size limit (Instagram Reels: 100 MB)
    hevc: bool = False                        # also render an H.265 copy next to the H.264 master
    photo_order: Literal["narrative", "manual"] = "narrative"   # manual: the photos' order in the request
    pacing: Optional[Literal["slow", "normal", "fast"]] = None   # 3-8 s shots (planner/config.py PACING); None: the preset's
    room_labels: Optional[RoomLabelsOption] = None   # room-name captions (absent: none)
    mls_copy: bool = False                    # also render the MLS copy (no captions)
    branding: Optional[dict[str, Any]] = None  # tour_worker.contract.plan.Branding: title / end card / logo
    voiceover: Optional[dict[str, Any]] = None # tour_worker.contract.plan.Voiceover: {uri, balance}


class RoomLabelsOption(BaseModel):
    """Room-name captions: a font from LABEL_FONTS, a scale within LABEL_SCALE_RANGE (x the maximum
    size; 0.8 is the default) and a placement from LABEL_PLACEMENTS (tour_worker.contract.plan)."""
    model_config = ConfigDict(extra="forbid")
    font: str = "sacramento"
    scale: float = 0.8
    placement: str = "bottom_left"

    @model_validator(mode="after")
    def _known(self):
        from tour_worker.contract.plan import LABEL_FONTS, LABEL_PLACEMENTS, LABEL_SCALE_RANGE
        if self.font not in LABEL_FONTS:
            raise ValueError(f"room_labels.font must be one of {', '.join(LABEL_FONTS)}")
        if self.placement not in LABEL_PLACEMENTS:
            raise ValueError(f"room_labels.placement must be one of {', '.join(LABEL_PLACEMENTS)}")
        lo, hi = LABEL_SCALE_RANGE
        if not lo <= self.scale <= hi:
            raise ValueError(f"room_labels.scale must be within {lo}-{hi}")
        return self


class PlanScene(BaseModel):
    """A 3D tour's Gaussian splat scene (plan 1.6 scene block): with one, tour_plan plans in 3D."""
    model_config = ConfigDict(extra="ignore")
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["ply", "splat"]
    up_axis: Optional[list[float]] = None     # the scene's up in its own axes; None: found from the geometry
    motion_uri: Optional[str] = None          # 3D motion (curtains, plants, flames), made once per scene
    motion_sha256: Optional[str] = None
    labels_uri: Optional[str] = None          # what each Gaussian is (the scene_label job): the planner
    labels_sha256: Optional[str] = None       # composes on beds / sofas, aims at windows, adds details


class PlanRoom(BaseModel):
    """A room of a 3D tour: what it is and one point inside it (scene axes, any height)."""
    model_config = ConfigDict(extra="ignore")
    room_key: Optional[str] = None            # default <type>_<n>
    room_type: str = Field(min_length=1)      # as for photos ("Living Room", "kitchen", ...)
    point: list[float] = Field(min_length=3, max_length=3)
    tour: bool = True                         # False: a room of the home that the tour leaves out


class SceneLabelRequest(_Request):
    """Label a splat scene once (at upload): what every Gaussian is (Mask2Former classes voted from
    rendered survey views of its rooms) -> <output_prefix>scene_labels.npz, and the scene's motion
    (curtains, plants, lit fireplaces) -> <output_prefix>scene_motion.npz. tour_plan reads the
    labels (scene.labels_uri); tour_render the motion (scene.motion_uri)."""
    job_type: Literal["scene_label"]
    scene: PlanScene
    rooms: list[PlanRoom] = Field(min_length=1)
    output_prefix: str = Field(min_length=1)
    image_urls: dict[str, str] = Field(default_factory=dict)   # optional: scene uri -> fetch URL


class TourPlanRequest(_Request):
    """Plan a tour (the plan tour_render renders). With `scene` + `rooms`: a 3D tour from the splat
    scene (tour_worker/planner3d); otherwise the photos are scored and planned (tour_worker/planner).
    A scene that cannot be loaded falls back to the photos when there are any."""
    job_type: Literal["tour_plan"]
    listing_id: str = Field(min_length=1)     # seeds the motion regions; in the plan
    photos: list[PlanPhoto] = Field(default_factory=list)
    scene: Optional[PlanScene] = None
    rooms: Optional[list[PlanRoom]] = None
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
        if self.scene is not None:
            if not self.rooms:
                raise ValueError("rooms: a 3D tour (scene) needs its rooms")
            keys = [r.room_key for r in self.rooms if r.room_key]
            if len(set(keys)) != len(keys):
                raise ValueError("rooms: room_key must be unique")
        elif not self.photos:
            raise ValueError("photos: at least one photo (or a scene and its rooms)")
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
    hevc_uri: Optional[str] = None            # the H.265 copy, when the plan asked for one
    mls_uri: Optional[str] = None             # the MLS copy (no captions), when the plan asked for one
    timings: Optional[dict[str, Any]] = None  # tour_worker.timing: seconds per stage, total, host
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
    timings: Optional[dict[str, Any]] = None  # tour_worker.timing: seconds per stage, total, host
    planner_3d: Optional[dict[str, Any]] = None   # a 3D tour: the scene as read, rooms, why each shot


class SegmentRegion(BaseModel):
    model_config = ConfigDict(extra="allow")
    object_id: str
    class_name: str
    confidence: float
    bbox: list[int]                           # [x1, y1, x2, y2] in `space` pixels
    space: list[int]                          # [w, h] the masks were made at
    mask_url: str


class SceneLabelResult(BaseModel):
    status: Literal["completed"]
    labels_uri: str
    labels_sha256: str
    motion_uri: Optional[str] = None          # None: nothing in the scene moves
    motion_sha256: Optional[str] = None
    objects: list[dict[str, Any]]             # each labelled object: class, centre, box, footprint, room
    stats: dict[str, Any]                     # views, rooms, labelled / seen Gaussians, per-class counts
    seconds: float
    timings: Optional[dict[str, Any]] = None


class TourSegmentResult(BaseModel):
    status: Literal["completed"]
    results: dict[str, Optional[list[SegmentRegion]]]   # photo_id -> regions (None: photo failed)
    scenes: dict[str, Optional[dict[str, Any]]]         # photo_id -> scene summary (None: failed)
    warnings: list[str]
    cached: list[str] = []                              # photo_ids whose known_version is current
    segment_version: Optional[str] = None               # what the results depend on (jobs.segment)
    model: str                                          # "<hf id>@<revision[:10]>"
    fireplace_model: Optional[str] = None               # (OneFormer era: the second fireplace pass; now unused)
    seconds: float
    timings: Optional[dict[str, Any]] = None            # tour_worker.timing: stages, total, host
