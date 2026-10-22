"""Output models. No numpy/cv2 imports, so a service that only consumes scores
(ranking, sequencing, shot planning) can import this without OpenCV installed."""
from pydantic import BaseModel, Field

from .config import ScoringConfig

CRITERIA = ("coverage", "framing", "lighting", "depth_of_field")


class CriterionResult(BaseModel):
    score: float | None
    details: dict[str, float | int | None] = Field(default_factory=dict)
    note: str | None = None


class MotionSignals(BaseModel):
    """Where to aim and how big a camera move a photo supports (motion.py). Not a score."""
    peak_x: float
    peak_y: float
    side: str                 # "left" | "right"
    mass_ratio: float
    spread: float
    spread_band: str          # "low" | "mid" | "high"
    room_size: str            # "large" | "medium" | "small"
    coverage_class: str       # "wide" | "medium" | "tight"
    # coarse saliency map (rows x grid_w, each row left to right, top row first, normalised so
    # the max cell is 1): lets the planner judge how interesting a camera window is
    grid: list[list[float]] = Field(default_factory=list)


class PhotoScore(BaseModel):
    photo_id: str
    room_type: str | None
    room_label: str | None = None
    room_key: str
    criteria: dict[str, CriterionResult]
    weights_used: dict[str, float]
    missing_criteria: list[str]
    total: float
    image_width: int | None = None
    image_height: int | None = None
    motion: MotionSignals | None = None
    flags: list[str] = Field(default_factory=list)


class RankedPhoto(BaseModel):
    photo_id: str
    total: float


class RoomResult(BaseModel):
    room_key: str
    room_type: str | None
    tier: str
    importance_rank: int
    screen_time_multiplier: float
    narrative_stage: str
    narrative_order: int
    hero_photo_id: str
    hero_total: float
    hero_flags: list[str]
    photos: list[RankedPhoto]


class ScoringReport(BaseModel):
    config: ScoringConfig
    photos: list[PhotoScore]
    rooms: list[RoomResult]                  # most important first
    sequence: list[str]                      # room_keys in narrative order
    warnings: list[str] = Field(default_factory=list)
