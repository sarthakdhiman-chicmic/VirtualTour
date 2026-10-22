"""All tunable scoring parameters. Every number is a PROPOSAL (the SOW names the four
criteria but gives no weights or thresholds). The full config is echoed in every report."""
from pydantic import BaseModel, Field


class Weights(BaseModel):
    # Coverage is a single-photo heuristic (no pose/segmentation) so it gets the lowest weight
    # until it is validated against human judgement. Lighting is the most reliable measurement.
    coverage: float = 0.20
    framing: float = 0.30
    lighting: float = 0.30
    depth_of_field: float = 0.20


class FramingConfig(BaseModel):
    thirds_sigma: float = 0.22
    thirds_weight: float = 0.6
    tilt_weight: float = 0.4
    tilt_sigma_deg: float = 2.0
    max_vertical_search_deg: float = 15.0
    min_vertical_lines: int = 3
    saliency_percentile: float = 80.0
    min_contrast_std: float = 0.01


class LightingConfig(BaseModel):
    blown_threshold: float = 0.98
    blown_limit: float = 0.05
    crushed_threshold: float = 0.02
    crushed_limit: float = 0.10
    brightness_low: float = 0.35
    brightness_high: float = 0.70
    evenness_limit: float = 0.25
    dynamic_range_target: float = 0.45
    w_blown: float = 0.30
    w_crushed: float = 0.20
    w_brightness: float = 0.20
    w_evenness: float = 0.15
    w_range: float = 0.15
    severity_gate: float = 0.8


class DepthOfFieldConfig(BaseModel):
    sharpness_k: float = 40.0
    edge_threshold: float = 8.0
    min_edge_fraction: float = 0.005
    top_fraction: float = 0.10
    min_bands: int = 2
    min_weight: float = 0.6
    mean_weight: float = 0.4


class CoverageConfig(BaseModel):
    """Single-photo coverage heuristic: how much of the room the frame can reveal."""
    # Camera field of view from EXIF (35mm-equivalent focal length -> diagonal FOV)
    fov_low_deg: float = 45.0        # <= this scores 0   (about a 50 mm equivalent)
    fov_high_deg: float = 100.0      # >= this scores 1   (about an 18 mm equivalent)
    # Structure cues from long straight edges
    horizontal_max_angle_deg: float = 20.0
    horizontal_min_len_frac: float = 0.25    # of image width
    top_zone: float = 0.30                   # top 30% of rows = "ceiling zone"
    bottom_zone: float = 0.35                # bottom 35% of rows = "floor zone"
    edge_span_target: float = 0.60           # summed edge length / width for a full zone score
    vertical_max_angle_deg: float = 10.0
    vertical_min_len_frac: float = 0.30      # of image height
    plane_merge_gap_frac: float = 0.04       # verticals closer than 4% of width are one corner
    plane_clusters_for_full: int = 4
    w_fov: float = 0.6
    w_ceiling_floor: float = 0.2
    w_wall_planes: float = 0.2
    min_content_std: float = 3.0      # gray-level std (0-255) below which the frame is considered blank


class MotionConfig(BaseModel):
    """Motion signals for the shot planner (motion.py). Saliency parameters match the tour
    worker's tools/tour_v2/saliency.py, so plans see the same numbers the v8 tours were tuned on."""
    sr_size: int = 64
    sr_box: int = 3
    sr_blur_ksize: int = 5
    sr_blur_sigma: float = 8.0
    analysis_w: int = 480
    smooth_sigma_frac: float = 0.02
    top_fraction: float = 0.05
    border_fraction: float = 0.05
    min_content_std: float = 3.0
    grid_w: int = 32                 # saliency grid width for the planner (rows follow the aspect)
    # coverage matrix: room size (from room type) x spread band -> coverage class.
    # Spread bands are provisional: calibrated on the 29 photos of two listings.
    spread_low: float = 0.37         # spread < this is "low"
    spread_high: float = 0.42        # spread >= this is "high"
    matrix: dict[str, dict[str, str]] = {
        "large": {"low": "medium", "mid": "wide", "high": "wide"},
        "medium": {"low": "tight", "mid": "medium", "high": "wide"},
        "small": {"low": "tight", "mid": "tight", "high": "medium"},
    }
    large_rooms: tuple[str, ...] = (
        "living_room", "family_room", "great_room", "open_concept", "sunroom", "kitchen",
        "basement", "bonus_room", "exterior", "front_exterior", "facade", "aerial", "backyard",
        "patio", "outdoor_patio", "deck", "pool_area", "garden", "balcony",
    )
    small_rooms: tuple[str, ...] = (
        "bedroom", "bed_room", "secondary_bedroom", "guest_bedroom", "childrens_bedroom", "nursery",
        "bathroom", "primary_bathroom", "master_bathroom", "secondary_bathroom", "ensuite",
        "powder_room", "entry", "entryway", "foyer", "hallway", "mudroom", "staircase", "stairs",
        "stairway", "closet", "walk_in_closet", "laundry_room", "utility_room", "butlers_pantry",
    )
    # EXIF field of view (coverage.py fov_score) bumps the class one step when present
    fov_wide_bump: float = 0.7
    fov_narrow_bump: float = 0.2


class RankingConfig(BaseModel):
    hero_room_count: int = 3
    hero_multiplier: float = 2.0
    exterior_multiplier: float = 1.5


class ScoringConfig(BaseModel):
    version: str = "0.3.0"
    weights: Weights = Field(default_factory=Weights)
    analysis_long_side: int = 1024
    min_acceptable_score: float = 0.40
    framing: FramingConfig = Field(default_factory=FramingConfig)
    lighting: LightingConfig = Field(default_factory=LightingConfig)
    depth_of_field: DepthOfFieldConfig = Field(default_factory=DepthOfFieldConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
