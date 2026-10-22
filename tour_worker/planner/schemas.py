from enum import Enum

from pydantic import BaseModel, Field


class CameraEasing(str, Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"


class OutputParams(BaseModel):
    width: int
    height: int
    fps: int
    codec: str = "h264"


class SourceImage(BaseModel):
    width: int
    height: int


class CameraPoint(BaseModel):
    cx: float
    cy: float
    zoom: float
    roll_deg: float = 0.0


class Keyframe(BaseModel):
    """keyframe_path point (plan 1.3). t in [0, 1]; roll_deg (addendum 1.3.3) is crop-only."""
    t: float
    cx: float
    cy: float
    zoom: float
    roll_deg: float | None = None


class Segment(BaseModel):
    """compound phase (addendum 1.3.2): move to (cx, cy, zoom) over path parameter [t0, t1]."""
    t0: float
    t1: float
    cx: float
    cy: float
    zoom: float


class Timing(BaseModel):
    s0: float
    s1: float
    turn: float | None = None


class CameraParams(BaseModel):
    """ken_burns (plan 1.0-1.2): start + end + easing.
    keyframe_path (1.3): move + keyframes (+ projection, hfov_deg).
    compound (1.3.2): move + start + segments + timing (+ projection, hfov_deg)."""
    type: str = "ken_burns"
    easing: CameraEasing = CameraEasing.EASE_IN_OUT
    preset: str | None = None
    start: CameraPoint | None = None
    end: CameraPoint | None = None
    move: str | None = None
    keyframes: list[Keyframe] | None = None
    segments: list[Segment] | None = None
    timing: Timing | None = None
    projection: str | None = None        # "crop" (default) | "rotate"
    hfov_deg: float | None = None
    # beat sync (plan 1.5): [output t, move t] points, monotone - the move planned for the shot's
    # normal length, retimed to its synced length (the rear speeds up / slows down to the cut)
    time_warp: list[list[float]] | None = None
    # beat sync: zoom punches on the music's hooks (a shot opening on a drop, an accent mid-shot)
    punches: list["Punch"] | None = None


class Punch(BaseModel):
    """A short zoom-in pulse on a musical hit: the camera's zoom x (1 + amount * pulse), the pulse
    rising over attack_s to 1 at at_s (s into the shot) and easing back to 0 over decay_s. Zoom
    in only, about the frame centre, so the frame stays inside the window the planner checked."""
    at_s: float
    amount: float
    attack_s: float = 0.0
    decay_s: float


class MotionRegion(BaseModel):
    """Perceptual motion inside an object mask (plan 1.4): the worker displaces the photo's pixels
    in the mask, in source-pixel space, before the camera warp. The mask is a full-frame 0/255 PNG
    of the detected image; the worker scales it to the source photo."""
    id: str
    effect: str                      # fabric_sway | foliage_sway | canopy_flutter | calm_water
    mask_uri: str
    bbox: list[float]                # [x1, y1, x2, y2], normalised to the photo (0-1)
    amplitude_px: float              # peak displacement, source px
    freq_hz: float
    anchor: str                      # "top" (hangs: curtains) | "bottom" (rooted: plants)
    seed: int
    label: str | None = None         # detect class name
    # sway phase (radians) for fabric_sway / foliage_sway: curtains and plants of a shot are set
    # half a cycle apart so they swing opposite ways (worker: random phase when None)
    phase: float | None = None


class TransitionParams(BaseModel):
    type: str = "cut"
    duration_s: float = 0.0


class Shot(BaseModel):
    shot_id: str
    order: int
    asset_id: str
    image_uri: str
    source: SourceImage
    room_key: str
    room_type: str | None = None
    narrative_stage: str
    role: str
    duration_s: float
    camera: CameraParams
    transition_in: TransitionParams = Field(default_factory=TransitionParams)
    motion_regions: list[MotionRegion] | None = None


class MusicTrack(BaseModel):
    """The tour's music (plan 1.5): the worker trims the (queued) track to end_s with a fade-out
    and muxes it. Cuts were snapped to its beats when synced (planner/music_sync.py)."""

    track_id: str
    uri: str                         # s3:// URI of the track
    sha256: str
    analysis_version: str
    tempo_bpm: float
    start_s: float = 0.0
    end_s: float                     # = the tour's length
    track_duration_s: float
    loops: int = 1                   # copies of the track queued back to back
    loop_s: float | None = None      # each copy plays [0, loop_s) (last bar before the music ends)
    fade_out_s: float
    synced: bool
    fallback_reason: str | None = None
    cuts_on_bar: int | None = None   # boundaries (incl. the tour end) on a bar start
    cut_lead_frames: int = 0         # cuts sit this many frames before their beat
    hooks_required: bool = False     # every hook moment was given a cut (else only preferred)
    hooks: list["MusicHook"] | None = None   # the track's hooks inside the tour and what got them


class MusicHook(BaseModel):
    t: float                         # s, tour time of the hook (before the cut lead)
    kind: str                        # "cut" | "accent" (zoom punch, no cut) | "covered" (its
    #                                  moment's cut / accent is within merge_s) | "missed"
    reason: str | None = None        # accent / missed: "merged" | "too_close_to_start" | "no_fit"


class PlanProvenance(BaseModel):
    scoring_config_version: str
    planner_config_version: str
    plan_hash: str | None = None


class ShotPlan(BaseModel):
    plan_version: str = "1.2"
    listing_id: str
    mode: str
    fallback_reason: str | None = None
    preset: str
    output: OutputParams
    total_duration_s: float
    shots: list[Shot]
    warnings: list[str] = Field(default_factory=list)
    provenance: PlanProvenance
    music: MusicTrack | None = None

def build_tour_render_request(
    plan: ShotPlan, 
    job_id: str, 
    output_prefix: str, 
    image_urls: dict[str, str] | None = None
) -> dict:
    """Build the JSON payload for the AI worker."""
    req = {
        "job_id": job_id,
        "job_type": "tour_render",
        "plan": plan.model_dump(mode="json"),
        "output_prefix": output_prefix
    }
    if image_urls is not None:
        req["image_urls"] = image_urls
        
    return {"input": req}
