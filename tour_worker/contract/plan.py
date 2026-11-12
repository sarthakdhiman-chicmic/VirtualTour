import hashlib
import json

from tour_worker.render.camera_path import PROJECTIONS, CompoundPath, KeyframePath, validate_compound, frame_margin_out_px, validate_keyframes
import math
from typing import List, Optional, Any, Union
from pydantic import BaseModel, ConfigDict, Field

# room labels (overlays, SOW 3.2.1a; user 2026-11-05): the room's name in a script font, drawn by
# tour_worker/render/overlays.py. The fonts ship in tour_worker/render/fonts (SIL OFL).
LABEL_FONTS = ("sacramento", "mr_dafoe", "pacifico", "kaushan_script")
LABEL_PLACEMENTS = ("bottom_left", "bottom_center", "bottom_right", "top_left", "top_center", "top_right")
LABEL_SCALE_RANGE = (0.6, 1.0)          # x the reviewed maximum size; the user's default is 0.8
MAX_CAPTION_CHARS = 40

class RoomLabels(BaseModel):
    font: str = "sacramento"
    scale: float = 0.8
    placement: str = "bottom_left"
    model_config = ConfigDict(extra='ignore')

# branding and finishing (client review 2026-11-05: "no title card, address, branding or end card"),
# drawn by tour_worker/render/overlays.py; the MLS copy is the unbranded version (no agent card,
# logo or room labels; the address title and the staged disclosure stay)
MAX_BRAND_CHARS = 120

class AgentCard(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    photo_uri: Optional[str] = None    # fetched like the photos (image_urls), drawn as a circle
    model_config = ConfigDict(extra='ignore')

class Branding(BaseModel):
    kicker: Optional[str] = None       # a line above the address ("Just Listed")
    address: Optional[str] = None      # the title card's first line
    locality: Optional[str] = None     # city, state - its second line
    title: Optional[str] = None        # the request's intro text, under the address
    agent: Optional[AgentCard] = None  # the end card (branded copy only)
    logo_uri: Optional[str] = None     # the brokerage logo: on the end card and in the corner
    corner_logo: bool = False
    model_config = ConfigDict(extra='ignore')

    def image_uris(self) -> list[str]:
        return [u for u in (self.agent.photo_uri if self.agent else None, self.logo_uri) if u]

class OutputConfig(BaseModel):
    width: int
    height: int
    fps: int
    codec: str
    max_bytes: Optional[int] = None    # a platform's file-size limit: the encoder caps its bitrate to fit
    hevc: bool = False                 # also render the optional H.265 copy (render/direct.py)
    room_labels: Optional[RoomLabels] = None   # draw each shot's caption (render/overlays.py)
    mls_copy: bool = False             # also render the MLS copy: no captions, the staged label kept
    branding: Optional[Branding] = None   # title card, end card, corner logo (render/overlays.py)
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

class BezierSegment(BaseModel):
    """Plan 1.6 bezier_path: one cubic Bezier for the eye and one for the point it looks at, four
    control points each, in scene units (tour_worker/scene). Consecutive segments join end to start."""
    eye: List[List[float]]
    target: List[List[float]]
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
    # photo's assumed horizontal field of view for "rotate" (tour_worker/render/camera_path.py)
    projection: Optional[str] = None
    # keyframe_path / compound: one value; bezier_path (plan 1.6): [start, end]
    hfov_deg: Optional[Union[float, List[float]]] = None
    # compound (addendum 1.3.2): start + segments + timing (tour_worker/render/camera_path.py);
    # bezier_path (plan 1.6): 1-3 BezierSegments
    segments: Optional[List[Union[Segment, BezierSegment]]] = None
    timing: Optional[Timing] = None
    # beat sync (plan 1.5): [output t, move t] points, monotone, (0,0) .. (1,1): the move planned
    # for the shot's normal length, retimed to its synced length (tour_worker/render/camera_path.time_warp_at)
    time_warp: Optional[List[List[float]]] = None
    # beat sync: zoom punches on musical hits (tour_worker/render/camera_path.punch_scale)
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

# cut (the default, on the beat); fade; whip (both shots slide sideways with motion blur) and push
# (zoom through into the next room) - beat-timed motion transitions centred on the cut (client
# review 2026-11-05; tour_worker/render/direct.py draws them, planner/transitions.py chooses)
TRANSITIONS = ("cut", "fade", "whip", "push")

class Transition(BaseModel):
    type: str
    duration_s: float
    model_config = ConfigDict(extra='ignore')

MOTION_EFFECTS = ("fabric_sway", "foliage_sway", "water_ripple", "calm_water", "canopy_flutter", "fire_flicker",
                  "sky_drift")
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
    image_uri: Optional[str] = None   # every photo shot has one; a 3D shot (plan 1.6) has none
    source: Optional[Size] = None     # likewise
    room_key: str
    room_type: Optional[str] = None   # the backend allows untagged rooms
    narrative_stage: str
    role: str
    duration_s: float
    camera: Camera
    transition_in: Transition
    motion_regions: Optional[List[MotionRegion]] = None   # plan 1.4
    caption: Optional[str] = None     # the room label, on a room's first shot (output.room_labels)
    staged: bool = False              # the photo is virtually staged: the disclosure label is drawn
    model_config = ConfigDict(extra='ignore')

class Provenance(BaseModel):
    scoring_config_version: str
    planner_config_version: str
    plan_hash: str
    model_config = ConfigDict(extra='ignore')

class MusicSegment(BaseModel):
    """A piece of the track (s, track time); the tour plays the segments back to back."""
    from_s: float
    to_s: float
    model_config = ConfigDict(extra='ignore')

class Music(BaseModel):
    """Plan 1.5: the tour's music (tour_worker/planner/music_sync.py). The track is queued `loops`
    times, each copy cut at loop_s (the bar where its beat stops), trimmed to end_s, faded out."""
    track_id: str
    uri: str
    sha256: Optional[str] = None
    start_s: float = 0.0                   # where in the track the tour's music starts (sync v2)
    # sync v2 music edits: the track pieces played back to back (bar-aligned, joined with a
    # crossfade_s equal-power crossfade centred on each join); when set, start_s / loops /
    # loop_s only describe it - the renderer plays the segments
    segments: Optional[List[MusicSegment]] = None
    crossfade_s: float = 0.03
    edits: Optional[List[dict]] = None     # which bars came out and why (information only)
    end_s: float
    loops: int = 1
    loop_s: Optional[float] = None
    track_duration_s: Optional[float] = None
    fade_out_s: float = 1.5
    synced: bool = False
    cut_lead_frames: int = 0               # cuts sit this many frames before their beat
    hooks_required: bool = False           # every hook moment got a cut (tools/quality/sync_quality.py)
    hooks: Optional[List[dict]] = None     # [{t, kind: cut|accent|missed, reason}]
    model_config = ConfigDict(extra='ignore')

class Voiceover(BaseModel):
    """A narration over the tour (client review 2026-11-05): it plays from the tour's start, the music
    ducks under it by `balance` (0: barely, 1: far down), and the mix is normalised like the music."""
    uri: str
    balance: float = 0.5
    model_config = ConfigDict(extra='ignore')

# plan 1.6: a tour built from a Gaussian splat scene (mode cinematic_3d, bezier_path cameras). The
# backend passes the scene file's link; scene.json (rooms and anchors) is optional.
SCENE_FORMATS = ("ply", "splat")
MODES = ("cinematic", "slideshow", "cinematic_3d")
FALLBACK_REASONS_3D = ("no_scene_model", "scene_invalid", "scene_render_failed")
MAX_BEZIER_SEGMENTS = 3

class Scene(BaseModel):
    uri: str
    sha256: str
    format: str                        # "ply" | "splat"
    json_uri: Optional[str] = None     # scene.json, when an operator authored one
    json_sha256: Optional[str] = None
    up_axis: Optional[List[float]] = None   # the scene's up in its own axes; None: +z
    # 3D motion (curtains sway, plants stir): a labels file made once per scene by
    # tools/scene/label_motion.py (tour_worker/scene/motion3d.py); None: the scene is still
    motion_uri: Optional[str] = None
    motion_sha256: Optional[str] = None
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
    voiceover: Optional[Voiceover] = None
    scene: Optional[Scene] = None      # plan 1.6
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
                "1.5": {"ken_burns", "keyframe_path", "compound"},   # 1.5 = 1.4 + a music block
                "1.6": {"bezier_path"}}                              # 1.6 = a 3D scene tour
PLAN_VERSIONS = tuple(CAMERA_TYPES)

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
        if shot.camera.type == "compound" and not all(isinstance(s, Segment) for s in shot.camera.segments or []):
            raise ValueError(f"invalid_plan: compound camera in shot {shot.shot_id} has bezier segments")
        if shot.camera.type == "bezier_path":
            check_bezier_camera(shot)
        if shot.camera.type in ("keyframe_path", "compound"):
            if shot.camera.hfov_deg is not None and not isinstance(shot.camera.hfov_deg, (int, float)):
                raise ValueError(f"invalid_plan: hfov_deg in shot {shot.shot_id} must be one number")
            if shot.camera.projection is not None and shot.camera.projection not in PROJECTIONS:
                raise ValueError(f"unsupported_projection: {shot.camera.projection}")
            if (shot.camera.type == "keyframe_path" and shot.camera.projection == "rotate"
                    and any(k.roll_deg for k in shot.camera.keyframes or [])):
                raise ValueError(f"invalid_plan: roll_deg in shot {shot.shot_id} is crop-only")
            if shot.camera.hfov_deg is not None and not 20 <= shot.camera.hfov_deg <= 120:
                raise ValueError(f"invalid_plan: hfov_deg in shot {shot.shot_id} must be 20-120")

def _finite_points(points, n: int) -> bool:
    return (isinstance(points, list) and len(points) == n
            and all(isinstance(p, list) and len(p) == 3 and all(math.isfinite(v) for v in p) for p in points))

def check_bezier_camera(shot):
    """bezier_path (plan 1.6): 1-3 segments of 4 finite 3D control points for the eye and for the
    target, joined end to start; hfov_deg [start, end] within 20-120; no projection, no punches."""
    c, sid = shot.camera, shot.shot_id
    segs = c.segments or []
    if not 1 <= len(segs) <= MAX_BEZIER_SEGMENTS or not all(isinstance(s, BezierSegment) for s in segs):
        raise ValueError(f"invalid_plan: bezier_path in shot {sid} needs 1-{MAX_BEZIER_SEGMENTS} eye/target segments")
    for s in segs:
        if not (_finite_points(s.eye, 4) and _finite_points(s.target, 4)):
            raise ValueError(f"invalid_plan: bezier_path in shot {sid} needs 4 finite 3D points per curve")
    for a, b in zip(segs, segs[1:]):
        if any(abs(u - v) > 1e-6 for p, q in ((a.eye[3], b.eye[0]), (a.target[3], b.target[0])) for u, v in zip(p, q)):
            raise ValueError(f"invalid_plan: bezier_path segments in shot {sid} must join end to start")
    h = c.hfov_deg
    if not (isinstance(h, list) and len(h) == 2 and all(isinstance(v, (int, float)) and 20 <= v <= 120 for v in h)):
        raise ValueError(f"invalid_plan: bezier_path hfov_deg in shot {sid} must be [start, end] within 20-120")
    if c.projection is not None or c.punches:
        raise ValueError(f"invalid_plan: bezier_path in shot {sid} takes no projection or punches")

def check_scene(plan: Plan):
    """Plan 1.6 is a 3D scene tour: a scene block, mode cinematic_3d, every shot a bezier_path with
    no photo, source size or motion regions. Earlier versions have photo shots and no scene."""
    if plan.plan_version == "1.6":
        sc = plan.scene
        if sc is None or plan.mode != "cinematic_3d":
            raise ValueError("invalid_plan: plan_version 1.6 needs a scene and mode cinematic_3d")
        if sc.format not in SCENE_FORMATS:
            raise ValueError(f"invalid_plan: scene.format must be one of {', '.join(SCENE_FORMATS)}")
        for u in (sc.uri, sc.json_uri, sc.motion_uri):
            if u is not None and not u.startswith(("s3://", "http://", "https://", "local://", "file://")):
                raise ValueError(f"invalid_plan: scene uri {u!r}")
        for h in (sc.sha256, sc.json_sha256, sc.motion_sha256):
            if h is not None and (len(h) != 64 or any(ch not in "0123456789abcdef" for ch in h)):
                raise ValueError("invalid_plan: scene sha256 must be 64 lowercase hex characters")
        if sc.up_axis is not None and not (len(sc.up_axis) == 3 and all(math.isfinite(v) for v in sc.up_axis)
                                           and math.hypot(*sc.up_axis) > 1e-6):
            raise ValueError("invalid_plan: scene.up_axis must be a non-zero 3D vector")
        if (sc.json_uri is None) != (sc.json_sha256 is None):
            raise ValueError("invalid_plan: scene.json_uri and scene.json_sha256 go together")
        if (sc.motion_uri is None) != (sc.motion_sha256 is None):
            raise ValueError("invalid_plan: scene.motion_uri and scene.motion_sha256 go together")
        for shot in plan.shots:
            if shot.image_uri is not None or shot.source is not None or shot.motion_regions:
                raise ValueError(f"invalid_plan: 3D shot {shot.shot_id} has no image_uri, source or motion_regions")
    else:
        if plan.scene is not None or plan.mode == "cinematic_3d":
            raise ValueError(f"invalid_plan: a scene and mode cinematic_3d need plan_version 1.6")
        for shot in plan.shots:
            if shot.image_uri is None or shot.source is None:
                raise ValueError(f"invalid_plan: shot {shot.shot_id} needs image_uri and source")

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
    adds its own 1 px margin on top). Frames are the shot's render span, as in tour_worker.render.timeline."""
    from tour_worker.render.timeline import compute_timeline
    timeline = compute_timeline(plan)
    for shot, tl in zip(plan.shots, timeline):
        if shot.source is None:            # a 3D shot: no photo to stay inside
            continue
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
        if shot.source is None:            # a 3D shot: no photo to stay inside
            continue
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

def check_music_segments(plan: Plan):
    m = plan.music
    if m is None or not m.segments:
        return
    total = 0.0
    for k, s in enumerate(m.segments):
        if not (0 <= s.from_s < s.to_s) or (k > 0 and s.from_s < m.crossfade_s / 2):
            raise ValueError("invalid_plan: music segment out of range")
        total += s.to_s - s.from_s
    if not 0 <= m.crossfade_s <= 0.2 or total < m.end_s - 0.05:
        raise ValueError("invalid_plan: music segments shorter than the tour, or crossfade out of range")

def check_easing(plan: Plan):
    valid_easings = {"linear", "ease_in", "ease_out", "ease_in_out"}
    for shot in plan.shots:
        if shot.camera.type in ("keyframe_path", "compound"):
            continue  # fixed global time profile (tour_worker/render/camera_path.py)
        if shot.camera.easing not in valid_easings:
            raise ValueError(f"unsupported_easing: {shot.camera.easing}")
        if plan.plan_version in ("1.0", "1.1") and shot.camera.easing != "ease_in_out":
            raise ValueError(f"invalid_plan: non-default easing '{shot.camera.easing}' not allowed in plan_version {plan.plan_version}")

def check_transitions(plan: Plan):
    if plan.plan_version not in PLAN_VERSIONS:
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
        else:
            t = shot.transition_in
            if t.type not in TRANSITIONS:
                raise ValueError(f"Unsupported transition type: {t.type}")
            if t.type != "cut" and t.duration_s > shot.duration_s * 0.5:
                raise ValueError(f"{t.type} duration cannot exceed 50% of shot duration")

def check_motion_regions(plan: Plan):
    for shot in plan.shots:
        if not shot.motion_regions:
            continue
        if plan.plan_version not in ("1.4", "1.5", "1.6"):
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

def check_overlays(plan: Plan):
    labels = plan.output.room_labels
    if labels is not None:
        if labels.font not in LABEL_FONTS:
            raise ValueError(f"invalid_plan: room_labels.font must be one of {', '.join(LABEL_FONTS)}")
        if labels.placement not in LABEL_PLACEMENTS:
            raise ValueError(f"invalid_plan: room_labels.placement must be one of {', '.join(LABEL_PLACEMENTS)}")
        lo, hi = LABEL_SCALE_RANGE
        if not lo <= labels.scale <= hi:
            raise ValueError(f"invalid_plan: room_labels.scale must be within {lo}-{hi}")
    for shot in plan.shots:
        if shot.caption is None:
            continue
        if labels is None:
            raise ValueError(f"invalid_plan: shot {shot.shot_id} has a caption but the plan has no room_labels")
        if not shot.caption.strip() or len(shot.caption) > MAX_CAPTION_CHARS:
            raise ValueError(f"invalid_plan: shot {shot.shot_id} caption must be 1-{MAX_CAPTION_CHARS} characters")

def check_voiceover(plan: Plan):
    v = plan.voiceover
    if v is None:
        return
    if not 0.0 <= v.balance <= 1.0:
        raise ValueError("invalid_plan: voiceover.balance must be within 0-1")
    if not v.uri.startswith(("s3://", "http://", "https://", "local://", "file://")):
        raise ValueError(f"invalid_plan: voiceover uri {v.uri!r}")

def check_branding(plan: Plan):
    b = plan.output.branding
    if b is None:
        return
    texts = [b.kicker, b.address, b.locality, b.title] + ([b.agent.name, b.agent.phone, b.agent.email] if b.agent else [])
    if any(t is not None and len(t) > MAX_BRAND_CHARS for t in texts):
        raise ValueError(f"invalid_plan: branding text must be at most {MAX_BRAND_CHARS} characters")
    for u in b.image_uris():
        if not u.startswith(("s3://", "http://", "https://", "local://")):
            raise ValueError(f"invalid_plan: branding image uri {u!r}")

def parse_plan(data: dict) -> Plan:
    validate_plan_hash(data)
    plan = Plan(**data)
    check_scene(plan)
    check_camera_type(plan)
    check_bounds(plan)
    check_easing(plan)
    check_transitions(plan)
    check_time_warps(plan)
    check_punches(plan)
    check_path_bounds(plan)
    check_motion_regions(plan)
    check_music_segments(plan)
    check_overlays(plan)
    check_branding(plan)
    check_voiceover(plan)
    return plan


def plan_failure(e: ValueError) -> dict:
    """The job failure for a plan parse_plan rejected: unsupported_camera_type, unsupported_easing,
    or invalid_plan for anything else (the message is the full text)."""
    from tour_worker.errors import failure
    msg = str(e)
    for code in ("unsupported_camera_type", "unsupported_easing"):
        if code in msg:
            return failure(code, msg)
    return failure("invalid_plan", msg)
