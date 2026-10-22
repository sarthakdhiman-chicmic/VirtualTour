import hashlib
import json
from typing import Any

from tour_worker.scoring.rooms import canon
from tour_worker.scoring.schemas import ScoringReport

from .allocate import SelectedShot, allocate_shots
from .camera import generate_ken_burns
from .cinematic import ShotInput, plan_cameras
from .config import PlannerConfig
from .music_sync import apply_sync, sync_to_music
from .perceptual import plan_motion_regions
from .presets import LATERAL_PRESETS, preset_to_camera, select_preset
from .scene_context import blend_grid, from_scene
from .schemas import (
    CameraEasing,
    CameraParams,
    CameraPoint,
    OutputParams,
    PlanProvenance,
    Shot,
    ShotPlan,
    SourceImage,
    TransitionParams,
)


def _compute_plan_hash(plan_dict: dict) -> str:
    # Remove provenance.plan_hash if exists
    if "provenance" in plan_dict and "plan_hash" in plan_dict["provenance"]:
        plan_dict["provenance"].pop("plan_hash", None)

    canonical_json = json.dumps(
        plan_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_shot_plan(
    report: ScoringReport,
    listing_id: str,
    assets: dict[str, dict[str, Any]],
    preset: str = "property_showcase",
    budget_blocked: bool = False,
    cfg: PlannerConfig | None = None,
    detections_by_photo: dict[str, list[dict] | None] | None = None,
    music_track: dict | None = None,
    scenes_by_photo: dict[str, dict | None] | None = None,
) -> ShotPlan:
    """detections_by_photo: photo_id -> mask regions (the tour_segment job, OneFormer), used
    for perceptual motion regions (plan 1.4). None or missing photos: no regions.
    music_track: {track_id, s3_uri, sha256, analysis} from the music library: shot lengths are
    snapped to its beats before the cameras are planned (plan 1.5, planner/music_sync.py).
    scenes_by_photo: photo_id -> OneFormer scene summary (segment_photos scenes_out): what each
    photo shows drives shot lengths and camera moves (planner/scene_context.py); without it the
    saliency signals alone are used."""
    if cfg is None:
        cfg = PlannerConfig()

    warnings = []

    # 9. Presets handling
    if preset not in ("property_showcase",):
        warnings.append("preset_not_defined")

    # Unclassified photos warning
    unclassified = sum(1 for p in report.photos if p.room_type is None)
    if unclassified > 0:
        warnings.append(f"unclassified_photos:{unclassified}")

    # 1. Mode determination
    num_photos = len(report.photos)
    distinct_rooms = len({p.room_type for p in report.photos if p.room_type})

    mode = "cinematic"
    fallback_reason = None

    if budget_blocked:
        mode = "slideshow"
        fallback_reason = "budget_exceeded"
    elif num_photos < cfg.min_photos:
        mode = "slideshow"
        fallback_reason = "insufficient_photos"
    elif distinct_rooms < cfg.min_room_types:
        mode = "slideshow"
        fallback_reason = "insufficient_room_types"

    # 2. Shot Selection
    contexts = {}
    for p in report.photos:
        m = p.motion
        contexts[p.photo_id] = from_scene(
            (scenes_by_photo or {}).get(p.photo_id),
            *((m.peak_x, m.peak_y, m.spread) if m else (0.5, 0.5, 0.4)),
        )

    shots_to_use: list[SelectedShot] = []
    if mode == "cinematic":
        shots_to_use, alloc_warnings = allocate_shots(report, cfg, contexts)
        for w in alloc_warnings:
            if w not in warnings:
                warnings.append(w)
    else:
        # Slideshow mode: up to slideshow_max_shots best photos (heroes first, narrative order)
        # We need heroes in narrative order, then remaining photos in narrative order,
        # sorted by score.
        heroes = []
        for r_key in report.sequence:
            room = next((r for r in report.rooms if r.room_key == r_key), None)
            if room:
                heroes.append((room, room.hero_photo_id))

        # Build list of candidates.
        # Determinism: ties broken by (score desc, photo_id). Actually report.photos
        # is already ordered? No.
        # But we sort here.
        hero_ids = {h[1] for h in heroes}
        candidates = []
        for h_room, h_id in heroes:
            candidates.append(
                SelectedShot(
                    h_id,
                    h_room.room_key,
                    "hero",
                    cfg.base_shot_s,
                    is_hero=True,
                    narrative_order=h_room.narrative_order,
                )
            )

        remaining = []
        for room in report.rooms:
            for p in room.photos:
                if p.photo_id not in hero_ids:
                    remaining.append((room, p))

        # Sort remaining by (narrative_order, -score, photo_id)
        remaining.sort(key=lambda x: (x[0].narrative_order, -x[1].total, x[1].photo_id))
        for r_room, r_p in remaining:
            candidates.append(
                SelectedShot(
                    r_p.photo_id,
                    r_room.room_key,
                    "detail",
                    cfg.base_shot_s,
                    is_hero=False,
                    narrative_order=r_room.narrative_order,
                )
            )

        max_allowed_shots = min(
            cfg.slideshow_max_shots,
            int(cfg.max_total_s // cfg.shot_min_s),
        )
        shots_to_use = candidates[:max_allowed_shots]

        # Allocate durations within [min_total_s, max_total_s]
        if shots_to_use:
            target_dur = max(
                cfg.min_total_s,
                min(cfg.max_total_s, len(shots_to_use) * cfg.base_shot_s),
            )
            per_shot = round(target_dur / len(shots_to_use), 2)
            per_shot = max(cfg.shot_min_s, min(cfg.shot_max_s, per_shot))

            # While total exceeds max_total_s, trim lowest priority shots
            while len(shots_to_use) * per_shot > cfg.max_total_s and len(shots_to_use) > 1:
                shots_to_use.pop()

            for i, s in enumerate(shots_to_use):
                s.order_idx = i + 1
                s.final_duration = per_shot

    # Build Output
    output_params = OutputParams(
        width=cfg.output_width, height=cfg.output_height, fps=cfg.fps, codec="h264"
    )

    # 3. Assemble Plan
    final_shots: list[Shot] = []
    total_dur = 0.0
    prev_selected_preset: str | None = None
    lateral_run = 0
    last_lateral: str | None = None
    wide_rooms = frozenset(cfg.motion_wide_rooms)
    tall_rooms = frozenset(cfg.motion_tall_rooms)
    small_rooms = frozenset(cfg.motion_small_rooms)

    for idx, sel_shot in enumerate(shots_to_use):
        # find photo_score
        ps = next((p for p in report.photos if p.photo_id == sel_shot.photo_id), None)
        if not ps:
            continue

        # extract dimensions and canonical uri from assets dict
        asset_info = assets.get(sel_shot.photo_id, {})
        sw = asset_info.get("width")
        sh = asset_info.get("height")
        canonical_uri = asset_info.get("uri", "")

        if not sw or not sh:
            if "source_dims_missing" not in warnings:
                warnings.append("source_dims_missing")
            # fallback static camera
            sw, sh = cfg.output_width, cfg.output_height

        # extract framing
        sub_x, sub_y = 0.5, 0.5
        framing_details = ps.criteria.get("framing")
        if framing_details and framing_details.details:
            if "subject_x" in framing_details.details and "subject_y" in framing_details.details:
                sub_x = float(framing_details.details["subject_x"] or 0.5)
                sub_y = float(framing_details.details["subject_y"] or 0.5)

        if cfg.use_motion_presets:
            shot_id = f"s{sel_shot.order_idx:02d}"

            selected_preset = select_preset(
                idx=idx,
                total_shots=len(shots_to_use),
                role=sel_shot.role,
                order=sel_shot.order_idx,
                prev_preset=prev_selected_preset,
                cycle=cfg.motion_preset_cycle,
                first_preset=cfg.motion_preset_first,
                last_preset=cfg.motion_preset_last,
                hero_preferred=cfg.motion_preset_hero_preferred,
                hero_alternate=cfg.motion_preset_hero_alternate,
                detail_preferred=cfg.motion_preset_detail_preferred,
                detail_alternate=cfg.motion_preset_detail_alternate,
                room_type=ps.room_type,
                lateral_run=lateral_run,
                last_lateral=last_lateral,
                wide_rooms=wide_rooms,
                tall_rooms=tall_rooms,
                max_lateral_run=cfg.motion_max_lateral_run,
                tall_preferred=cfg.motion_preset_tall_preferred,
                tall_alternate=cfg.motion_preset_tall_alternate,
                hero_sweep=cfg.motion_preset_hero_sweep,
            )
            prev_selected_preset = selected_preset

            room_speed = (
                cfg.motion_speed_small if canon(ps.room_type) in small_rooms else cfg.motion_speed
            )
            speed = room_speed * min(1.0, sel_shot.final_duration / cfg.motion_reference_shot_s)

            cam_dict, fallback_label = preset_to_camera(
                preset=selected_preset,
                src_w=sw,
                src_h=sh,
                out_w=cfg.output_width,
                out_h=cfg.output_height,
                subject_x=sub_x,
                subject_y=sub_y,
                order=sel_shot.order_idx,
                max_zoom=cfg.motion_max_zoom,
                zoom_step=cfg.motion_zoom_step,
                scale_factors=cfg.motion_scale_factors,
                subject_x_high=cfg.motion_subject_x_high,
                subject_x_low=cfg.motion_subject_x_low,
                speed=speed,
            )

            if cam_dict["preset"] in LATERAL_PRESETS:
                lateral_run += 1
                last_lateral = cam_dict["preset"]
            else:
                lateral_run = 0

            if fallback_label != "none":
                warnings.append(f"preset_downgraded:{shot_id}:{fallback_label}")

            cam_params = CameraParams(**cam_dict)
        else:
            # Variety based on order_idx (4 distinct cinematic moves:
            # dramatic push-in, sweeping pan, gentle push+vertical pan, diagonal drift)
            variety = sel_shot.order_idx % 4
            ex, ey = sub_x, sub_y

            # Deterministic delta based on role
            if idx == len(shots_to_use) - 1:
                delta = cfg.kb_final_zoom_delta * 2.0  # Make it more dramatic
            elif sel_shot.is_hero:
                delta = cfg.kb_hero_zoom_delta * 1.5
            else:
                delta = cfg.kb_zoom_delta * 1.5

            if variety == 0:
                # Variety 0: Dramatic Push-In (Center to Subject)
                sx, sy = 0.5, 0.5
                z_start = 1.0
                z_end = 1.0 + delta + 0.10
            elif variety == 1:
                # Variety 1: Sweeping Horizontal Pan (Constant Zoom)
                sx = 0.2 if sub_x >= 0.5 else 0.8
                sy = sub_y
                z_start = 1.15
                z_end = 1.15 + (delta * 0.5)
            elif variety == 2:
                # Variety 2: Gentle Push-In + Vertical Pan
                sx = sub_x
                sy = 0.2 if sub_y >= 0.5 else 0.8
                z_start = 1.10
                z_end = 1.10 + delta
            else:
                # Variety 3: Slow Diagonal Drift
                sx = 0.2 if sub_x >= 0.5 else 0.8
                sy = 0.2 if sub_y >= 0.5 else 0.8
                z_start = 1.15
                z_end = 1.15 + (delta * 0.5)

            cam_params, cam_warning = generate_ken_burns(
                source_w=sw,
                source_h=sh,
                output_w=cfg.output_width,
                output_h=cfg.output_height,
                start_x=sx,
                start_y=sy,
                end_x=ex,
                end_y=ey,
                zoom_start=z_start,
                zoom_end=z_end,
            )
            if cam_warning and cam_warning not in warnings:
                warnings.append(cam_warning)

            # For missing dimensions, enforce static camera (zoom 1.0 at both ends, centred)
            if sw is None or sh is None:
                c = CameraPoint(cx=0.5, cy=0.5, zoom=1.0)
                cam_params = CameraParams(
                    type="ken_burns", easing=CameraEasing.EASE_IN_OUT, start=c, end=c
                )

        # Transition
        transition = TransitionParams(type="cut", duration_s=0.0)

        # URI
        # As per instructions: "each shot's image_uri is what the worker fetches".
        # The planner contract needs an image_uri.
        # "If a worker-side fallback is needed, add an optional image_urls map
        # keyed by s3:// URI, outside the plan and excluded from plan_hash."

        narrative_stage_val = "unknown"
        room = next((r for r in report.rooms if r.room_key == sel_shot.room_key), None)
        if room:
            narrative_stage_val = room.narrative_stage

        final_shots.append(
            Shot(
                shot_id=f"s{sel_shot.order_idx:02d}",
                order=sel_shot.order_idx,
                asset_id=sel_shot.photo_id,
                image_uri=canonical_uri,
                source=SourceImage(width=sw, height=sh),
                room_key=sel_shot.room_key,
                room_type=ps.room_type or "unknown",
                narrative_stage=narrative_stage_val,
                role=sel_shot.role,
                duration_s=sel_shot.final_duration,
                camera=cam_params,
                transition_in=transition,
            )
        )
        total_dur += sel_shot.final_duration

    # Beat sync (plan 1.5): cuts on the track's hooks and beats within each shot's role bounds.
    # The cameras are planned for the shots' NORMAL lengths (below); each shot then gets its
    # synced length and a time warp: the first part at normal pace, the rear sped up / slowed so
    # the move lands on the cut (user 2026-10-19)
    music, synced_durations = None, None
    if music_track and final_shots:
        synced = sync_to_music(
            [s.duration_s for s in final_shots], music_track, cfg, [s.role for s in final_shots]
        )
        synced_durations = synced.durations
        music = synced.music
        if synced.warning:
            warnings.append(synced.warning)

    chosen_plan_version = "1.2" if cfg.use_motion_presets else "1.1"

    # Cinematic engine (plan 1.3): replaces each shot's camera with a reveal chosen from the room
    # type, role, the photo's motion signals (scoring/motion.py) and the shot's allocated length.
    if cfg.motion_engine == "cinematic" and mode == "cinematic" and final_shots:
        scores = {p.photo_id: p for p in report.photos}
        inputs = []
        for shot in final_shots:
            m = scores[shot.asset_id].motion
            ctx = contexts.get(shot.asset_id)
            inputs.append(ShotInput(
                shot_id=shot.shot_id, src_w=shot.source.width, src_h=shot.source.height,
                room_type=shot.room_type, narrative_stage=shot.narrative_stage, role=shot.role,
                duration_s=shot.duration_s,
                **({"peak_x": ctx.interest_x if ctx else m.peak_x,
                    "peak_y": ctx.interest_y if ctx else m.peak_y, "spread": m.spread,
                    "coverage_class": m.coverage_class,
                    "grid": blend_grid(m.grid, ctx) if ctx else m.grid} if m else {}),
                ctx=ctx if ctx is not None and ctx.semantic else None,
            ))
        # rendered frames per shot: the worker's timeline for hard cuts (boundaries rounded)
        bounds, cum = [0], 0.0
        for shot in final_shots:
            cum += shot.duration_s
            bounds.append(round(cum * cfg.fps))
        frames = [b - a for a, b in zip(bounds, bounds[1:], strict=False)]
        result = plan_cameras(
            inputs, frames, cfg.output_width, cfg.output_height, cfg.cinematic_speed
        )
        for shot, cam in zip(final_shots, result.cameras, strict=True):
            shot.camera = cam
        warnings.extend(w for w in result.warnings if w not in warnings)
        chosen_plan_version = "1.3"

        # Perceptual motion (plan 1.4): curtains, plants, trees and water move inside their masks.
        if cfg.perceptual_motion and detections_by_photo:
            # every shot, exteriors included: OneFormer finds trees and water as well as
            # curtains and plants
            for shot in final_shots:
                dets = detections_by_photo.get(shot.asset_id)
                if dets is None:
                    if shot.asset_id in detections_by_photo:
                        warnings.append(f"perceptual_unavailable:{shot.shot_id}")
                    continue
                regions = plan_motion_regions(
                    listing_id, shot.shot_id, shot.camera, dets, shot.source.width,
                    shot.source.height, cfg.output_width, cfg.output_height, cfg,
                    room_type=shot.room_type,
                )
                if regions:
                    shot.motion_regions = regions
                    chosen_plan_version = "1.4"

    if synced_durations is not None:                 # retime every shot to its synced length
        apply_sync(final_shots, synced, cfg)
        total_dur = sum(synced_durations)
    if music is not None:
        chosen_plan_version = "1.5"

    prov = PlanProvenance(
        scoring_config_version=report.config.version,
        planner_config_version=cfg.version,
        plan_hash=None,
    )

    plan = ShotPlan(
        plan_version=chosen_plan_version,
        listing_id=listing_id,
        mode=mode,
        fallback_reason=fallback_reason,
        preset=preset,
        output=output_params,
        total_duration_s=round(total_dur, 2),
        shots=final_shots,
        warnings=sorted(warnings),
        provenance=prov,
        music=music,
    )

    plan_dict = plan.model_dump(mode="json")
    plan.provenance.plan_hash = _compute_plan_hash(plan_dict)

    return plan
