"""A 3D tour plan (plan 1.6) from a splat scene and its rooms (type + a point in each).

What the photo planner decides, it decides here too - one source of truth for both:
  order        scoring/rooms.py narrative order (entry -> living -> private -> service -> amenities)
  screen time  scoring/selector.py: the hero rooms (best living room, kitchen, primary bedroom) 2x,
               exteriors 1.5x; planner/allocate.py splits each room's budget into shots (so a hero
               room gets two angles), the opening and closing shots hold longer, the tour lands at
               60-120 s (min_total_s / max_total_s) at the chosen pacing
  captions     planner/captions.py; branding, room labels, voiceover, HEVC / MLS copies as for photos
  music        planner/music_sync.py, the same beat sync - applied as an even retime of each shot
               (the move is scaled, never warped: no ramps or zoom punches into a cut)
Lengths are the photo tours' (planner/lengths.py: room 3.5-5 s, hero 5-8 s, detail 1.5-2.5 s at
Normal); a move is planned for its shot's length, so a short shot gets a smaller move.

Per shot, planner3d/moves.py picks the move. Transitions: a cut inside a room or between rooms of the
same part of the home (never a jump cut - the move generator avoids those); a fade into a new part
of the home (the narrative stage changes). No push or zoom transitions.
"""
from __future__ import annotations

import time

import numpy as np

from tour_worker.planner.allocate import allocate_content
from tour_worker.planner.captions import apply_captions
from tour_worker.planner.config import PlannerConfig
from tour_worker.planner.schemas import (BezierSegment, Branding, CameraParams, OutputParams, PlanProvenance,
                                         RoomLabels, Scene, Shot, ShotPlan, TransitionParams, Voiceover)
from tour_worker.planner3d.moves import HFOV, MAX_MOVE_MPS, MAX_TURN_DPS, RoomMoves, fit_pair, speeds_of
from tour_worker.planner3d.occlusion import Occlusion
from tour_worker.planner3d.space import Space
from tour_worker.scoring.config import ScoringConfig
from tour_worker.scoring.rooms import canon
from tour_worker.scoring.schemas import CriterionResult, PhotoScore
from tour_worker.scoring.selector import build_report

FADE_S = 0.5
# rooms one passes through, not stops (user 2026-11-13: "start in living, then kitchen, dining, then
# from dining pan onto the hallway then move into bedroom"): shown by a link shot where the tour's
# walking route first crosses them
CONNECTORS = {"hallway", "corridor", "landing", "staircase", "stairs", "stairway"}
MAX_DETAILS = 2                # close-ups a tour (client: details on the drops; user: option A)
PLANNER3D_VERSION = "3d-2"


def room_keys(rooms: list[dict]) -> list[str]:
    """Each room's key: its own room_key, else <type>_<n> in request order."""
    seen: dict[str, int] = {}
    keys = []
    for r in rooms:
        k = r.get("room_key")
        if not k:
            base = canon(r.get("room_type")) or "room"
            seen[base] = seen.get(base, 0) + 1
            k = f"{base}_{seen[base]}"
        keys.append(k)
    if len(set(keys)) != len(keys):
        raise ValueError("rooms: room_key must be unique")
    return keys


def _signals(coverage: str):
    from tour_worker.scoring.schemas import MotionSignals
    return MotionSignals(peak_x=0.5, peak_y=0.5, side="left", mass_ratio=1.0, spread=0.4, spread_band="mid",
                         room_size="medium", coverage_class=coverage)


def _report(toured: list[tuple[str, str]], details: dict[str, int], second: set[str] = frozenset()):
    """The rooms as the photo planner's scoring report: each room one "view" - a second angle for
    the rooms in `second` - and, when it gets a close-up, a close-up "view" (coverage tight -> a
    detail shot), so the selector ranks, multiplies and orders them as photo rooms. A room's extra
    time makes its shot longer, never another shot (the v3 standard, user 2026-11-13)."""
    scores = []
    for key, rtype in toured:
        views = ([(key, 1.0, "wide")] + [(f"{key}#2", 0.9, "wide")] * (key in second)
                 + [(f"{key}#d", 0.8, "tight")] * min(1, details.get(key, 0)))
        for k, total, cov in views:
            crit = {c: CriterionResult(score=0.5) for c in ("coverage", "framing", "lighting", "depth_of_field")}
            scores.append(PhotoScore(photo_id=k, room_type=rtype, room_key=key, criteria=crit, weights_used={},
                                     missing_criteria=[], total=total, motion=_signals(cov)))
    return build_report(scores, ScoringConfig())


def _richness(space, room: str) -> float:
    """How much a room holds, from its labelled objects: distinct features and details, 0-1."""
    from tour_worker.planner3d.moves import DETAILS, FEATURES
    kinds = {o["class"] for o in space.room_objects(room) if o["class"] in FEATURES or o["class"] in DETAILS}
    return min(1.0, len(kinds) / 5.0)


def _route_tour(space, selected: list, connectors: set[str], cfg: PlannerConfig):
    """Hallways (CONNECTORS) as links, not stops: the stops keep their narrative order; between two
    consecutive rooms the walking route (Space.route) says what way the next lies and what it passes
    through. Where it first crosses a hallway, that hallway's shot becomes a link (a pan from the room
    being left onto the hallway); wherever a route runs through a hallway, the next room's first shot
    walks in from it. Returns (shots, {id(link shot): (from, to, route)}, {id(entering shot): route},
    warnings)."""
    from tour_worker.planner.lengths import shot_length
    stops = [s for s in selected if s.room_key not in connectors]
    if not connectors or not stops:
        return selected, {}, {}, []
    hall_shot = {}
    for s in selected:
        if s.room_key in connectors and s.room_key not in hall_shot:
            hall_shot[s.room_key] = s
    seq, links, enters, used = [], {}, {}, set()
    last_room = None
    for s in stops:
        if last_room is not None and s.room_key != last_room:
            r = space.route(last_room, s.room_key)
            on = [h for h in (r[1] if r else []) if h in connectors]
            if on:
                h = next((h for h in on if h not in used and h in hall_shot), None)
                if h is not None:
                    link = hall_shot[h]
                    link.kind, link.role, link.is_hero = "room", "link", False
                    link.final_duration = shot_length("room", cfg.pacing, 0.8)
                    links[id(link)] = (last_room, s.room_key, r)
                    used.add(h)
                    seq.append(link)
                enters[id(s)] = r
        seq.append(s)
        last_room = s.room_key
    # the tour opens on a room, at hero length (as allocate gives a tour's first shot)
    if seq and seq[0].final_duration < shot_length("hero", cfg.pacing, 0.5):
        seq[0].final_duration = shot_length("hero", cfg.pacing, 0.5)
    seq[0].role, seq[0].kind = "hero", "hero"
    warn = [f"connector_not_on_route:{h}" for h in sorted(connectors) if h not in used and h in hall_shot]
    return seq, links, enters, warn


def plan_tour_3d(model, scene: dict, rooms: list[dict], listing_id: str, cfg: PlannerConfig,
                 music_track: dict | None = None, preset: str = "property_showcase",
                 check_occlusion: bool = True, labels=None):
    """(plan, planning report, the rooms as a scoring report). scene: the plan's scene block (uri, sha256, format, up_axis?, motion_*?);
    rooms: [{room_key?, room_type, point: [x, y, z] in scene axes, tour: bool = True}]."""
    warnings: list[str] = []
    space = Space(model, scene.get("up_axis"))
    warnings += space.warnings
    keys = room_keys(rooms)
    pts = {k: space.to_local(np.asarray(r["point"], np.float64))[:2] for k, r in zip(keys, rooms)}
    warnings += space.set_rooms(pts)
    space.room_types = {k: r.get("room_type") for k, r in zip(keys, rooms)}
    if labels is not None:                     # what the rooms hold (scene_label job)
        space.attach_labels(labels)
    toured = [(k, r.get("room_type")) for k, r in zip(keys, rooms) if r.get("tour", True) and k in space.room_keys]
    if not toured:
        raise ValueError("no toured room is on the scene's floor")

    from types import SimpleNamespace
    from tour_worker.planner3d.moves import DETAILS
    c3 = cfg                                   # the same lengths as photo tours (planner/lengths.py)
    # shots per room (the green-house v3 standard, user 2026-11-13): one room shot each; a hero room
    # (best living room, kitchen, primary bedroom) with a piece to compose on gets a second angle of
    # it; a hero room without one gets a close-up of its best detail instead - at most MAX_DETAILS a
    # tour, never more than two shots a room
    from tour_worker.scoring.rooms import tier_for
    hero_tiers = {"living_room", "kitchen", "primary_bedroom"}
    heroes = [k for k, t in toured if tier_for(t) in hero_tiers]
    second = {k for k in heroes if space.main_pieces(k)}
    best = {k: max((DETAILS[o["class"]] for o in space.room_objects(k, DETAILS)), default=0.0) for k, _ in toured}
    ranked = sorted((k for k in heroes if k not in second and best[k] > 0), key=lambda k: (-best[k], k))
    details = {k: 1 for k in ranked[:MAX_DETAILS]}
    report = _report(toured, details, second)
    contexts = {k: SimpleNamespace(richness=_richness(space, k)) for k, _ in toured} if labels is not None else {}
    selected, alloc_w = allocate_content(report, c3, contexts)
    warnings += alloc_w
    rtype = dict(toured)
    stage = {r.room_key: r.narrative_stage for r in report.rooms}
    selected, links, enters, link_w = _route_tour(space, selected, {k for k, t in toured if canon(t) in CONNECTORS}, c3)
    warnings += link_w

    mover = RoomMoves(space, Occlusion(model, space) if check_occlusion else None)
    moves, plan_rows, prev = [], [], None
    poses: dict[str, list] = {}
    spare: dict[str, float] = {}
    for i, sel in enumerate(selected):
        # no repeats across the room's open-plan space: a hall and the living room it opens into
        # are one space, and a camera in one films the other
        earlier = [p for k in sorted(space.spaces.get(sel.room_key, {sel.room_key})) for p in poses.get(k, [])]
        t_shot = time.perf_counter()
        nxt = next((s.room_key for s in selected[i + 1:] if s.room_key != sel.room_key), None)
        if id(sel) in links:                          # a hallway between two rooms
            src, dst, route = links[id(sel)]
            mv, w = mover.link(src, sel.room_key, dst, route, sel.final_duration, prev)
        else:
            mv, w = mover.best(sel.room_key, sel.final_duration, earlier=earlier, prev=prev,
                               piece_rank=len(poses.get(sel.room_key, [])), opening=i == 0,
                               closing=i == len(selected) - 1, kind=sel.kind, next_room=nxt,
                               enter_route=enters.get(id(sel)),
                               follows_link=i > 0 and id(selected[i - 1]) in links)
        if mv is None and poses.get(sel.room_key) and sel.kind != "detail":
            w = [f"second_angle_dropped:{sel.room_key}"]      # no angle with room to breathe: the first keeps the time
        if mv is not None:
            mv.why["plan_s"] = round(time.perf_counter() - t_shot, 1)
        warnings += [x for x in w if x not in warnings]
        if mv is None:
            spare[sel.room_key] = spare.get(sel.room_key, 0.0) + sel.final_duration
            continue
        poses.setdefault(sel.room_key, []).append((mv.first, mv.last))
        prev = mv.last
        moves.append((sel, mv))
    if not moves:
        raise ValueError("no room allows a camera move")
    # a room is never shown by a close-up alone: without a room shot its close-up goes too
    room_filmed = {s.room_key for s, mv in moves if s.kind != "detail"}
    moves = [(s, mv) for s, mv in moves if s.room_key in room_filmed]
    filmed = {s.room_key for s, _ in moves}
    warnings += [f"room_skipped:{k}" for k in sorted({s.room_key for s in selected} - filmed)]
    for sel, _ in moves:              # a dropped angle gives its time to the room's shot: the same move, slower
        if sel.kind != "detail" and spare.get(sel.room_key):
            add = min(spare.pop(sel.room_key), c3.shot_max_s - sel.final_duration)
            sel.final_duration = round(sel.final_duration + max(0.0, add), 2)

    shots: list[Shot] = []
    for n, (sel, mv) in enumerate(moves, start=1):
        segs, err = fit_pair(mv.eye3, mv.tgt)
        bz = [BezierSegment(eye=np.round(space.to_scene(e), 4).tolist(), target=np.round(space.to_scene(t), 4).tolist())
              for e, t in segs]
        role = sel.role                               # hero | detail (a room shot) | close
        shots.append(Shot(shot_id=f"s{n:02d}", order=n, asset_id="scene", room_key=sel.room_key,
                          room_type=rtype[sel.room_key], narrative_stage=stage[sel.room_key], role=role,
                          duration_s=round(sel.final_duration, 3),
                          camera=CameraParams(type="bezier_path", easing="ease_in_out", move=mv.kind, segments=bz,
                                              hfov_deg=[HFOV, HFOV]),
                          transition_in=TransitionParams(type="cut", duration_s=0.0)))
        plan_rows.append({"shot_id": f"s{n:02d}", "room_key": sel.room_key, "move": mv.kind,
                          "planned_s": round(sel.final_duration, 2), "score": round(mv.score, 3),
                          "piece": mv.piece.to_dict() if mv.piece is not None else None,
                          "fit_error_m": round(err, 3), **mv.why})

    music = None
    if music_track:                                   # the same beat sync, as an even retime
        from tour_worker.planner.music_sync import sync_to_music
        # the opening and closing shots sync as heroes (they hold); room shots only grow; close-ups flex
        roles = ["hero" if s.role == "hero" or k in (0, len(shots) - 1) else s.role for k, s in enumerate(shots)]
        r = sync_to_music([s.duration_s for s in shots], music_track, c3, roles)
        for s, d in zip(shots, r.durations):
            s.duration_s = round(float(d), 4)
        music = r.music
        warnings += [x for x in [r.warning, *(r.extra_warnings or [])] if x]

    # transitions: a fade into a new part of the home, cuts otherwise
    for a, b in zip(shots, shots[1:]):
        # (a link and the shots either side of it are one walk through the home: cuts)
        if b.room_key != a.room_key and b.narrative_stage != a.narrative_stage and "link" not in (a.role, b.role):
            b.transition_in = TransitionParams(type="fade", duration_s=round(min(FADE_S, 0.5 * min(a.duration_s, b.duration_s)), 3))
    apply_captions(shots, cfg.output_room_labels is not None)

    # the speed limits on the shots as they will play (after the sync)
    for s, row in zip(shots, plan_rows):
        turn, move = speeds_of([x.model_dump() for x in s.camera.segments], s.duration_s, "ease_in_out", fps=cfg.fps)
        row.update(played_s=s.duration_s, played_turn_dps=round(turn, 1), played_move_mps=round(move, 2))
        if turn > MAX_TURN_DPS + 0.05 or move > MAX_MOVE_MPS + 0.005:
            warnings.append(f"camera_speed:{s.shot_id}")

    up = scene.get("up_axis")
    if up is None and float(space.up_scene @ np.array([0.0, 0.0, 1.0])) < 0.999:
        up = [round(float(x), 6) for x in space.up_scene]
    plan = ShotPlan(
        plan_version="1.6", listing_id=listing_id, mode="cinematic_3d", preset=preset,
        output=OutputParams(width=cfg.output_width, height=cfg.output_height, fps=cfg.fps, codec="h264",
                            max_bytes=cfg.output_max_bytes, hevc=cfg.output_hevc or None,
                            room_labels=RoomLabels(**cfg.output_room_labels) if cfg.output_room_labels is not None else None,
                            mls_copy=cfg.output_mls_copy or None,
                            branding=Branding(**cfg.output_branding) if cfg.output_branding else None),
        total_duration_s=round(sum(s.duration_s for s in shots), 3), shots=shots,
        warnings=sorted(set(warnings)),
        provenance=PlanProvenance(scoring_config_version=report.config.version,
                                  planner_config_version=f"{cfg.version}+{PLANNER3D_VERSION}"),
        music=music, voiceover=Voiceover(**cfg.voiceover) if cfg.voiceover else None,
        scene=Scene(uri=scene["uri"], sha256=scene["sha256"], format=scene["format"], up_axis=up,
                    json_uri=scene.get("json_uri"), json_sha256=scene.get("json_sha256"),
                    motion_uri=scene.get("motion_uri"), motion_sha256=scene.get("motion_sha256")))
    from tour_worker.planner.planner import _compute_plan_hash
    plan.provenance.plan_hash = _compute_plan_hash(plan.model_dump(mode="json"))
    rep = {
        "scene": {"up": [round(float(x), 4) for x in space.up_scene], "up_source": space.frame.source,
                  "room_height_m": round(space.room_height, 2), "pieces": len(space.pieces),
                  "labels": labels is not None, "objects": len(space.objects)},
        "rooms": [{"room_key": k, "room_type": rtype.get(k) or next((r.get("room_type") for kk, r in zip(keys, rooms) if kk == k), None),
                   "toured": k in rtype, "area_m2": round(space.room_area(k), 1),
                   "open_plan_with": sorted(space.spaces.get(k, {k}) - {k}),
                   "main_pieces": [p.to_dict() for p in space.main_pieces(k)],
                   "objects": sorted({o["class"] for o in space.room_objects(k)})} for k in space.room_keys],
        "sequence": [r for r in report.sequence],
        "shots": plan_rows,
    }
    return plan, rep, report

