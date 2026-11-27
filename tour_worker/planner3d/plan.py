"""A 3D tour plan (plan 1.6) from a splat scene and its rooms (type + a point in each).

What the photo planner decides, it decides here too - one source of truth for both:
  order        scoring/rooms.py narrative order (entry -> living -> private -> service -> amenities)
  screen time  scoring/selector.py: the hero rooms (best living room, kitchen, primary bedroom) 2x,
               exteriors 1.5x; planner/allocate.py splits each room's budget into shots (so a hero
               room gets two angles), the opening and closing shots hold longer, the tour lands at
               60-90 s (min_total_s / max_total_s) at the chosen pacing
  captions     planner/captions.py; branding, room labels, voiceover, HEVC / MLS copies as for photos
  music        planner/music_sync.py, the same beat sync - applied as an even retime of each shot
               (the move is scaled, never warped: no ramps or zoom punches into a cut)
Lengths are the photo tours' (planner/lengths.py: room 3.5-5 s, hero 5-8 s, detail 1.5-2.5 s at
Normal); a move is planned for its shot's length, so a short shot gets a smaller move.

Per shot, planner3d/moves.py picks the move. Transitions: cuts between interior shots (never a jump
cut - the move generator avoids those); a fade only between a drone clip and the interior.
"""
from __future__ import annotations

import math
import time

import numpy as np

from tour_worker.planner.allocate import allocate_content
from tour_worker.planner.captions import apply_captions
from tour_worker.planner.config import PlannerConfig
from tour_worker.planner.schemas import (BezierSegment, Branding, CameraParams, OutputParams, PlanProvenance,
                                         RoomLabels, Scene, Shot, ShotPlan, TransitionParams, Voiceover)
from tour_worker.planner3d.moves import (HFOV, MAX_MOVE_MPS, MAX_TURN_DPS, Move, RoomMoves, ends_backward, fit_pair,
                                         move_motion, speeds_of)
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
LOOK_BACK = {"balcony", "patio", "outdoor_patio", "terrace", "deck", "porch", "garden", "backyard", "rooftop"}
MAX_WIDE = 3                  # wide shots a room at most (coverage: user 2026-11-16)
PLAN_BUDGET_S = 240.0         # s of planning before each shot settles for its best within LATE_CHECKS renders
SYNC_REACH = 0.85            # with music: shots planned at this share of their lengths (the sync lengthens)
DOOR_LEAD = 0.6               # m before a room's door a walk in through it always starts
SYNC_HEADROOM = 0.88          # the cap is met before the sync with this share of it
TAKE_PACE_FLEX = 0.15        # a walk in a take may run this much faster / slower to put its door on the beat
MIN_SHOT_S = 3.0              # s a shot plays at least (client item 9)
JOIN_MAX_S = 10.0             # s the bridge from a walk-in to the room's first scene may take ...
JOIN_MAX_TURN_DEG = 100.0     # ... and deg it may turn - else the walk-in stays the room's opening
JOIN_STRETCH_MAX = 1.5        # a joined walk-in may take this much longer to enter at the cut's pace
REPAIR_ROUNDS = 3             # extra angles the QA repair may drop, one a round
SYNC_DRIFT_S = 0.3            # s a continuous-cut sync adds a shot on average (within its handle)
COVER_ENOUGH = 0.85           # a room whose walls are this covered gets no further shot
MAX_DETAILS = 0                # 3D close-ups off (user 2026-11-16, Home Scan: a painting rendered as a black
#                               blob, wall photos blurred - a scan rarely holds detail at 1 m); was 2
PLANNER3D_VERSION = "3d-2"


def _enters_at(shot, space, fps: float) -> float | None:
    """Seconds into a 3D shot where its camera first stands in the shot's own room (its played
    path, as the renderer plays it: path_s, after the speed band) - None if it starts there."""
    from tour_worker.scene.camera import BezierCameraPath
    cam = shot.camera
    path = BezierCameraPath([s.model_dump() for s in cam.segments], cam.hfov_deg,
                            easing=getattr(cam.easing, "value", cam.easing))
    n = max(2, int(round(shot.duration_s * fps)))
    for f in range(0, n, 2):
        eye = path.pose_at_frame(f, n, fps, path_s=cam.path_s)[0]
        if space.room_at(space.to_local(np.asarray(eye, float))[:2]) == shot.room_key:
            return None if f == 0 else round(f / fps, 2)
    return None


def _door_shares(moves, space) -> dict[int, tuple[float, float]]:
    """{shot index: (path share, least share)} for each flight through a door (a "through" move,
    its room's opening shot): from DOOR_LEAD m before it first stands in its own room to its end -
    the walk under the room's door and the turn onto the room are always played; the speed band
    trims only the way there (speed_band.apply_speed_band_3d; user 2026-11-27: "glides through the
    exit doorway, enters the doorway and pans into the room, all in one scene")."""
    from tour_worker.planner.speed_band import DOOR_LEAD
    out = {}
    for n, (sel, mv) in enumerate(moves):
        if mv.kind != "through" or len(mv.eye) < 2:
            continue
        at = [space.room_at(q) for q in mv.eye]
        k = next((j for j in range(len(at)) if at[j] == sel.room_key), None)
        s_ = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(mv.eye, axis=0), axis=1))]
        if not k or s_[-1] < 1e-6:
            continue
        a = float(max(0.0, s_[k] - DOOR_LEAD) / s_[-1])
        out[n] = (a, 0.0)
    return out


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


UTILITY = {"bathroom", "ensuite", "powder_room", "half_bath", "laundry", "laundry_room", "utility_room",
           "garage", "storage", "closet"}


def _walk_order(space, selected: list, stage: dict[str, str], rtype: dict[str, str], cfg=None) -> list:
    """The narrative order kept stage by stage; within a stage the rooms follow the walk (user
    2026-11-16, the Home Scan jumped living -> far family room -> back to the kitchen beside the
    living room): each stage opens on its narrative first room (its lead: the living room, the
    primary bedroom), then nearest next by walking distance from the room before. Each room's shots
    stay together, in their order.
    A tour never closes in a bathroom or other utility room: it moves before the last other room."""
    blocks: list[tuple[str, list]] = []
    for s_ in selected:
        if blocks and blocks[-1][0] == s_.room_key:
            blocks[-1][1].append(s_)
        else:
            blocks.append((s_.room_key, [s_]))
    dist_cache: dict[tuple[str, str], float] = {}

    def dist(a: str, b: str) -> float:
        """Walking distance from a's centre until one steps into b (user 2026-11-26: the study's door
        is the first one out of the green house's primary bedroom, though its centre is farther)."""
        if (a, b) not in dist_cache:
            r = space.route(a, b)
            d = float("inf")
            if r:
                seg = np.linalg.norm(np.diff(r[0], axis=0), axis=1)
                at = [getattr(space, "room_at", lambda q: None)(q) for q in r[0]]
                k_in = next((k for k in range(1, len(at)) if at[k] == b), None)
                d = float(seg[:k_in].sum()) if k_in is not None else float(seg.sum())
            if not np.isfinite(d):                   # rooms in parts of their own: as the crow flies
                pa, pb = space.room_points.get(a), space.room_points.get(b)
                d = float(np.linalg.norm(pa - pb)) + 100.0 if pa is not None and pb is not None else 1e9
            dist_cache[(a, b)] = d
        return dist_cache[(a, b)]

    out: list[tuple[str, list]] = []
    i = 0
    while i < len(blocks):
        st = stage.get(blocks[i][0])
        j = i
        while j < len(blocks) and stage.get(blocks[j][0]) == st:
            j += 1
        group = blocks[i:j]
        # each stage opens on its narrative first room - the living room, the primary bedroom
        # (user 2026-11-16: the primary bedroom came after the other bedrooms, by walking distance)
        lead = 0
        # the tour opens in THE living room (user 2026-11-17: the Home Scan opened on its family room)
        # - the first stage with a room that is not a hallway (an entry hall is a link, not a stop)
        if not any(canon(rtype.get(k)) not in CONNECTORS for k, _ in out):
            # (two living rooms - the 3D finder names no family room: the larger one)
            # (several living rooms: the one with the most seating - its sofas - then the larger;
            # floor area alone opened the Home Scan in its family room once its den was a room of its own)
            lv = [n for n, (k, _) in enumerate(group) if canon(rtype.get(k)) == "living_room"]
            keys_ = getattr(space, "room_keys", [])

            def _seating(k):
                from tour_worker.planner3d.discover import _area
                return sum(_area(o) for o in getattr(space, "objects", [])
                           if o.get("room") == k and o.get("class") == "sofa")
            lead = max(lv, key=lambda n: (round(_seating(group[n][0]), 1),
                                          float((space.room_fill == keys_.index(group[n][0])).sum())
                                          if group[n][0] in keys_ else 0.0, -n)) if lv else 0
        # a stage with no narrative first room (the other bedrooms, bathrooms, a study) starts with
        # the one nearest where the tour is (user 2026-11-26: from the primary bedroom, through its
        # door, into the study - not back across the hall to the far bedroom)
        if out and len(group) > 1 and canon(rtype.get(group[0][0])) not in ("living_room", "primary_bedroom"):
            lead = min(range(len(group)), key=lambda n: dist(out[-1][0], group[n][0]))
        ordered, prev = [group[lead]], group[lead][0]
        group = group[:lead] + group[lead + 1:]
        while group:
            k = min(range(len(group)), key=lambda n: dist(prev, group[n][0]))
            ordered.append(group.pop(k))
            prev = ordered[-1][0]
        out += ordered
        i = j
    if len(out) > 1 and canon(rtype.get(out[-1][0])) in UTILITY:
        last_other = max((n for n, (k, _) in enumerate(out) if canon(rtype.get(k)) not in UTILITY), default=None)
        if last_other is not None:
            out.insert(last_other, out.pop())
    seq = [s_ for _, shots in out for s_ in shots]
    if seq and selected and seq[-1] is not selected[-1] and cfg is not None:
        # the closing length (allocate gave it to the narrative's last shot) goes with the closing
        from tour_worker.planner.lengths import shot_length
        old_last = selected[-1]
        seq[-1].final_duration = max(seq[-1].final_duration, old_last.final_duration)
        old_last.final_duration = shot_length("room", cfg.pacing, 0.5)
    return seq


def _route_tour(space, selected: list, connectors: set[str], cfg: PlannerConfig):
    """Hallways (CONNECTORS) as links, not stops: the stops keep their narrative order; between two
    consecutive rooms the walking route (Space.route) says what way the next lies and what it passes
    through. Where it first crosses a hallway, that hallway's shot becomes a link (a pan from the room
    being left onto the hallway); wherever a route runs through a hallway, the next room's first shot
    walks in from it. Returns (shots, {id(link shot): (from, to, route)}, {id(entering shot): route},
    warnings)."""
    from tour_worker.planner.lengths import shot_length
    stops = [s for s in selected if s.room_key not in connectors]
    if not stops:
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
            # a room reached through a doorway (not across an open plan) opens walking in from it,
            # when that walk is a good shot (user 2026-11-16: a hallway view into the primary bedroom
            # came as its last shot - "it could have been the first scene")
            if r and s.room_key not in space.spaces.get(last_room, {last_room}):
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
                 check_occlusion: bool = True, labels=None, drone_clips: list[dict] | None = None):
    """(plan, planning report, the rooms as a scoring report). scene: the plan's scene block (uri, sha256, format, up_axis?, motion_*?);
    rooms: [{room_key?, room_type, point: [x, y, z] in scene axes, tour: bool = True}].
    drone_clips: [{video_s3_url, role, in_point_s?, out_point_s?, zoom_end?, transition_to_interior?}]."""
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
    # shots per room (user 2026-11-16, superseding the v3 "one shot, a hero's second angle"): up to
    # MAX_WIDE wide shots a room, as many as it takes to show its walls (added below, by coverage);
    # a close-up of a hero room's best detail - at most MAX_DETAILS a tour
    from tour_worker.scoring.rooms import tier_for
    hero_tiers = {"living_room", "kitchen", "primary_bedroom"}
    heroes = [k for k, t in toured if tier_for(t) in hero_tiers]
    best = {k: max((DETAILS[o["class"]] for o in space.room_objects(k, DETAILS)), default=0.0) for k, _ in toured}
    ranked = sorted((k for k in heroes if best[k] > 0), key=lambda k: (-best[k], k))
    details = {k: 1 for k in ranked[:MAX_DETAILS]}
    report = _report(toured, details, set())
    contexts = {k: SimpleNamespace(richness=_richness(space, k)) for k, _ in toured} if labels is not None else {}
    selected, alloc_w = allocate_content(report, c3, contexts)
    warnings += alloc_w
    rtype = dict(toured)
    stage = {r.room_key: r.narrative_stage for r in report.rooms}
    selected = _walk_order(space, selected, stage, rtype, c3)
    selected, links, enters, link_w = _route_tour(space, selected, {k for k, t in toured if canon(t) in CONNECTORS}, c3)
    warnings += link_w

    # with music, shots are planned short of their lengths (SYNC_REACH): the sync only lengthens, so
    # this gives it room to land each cut on its moment instead of the next beat after a planned end
    # (2026-11-17, Home Scan v8: cuts a beat after a strong hit, one between beats; offline at 0.85:
    # none of either) - the moves are planned for these lengths, so the sync only slows them
    # continuous cuts (client review 2026-11-20, items 2-3): instead, every move is planned a
    # handle longer than its shot (camera.path_s) at its own pace, and the sync only picks the cut
    # inside it (planner/music_sync.length_bounds) - never retimes it
    continuous = bool(getattr(c3, "continuous_cuts", False))
    handle = c3.cut_handle_s if (continuous and music_track) else 0.0
    reach = 1.0 if continuous else (SYNC_REACH if music_track else 1.0)
    if reach != 1.0 or handle:
        for s_ in selected:
            s_.final_duration = round(s_.final_duration * reach + handle, 3)
    mover = RoomMoves(space, Occlusion(model, space) if check_occlusion else None)
    mover.deadline = time.perf_counter() + PLAN_BUDGET_S      # past it: each shot's best within a few checks
    mover.continuous = continuous
    one_take = bool(continuous and getattr(c3, "one_take_3d", False))
    if continuous and getattr(c3, "speed_band", False):    # every candidate planned at the 3D pace
        from tour_worker.planner.speed_band import target_speed_3d
        mover.pace_target = target_speed_3d(c3.pacing)
    mover.tour_shots = len(selected)               # the moving-in cap: a third of the tour's shots
    last_room = selected[-1].room_key if selected else None
    moves, plan_rows, prev = [], [], None
    poses: dict[str, list] = {}
    spare: dict[str, float] = {}
    covered: dict[str, np.ndarray] = {}           # per room: its wall cells the shots so far show
    wide: dict[str, int] = {}
    first_s: dict = {}
    extras: list = []                             # (walls_new, (sel, mv)) of the shots added for coverage

    def _unrecord(before=None):
        """Undo the planner's record of the shot just chosen (it is not taken): the cut into the next
        one is measured from `before` ((sel, mv), default the last move taken) again."""
        hist = mover.__dict__.get("_history", [])
        if hist:
            hist.pop()
        b = before if before is not None else (moves[-1] if moves else None)
        mover._last_exit = move_motion(b[1], end=True) if b is not None else None
        mover._last_exit_pct = (mover.pace_ends(b[1], float(b[0].final_duration))[1]
                                if b is not None and getattr(mover, "pace_target", None) else None)

    def _own_scene_needed(ms, mv) -> bool:
        """Dropping mv would leave its room shown only by walking into it (user 2026-11-27: "the
        bedroom needs another shot, it doesn't even get shown at all") - it stays."""
        if mv is None:
            return False
        room = next((sel.room_key for sel, m in ms if m is mv), None)
        rest = [m for sel, m in ms if sel.room_key == room and m is not mv]
        return bool(rest) and all(m.kind in ("through", "bridge", "exit", "walk") for m in rest)

    def _cover_room(room: str, i: int, prev):
        """Further wide shots of `room` while its walls are not covered (COVER_ENOUGH), at most
        MAX_WIDE in all, each showing walls the earlier ones did not (RoomMoves.best covered=)."""
        import dataclasses
        from tour_worker.planner.lengths import length_range
        walls = space.wall_cells(room)
        lead = first_s[room]
        # a room that flies on to the next (an exit) keeps a shot of its MAX_WIDE for it
        nxt = selected[i + 1] if i + 1 < len(selected) else None
        flies = nxt is not None and id(nxt) not in links and nxt.kind != "detail"
        limit = MAX_WIDE - (1 if flies else 0)
        while wide.get(room, 0) < limit and len(walls) and len(covered.get(room, ())) / len(walls) < COVER_ENOUGH:
            kind = lead.kind if lead.kind in ("room", "hero") else "room"
            # a further angle is a room shot's length (planner/lengths), never a hero's: the time
            # goes to showing more rooms whole (user 2026-11-26)
            lo_, hi_ = (x * reach + handle for x in length_range("room", c3.pacing))
            dur = round(min(hi_, max(lo_, 0.85 * lead.final_duration)), 2)
            earlier_ = [p for k in sorted(space.spaces.get(room, {room})) for p in poses.get(k, [])]
            mover.cover_stats = {}
            mv, cw = mover.best(room, dur, earlier=earlier_, prev=prev, piece_rank=wide.get(room, 0),
                                kind=kind, covered=covered.get(room, np.zeros(0, np.int64)),
                                final=room == last_room, then_forward=_leaves_flying(i))
            if mv is None and wide.get(room, 0) == 1 and id(lead) in flown_in_lead:
                # walked into through its door, the room's opening is that walk; its next angle is
                # chosen as its opening shot was before (no new-wall share asked of it) - the green
                # house's primary bedroom was otherwise left with the walk-in alone (2026-11-27)
                mv, cw = mover.best(room, dur, earlier=earlier_, prev=prev, piece_rank=1, kind=kind,
                                    final=room == last_room, then_forward=_leaves_flying(i))
                if mv is not None:
                    mv.why["walls_new"] = round(mover.new_walls(room, mv, covered.get(room, np.zeros(0, np.int64))), 2)
            if mv is not None and continuous and (mv.why.get("cut") or {}).get("cost", 0.0) > 0:
                # an extra angle that breaks the cut into it is not taken (QA: every cut within the
                # limits, user 2026-11-27) - the room keeps the angles it has
                _unrecord()
                warnings.append(f"coverage_stopped:{room}:breaks_cut")
                break
            if mv is None:
                st = getattr(mover, "cover_stats", {}) or {}
                seen = len(covered.get(room, ())) / max(len(walls), 1)
                warnings.append(f"coverage_stopped:{room}:walls_seen_{seen:.2f}:"
                                + "/".join(str(st.get(k, "-")) for k in ("candidates", "room_to_breathe_and_good",
                                                                          "composed_own_room_new_walls")))
                break
            sel = dataclasses.replace(lead, role="detail", kind="room", is_hero=False, is_lead=False,
                                      final_duration=dur, raw_duration=dur)
            mv.why["coverage_shot"] = True
            poses.setdefault(room, []).append((mv.first, mv.last))
            moves.append((sel, mv))
            extras.append((mv.why.get("walls_new", 0.0), (sel, mv)))
            covered[room] = np.union1d(covered[room], mover.walls_seen(room, mv))
            wide[room] += 1
            prev = mv.last
        # the room's last extra angle must lead into the next room's move; else it goes (QA)
        from tour_worker.planner3d.moves import breaks_into
        nxt_in = _leaves_flying(i) if continuous else False
        if (nxt_in and moves and moves[-1][1].why.get("coverage_shot") and moves[-1][0].room_key == room
                and breaks_into(moves[-1][1], nxt_in)):
            gone = moves.pop()
            extras[:] = [e for e in extras if e[1][1] is not gone[1]]
            if poses.get(room):
                poses[room].pop()
            wide[room] -= 1
            _unrecord(moves[-1] if moves else None)
            warnings.append(f"coverage_dropped_breaks_cut:{room}")
            prev = moves[-1][1].last if moves else None
        return prev
    exits: list = []                              # flight chains (lists of (sel, mv)): dropped whole for length
    after_exit: set = set()                       # rooms whose first shot picks a matched cut at their door up
    flown_in: set = set()                         # rooms a through flight already entered
    flown_in_lead: set = set()                    # (ids of) the through flights that are a room's opening shot
    room_changes: list = []                       # per room change: its doors and how it was crossed (report)

    def _rechoose(before, pair):
        """pair (sel, mv) planned again after `before` - the next room's first shot when the flight
        into it is dropped; the old move when no other passes."""
        from tour_worker.planner3d.moves import move_motion
        sel, old = pair
        mover._last_exit = move_motion(before[1], end=True)
        earlier_ = [p for k in sorted(space.spaces.get(sel.room_key, {sel.room_key})) for p in poses.get(k, [])
                    if p[0] is not old.first[0]]
        mv, _ = mover.best(sel.room_key, sel.final_duration, earlier=earlier_, prev=before[1].last, piece_rank=0,
                           kind=sel.kind, enter_route=enters.get(id(sel)), final=sel.room_key == last_room)
        if mv is None:
            return pair
        mv.why["rechosen_after_dropped_flight"] = True
        return (sel, mv)

    can_fly: dict = {}

    def _last_into_next(sel, i: int, prev, earlier):
        """A room filmed by its opening shot alone (no coverage shot followed it): that shot is the
        room's last, so it is chosen again - when it would break the cut into the next room's move
        (breaks_into) - by the rule a room's last shot keeps (user 2026-11-27: the green house
        bathroom backed out into the bedroom's walk-in). The first choice stays if nothing passes."""
        from tour_worker.planner3d.moves import breaks_into
        nxt_in = _leaves_flying(i)
        if not nxt_in or not moves or moves[-1][0] is not sel or not breaks_into(moves[-1][1], nxt_in):
            return prev
        old = moves[-1][1]
        before = moves[-2][1] if len(moves) > 1 else None
        hist = mover.__dict__.get("_history", [])
        if hist:
            hist.pop()                                 # (the first choice, chosen again)
        mover._last_exit = move_motion(before, end=True) if before is not None else None
        mover._last_exit_pct = (mover.pace_ends(before, float(before.why.get("dur_s") or moves[-2][0].final_duration))[1]
                                if before is not None and getattr(mover, "pace_target", None) else None)
        earlier_ = [p for p in earlier if p[0] is not old.first[0]]
        mv, _ = mover.best(sel.room_key, sel.final_duration, earlier=earlier_,
                           prev=before.last if before is not None else None, piece_rank=0, kind=sel.kind,
                           enter_route=enters.get(id(sel)), final=sel.room_key == last_room,
                           then_forward=nxt_in, last_shot=True)
        if mv is None or breaks_into(mv, nxt_in):
            mover.record(old, sel.final_duration)       # (as it was)
            return prev
        mv.why["rechosen_into_next"] = True
        moves[-1] = (sel, mv)
        lst = poses.get(sel.room_key, [])
        if lst:
            lst[-1] = (mv.first, mv.last)
        covered[sel.room_key] = mover.walls_seen(sel.room_key, mv)
        return mv.last

    def _hall_walk(hall: str, route, prev):
        """A walk along the hallway's stretch of the walking route: from just inside it (past the
        door the tour came through) to EXIT_STOP short of where it leaves for the next room."""
        from tour_worker.planner3d.moves import AFTER_EXIT, EXIT_STOP
        poly = route[0]
        s_ = mover._along(poly)
        at = [space.room_at(q) for q in poly]
        ins = [k for k in range(len(poly)) if at[k] == hall]
        if not ins:
            return None, [f"hall_walk_no_route:{hall}"]
        s_in, s_out = float(s_[ins[0]]), float(s_[ins[-1]])
        if s_out - s_in < 1.2:
            return None, [f"hall_walk_too_short:{hall}"]
        last_mv = moves[-1][1] if moves else None
        pace = (float(np.linalg.norm(np.diff(last_mv.eye, axis=0), axis=1).sum()) / max(moves[-1][0].final_duration, 1e-6)
                if last_mv is not None else 0.3)
        earlier_ = [p for k in sorted(space.spaces.get(hall, {hall})) for p in poses.get(k, [])]
        return mover.flight("walk", hall, route, (s_in + AFTER_EXIT[0] * 0.5, s_in + AFTER_EXIT[1]),
                            (max(s_in + 1.0, s_out - EXIT_STOP - 0.6), max(s_in + 1.2, s_out - EXIT_STOP + 0.4)),
                            pace, handle, prev, earlier_, end_room=hall, look_max_s=s_out + 0.5)

    def _leaves_flying(i: int):
        """What follows selected[i], its room's last planned shot, when that is a move on: the flight
        on's start motion (a dry run - its last shot is chosen to cut into it, user 2026-11-27), True
        for the next room's walk in from its door (forward, heading unknown), else False."""
        nxt = selected[i + 1] if i + 1 < len(selected) else None
        if not (nxt is not None and nxt.room_key != selected[i].room_key and id(nxt) not in links
                and nxt.kind != "detail" and id(selected[i]) not in links):
            return False
        key = (selected[i].room_key, nxt.room_key)
        if key not in can_fly:                     # a dry run of the flight (geometry only, no renders)
            mv_ = _fly_on(selected[i].room_key, i, None, dry=True)
            can_fly[key] = move_motion(mv_, end=False) if isinstance(mv_, Move) else bool(mv_)
        v = can_fly[key]
        if v is not None and v is not False:
            return v
        return id(nxt) in enters

    def _last_toward(flight) -> None:
        """The room's last shot against the flight on as planned (its dry run started on another
        line - green house: the primary bedroom's pan swept left into a walk drifting right): when
        it breaks that cut and reads the same both ways (REVERSIBLE), it is played in reverse - if
        that passes the render checks and the cut into it."""
        import dataclasses
        from tour_worker.planner3d.moves import MAX_OCC, REVERSIBLE, VOID_MAX, breaks_into
        sel_l, last = moves[-1]
        if last.kind not in REVERSIBLE or not breaks_into(last, move_motion(flight, end=False)):
            return
        rev = dataclasses.replace(last, eye=last.eye[::-1].copy(), tgt=last.tgt[::-1].copy(),
                                  z=None if last.z is None else last.z[::-1].copy(), occ=None,
                                  why={**last.why, "reversed_into_next": True})
        if breaks_into(rev, move_motion(flight, end=False)):
            return
        if len(moves) > 1 and moves[-2][0].room_key == sel_l.room_key and breaks_into(moves[-2][1], move_motion(rev, end=False)):
            return
        if mover.occ is not None and (mover._cover(rev) > MAX_OCC or mover._void(rev) > VOID_MAX):
            return
        moves[-1] = (sel_l, rev)
        lst = poses.get(sel_l.room_key, [])
        if lst:
            lst[-1] = (rev.first, rev.last)

    def _fly_on(room: str, i: int, prev, dry: bool = False):
        """The room change after `room` (client review item 6; user 2026-11-26, option c): along the
        walking route, cut only at narrow doors (Space doorways under THROUGH_MIN_W, not open plan),
        each cut matched - the leg before ends heading for the door (an exit, or a walk along a hall),
        the leg after starts just past it on the same heading. With no narrow door left before the
        next room, the last leg flies through into it and turns onto what it holds; else the next
        room's first shot is the matched walk-in (after_exit). A leg that cannot be flown drops the
        chain: the next room is entered as before."""
        import dataclasses
        from tour_worker.planner3d.moves import (AFTER_EXIT, EXIT_LEN, EXIT_STOP, THROUGH_AFTER,
                                                 THROUGH_AFTER_LEAD, THROUGH_BEFORE, WALK_LEN)
        DOOR_GAP = EXIT_STOP + AFTER_EXIT[1] + 0.6
        nxt = selected[i + 1] if i + 1 < len(selected) else None
        if nxt is None or id(nxt) in links or nxt.kind == "detail":
            return prev
        if not dry and moves and moves[-1][1].kind == "backout" and ends_backward(moves[-1][1]):
            return prev          # (backed out of: a walk on would reverse at the cut - the next room by a cut)
        dst = nxt.room_key
        lead_s = 0.0          # (the room's first scene follows the walk-in with its own time; was the opening's)
        r = space.route(room, dst)
        if not r:
            return prev
        cr = mover.crossings(r)
        into = [c for c in cr if c["to"] == dst]
        if not into:
            return prev
        entry = into[-1]
        # narrow doors closer than a leg needs between them (EXIT_STOP + AFTER_EXIT + a little) are one
        # cut: the leg before stops short of the first, the leg after starts past the last
        cuts = []
        for c in (c for c in cr if c["narrow"] and c["s"] <= entry["s"]):
            if cuts and c["s"] - cuts[-1]["last_s"] < DOOR_GAP:
                cuts[-1]["last_s"], cuts[-1]["to"], cuts[-1]["is_entry"] = c["s"], c["to"], c is entry
            else:
                cuts.append({"s": c["s"], "last_s": c["s"], "to": c["to"], "is_entry": c is entry})
        if dry:
            log = {"legs": []}
        else:
            log = {"from": room, "to": dst, "route_m": round(float(mover._along(r[0])[-1]), 2),
                   "doors": [{**{k: c[k] for k in ("from", "to", "width_m", "open", "narrow")}, "at_m": round(c["s"], 2)} for c in cr],
                   "legs": []}
            room_changes.append(log)
        # the previous shot's pace (item 4: the speed carries through the cut)
        if moves and not dry:
            last_sel, last_mv = moves[-1]
            pace = float(np.linalg.norm(np.diff(last_mv.eye, axis=0), axis=1).sum()) / max(last_sel.final_duration, 1e-6)
        else:
            pace = 0.3
        pieces = space.main_pieces(dst)
        piece = pieces[0] if pieces else None
        fp = (piece.frame or piece) if piece is not None else None
        aim = np.array([*fp.centre, fp.top]) if fp is not None else mover._goal(dst)
        chain, p_ = [], prev
        if cuts:
            # one continuous move first, narrow doors included (user 2026-11-26: "we were getting out
            # of the bedroom correctly then we just cut to the bathroom - it needs to be one
            # continuous moment"): from inside the room, through every door on the way, into the
            # next room; the cuts at narrow doors below only when it cannot be flown
            out_s = next((c["s"] for c in cr if c["from"] == room), cuts[0]["s"])
            start0 = (max(0.0, out_s - THROUGH_BEFORE[1]), max(0.3, out_s - THROUGH_BEFORE[0]))
            end0 = (entry["s"] + THROUGH_AFTER_LEAD[0], entry["s"] + THROUGH_AFTER_LEAD[1])
            earlier0 = [p for k in sorted(space.spaces.get(dst, {dst})) for p in poses.get(k, [])]
            mv, w = mover.flight("through", dst, r, start0, end0, pace, handle, None if dry else prev, earlier0,
                                 aim=aim, piece=piece, end_room=dst, geometry_only=dry, start_room=room,
                                 min_s=lead_s)
            if mv is None:                       # a bend past the door: look at one point in the room
                mv, w = mover.flight("through", dst, r, start0, end0, pace, handle, None if dry else prev, earlier0,
                                     aim=aim, piece=piece, end_room=dst, geometry_only=dry, start_room=room,
                                     look_fixed_s=entry["s"] + THROUGH_AFTER_LEAD[1] + 1.0, min_s=lead_s)
            if mv is not None:
                if dry:
                    return mv
                dur = round(mv.why["dur_s"] + handle, 2)
                sel = dataclasses.replace(nxt, final_duration=dur, raw_duration=dur)     # the room's opening shot
                chain = [(sel, mv)]
                log["legs"].append({"kind": "through", "room": dst, "travel_m": mv.why["travel_m"],
                                    "dur_s": mv.why["dur_s"], "doors_passed": len(cuts)})
                p_ = mv.last
            elif not dry:
                warnings.extend(x for x in w if x not in warnings)
        for j in (range(len(cuts) + 1) if not chain else ()):
            door = cuts[j] if j < len(cuts) else None
            if door is None and cuts and cuts[-1]["is_entry"]:
                break                                  # the next room's own door is the last cut
            if door is None:
                kind, end, end_room, aim_, piece_ = "through", (entry["s"] + THROUGH_AFTER_LEAD[0], entry["s"] + THROUGH_AFTER_LEAD[1]), dst, aim, piece
            else:
                kind = "exit" if j == 0 else "walk"
                end, end_room, aim_, piece_ = (door["s"] - EXIT_STOP - 0.6, door["s"] - EXIT_STOP), None, None, None
            if j == 0:
                start = ((entry["s"] - THROUGH_BEFORE[1], entry["s"] - THROUGH_BEFORE[0]) if door is None
                         else (end[0] - EXIT_LEN[1], end[1] - EXIT_LEN[0]))
                start = (max(0.0, start[0]), max(0.3, start[1]))   # a room point near its door: from it
            else:
                d0 = cuts[j - 1]["last_s"]
                start = (d0 + AFTER_EXIT[0], d0 + AFTER_EXIT[1])
                if end[1] - start[0] > WALK_LEN[1] + (THROUGH_AFTER[1] if door is None else 0.0):
                    if not dry:
                        warnings.append(f"flight_too_long:{room}:{dst}")
                    chain = None
                    break
            k_mid = int(np.searchsorted(mover._along(r[0]), 0.5 * (start[0] + end[1])))
            mid = space.room_at(r[0][min(len(r[0]) - 1, k_mid)])
            rk = dst if kind == "through" else room if kind == "exit" else (mid if mid in rtype else room)
            earlier_ = [p for k in sorted(space.spaces.get(rk, {rk})) for p in poses.get(k, [])]
            mv, w = mover.flight(kind, rk, r, start, end, pace, handle, None if dry else p_, earlier_, aim=aim_,
                                 piece=piece_, end_room=end_room, matched=j > 0,
                                 look_max_s=None if door is None else door["s"] + 0.5, geometry_only=dry,
                                 start_room=room if j == 0 else None, min_s=lead_s if kind == "through" else 0.0)
            if mv is None:
                if not dry:
                    warnings.extend(x for x in w if x not in warnings)
                chain = None
                break
            if dry:
                chain.append((None, mv))
                p_ = mv.last
                continue
            dur = round(mv.why["dur_s"] + handle, 2)
            sel = (dataclasses.replace(nxt, final_duration=dur, raw_duration=dur) if kind == "through"   # the room's opening shot
                   else dataclasses.replace(first_s[room], room_key=rk, role="detail", kind="room", is_hero=False,
                                            is_lead=False, final_duration=dur, raw_duration=dur))
            chain.append((sel, mv))
            log["legs"].append({"kind": kind, "room": rk, "travel_m": mv.why["travel_m"], "dur_s": mv.why["dur_s"]})
            p_ = mv.last
        if dry:
            return chain[0][1] if chain else None
        if chain and moves and moves[-1][0].room_key == room:
            _last_toward(chain[0][1])
        if not chain:
            log["legs"] = []
            log["crossed"] = "as_before"
            return prev
        for sel, mv in chain:
            poses.setdefault(sel.room_key, []).append((mv.first, mv.last))
            moves.append((sel, mv))
        exits.append(chain)
        if chain[-1][1].kind == "through":
            flown_in.add(id(nxt))
            log["crossed"] = "through"
        else:
            after_exit.add(id(nxt))
            log["crossed"] = "matched_cut"
        return p_

    take_built = False
    if one_take:
        # one take through the home (user 2026-11-27): each room a loop round it, the rooms a walking
        # route joins walked into each other through their doors (planner3d/take.py) - in place of
        # the shots below; the shots below when no room gets a loop
        import dataclasses
        from tour_worker.planner3d.take import ROOM_WEIGHT_HERO, plan_takes
        order: list[str] = []
        lead_of: dict = {}
        for s_ in selected:
            if id(s_) in links or s_.kind == "detail":
                continue
            lead_of.setdefault(s_.room_key, s_)
            if not order or order[-1] != s_.room_key:
                order.append(s_.room_key)
        weights = {k: math.sqrt(max(space.room_area(k), 1.0)) * (ROOM_WEIGHT_HERO if tier_for(rtype.get(k)) in hero_tiers else 1.0)
                   for k in order}
        subjects = {}
        for k in order:
            pc = next(iter(space.main_pieces(k)), None)
            if pc is not None:                     # (what the loop circles: its centre, and the piece itself)
                subjects[k] = (np.asarray((pc.frame or pc).centre, float), pc)
        tk, tw = plan_takes(mover, order, weights, getattr(c3, "max_total_s", None) or 90.0, subjects)
        if any(w_.startswith("take_room_no_loop") for w_ in tw):
            # a room with no loop the checks pass: the tour of shots (cut within the limits) instead -
            # never a room left out of the take
            warnings += tw + ["take_fallback:shots"]
            tk, one_take = [], False
        else:
            warnings += tw
        if tk:
            for n_, (rk, mv_, seam_) in enumerate(tk):
                dur_ = round(float(mv_.why["dur_s"]), 2)
                base = lead_of.get(rk) or selected[0]
                sel_ = dataclasses.replace(base, room_key=rk, final_duration=dur_, raw_duration=dur_,
                                           role="hero" if n_ == 0 else ("detail" if mv_.kind == "bridge" else base.role),
                                           kind="hero" if n_ == 0 else ("room" if mv_.kind == "bridge" else base.kind))
                mv_.why["seamless_out"] = bool(seam_)
                if n_ > 0 and tk[n_ - 1][2]:
                    mv_.why["seamless_in"] = True
                moves.append((sel_, mv_))
                poses.setdefault(rk, []).append((mv_.first, mv_.last))
                first_s.setdefault(rk, sel_)
            take_built = True
    for i, sel in enumerate(selected if not take_built else []):
        # no repeats across the room's open-plan space: a hall and the living room it opens into
        # are one space, and a camera in one films the other
        earlier = [p for k in sorted(space.spaces.get(sel.room_key, {sel.room_key})) for p in poses.get(k, [])]
        t_shot = time.perf_counter()
        nxt = next((s.room_key for s in selected[i + 1:] if s.room_key != sel.room_key), None)
        if id(sel) in links:                          # a hallway between two rooms
            src, dst, route = links[id(sel)]
            mv, w = mover.link(src, sel.room_key, dst, route, sel.final_duration, prev)
            if mv is None:
                # no pan onto the hallway passes: walk down it instead, toward the next room (user
                # 2026-11-26: the Home Scan's hallway was dropped when its link was blocked)
                mv, w2 = _hall_walk(sel.room_key, route, prev)
                w = w + w2 if mv is None else [f"link_walked:{sel.room_key}"]
        elif id(sel) in flown_in:
            # flown into through its door: that flight is the room's opening shot - one move under the
            # doorway and on into the room, never cut to another angle just past the door (user
            # 2026-11-27); it is already in moves (_fly_on), here the room is counted as filmed
            mv = next(m for s_, m in reversed(moves) if s_.room_key == sel.room_key and m.kind == "through")
            sel_ = next(s_ for s_, m in reversed(moves) if m is mv)
            covered[sel.room_key] = np.union1d(covered.get(sel.room_key, np.zeros(0, np.int64)), mover.walls_seen(sel.room_key, mv))
            wide[sel.room_key] = wide.get(sel.room_key, 0) + 1
            first_s.setdefault(sel.room_key, sel_)
            flown_in_lead.add(id(first_s[sel.room_key]))
            prev = mv.last
            # ... and the room's first scene carried straight on from it (user 2026-11-27: "only walk
            # outs and walk-ins and first scene of the room is continuous; rest of the scenes follow
            # the normal methodology"): the first scene as the room's first shot is chosen, joined to
            # the walk-in by a seamless bridge - else the walk-in stays the room's opening
            if continuous:
                import dataclasses
                lead, lw = mover.best(sel.room_key, sel.final_duration, earlier=earlier, prev=None, piece_rank=0,
                                      kind=sel.kind, next_room=nxt, final=sel.room_key == last_room,
                                      then_forward=_leaves_flying(i), last_shot=False,
                                      join_from=(mv.eye[-1], mv.tgt[-1], mover.walk_dir(mv)))   # (prev None: no jump-cut test)
                br = mover.bridge(mv, lead) if lead is not None else None
                # carried on only forward: never back along its way, rotating and on again (user 2026-11-27)
                if (br is not None and br.why["dur_s"] <= JOIN_MAX_S and br.why["turn_deg"] <= JOIN_MAX_TURN_DEG
                        and mover.ahead_of((mv.eye[-1], mv.tgt[-1], mover.walk_dir(mv)), lead)
                        and mover.joins_forward(mv, br)):
                    mv.why["seamless_out"] = True
                    br.why.update(seamless_in=True, seamless_out=True)
                    lead.why["seamless_in"] = True
                    bdur = round(float(br.why["dur_s"]), 2)
                    moves.append((dataclasses.replace(sel_, role="detail", kind="room", is_hero=False, is_lead=False,
                                                      final_duration=bdur, raw_duration=bdur), br))
                    moves.append((sel, lead))
                    poses.setdefault(sel.room_key, []).append((lead.first, lead.last))
                    covered[sel.room_key] = np.union1d(covered[sel.room_key], mover.walls_seen(sel.room_key, lead))
                    wide[sel.room_key] += 1
                    prev = lead.last
                else:
                    if lead is not None:
                        hist_ = mover.__dict__.get("_history", [])
                        if hist_:
                            hist_.pop()                     # (not taken: the cut is measured from the walk-in)
                        mover._last_exit = move_motion(mv, end=True)
                        mover._last_exit_pct = mover.pace_ends(mv, float(sel_.final_duration))[1] if getattr(mover, "pace_target", None) else None
                    warnings.append(f"walk_in_not_joined:{sel.room_key}")
            if i == len(selected) - 1 or selected[i + 1].room_key != sel.room_key:
                if canon(rtype.get(sel.room_key)) not in LOOK_BACK:
                    prev = _cover_room(sel.room_key, i, prev)
                    prev = _fly_on(sel.room_key, i, prev)
            continue
        else:
            # the closing look-back pan suits an outdoor space looking back at the home (the green
            # house's balcony); another last room is filmed as any room, its last coverage shot closing
            closing = i == len(selected) - 1 and canon(rtype.get(sel.room_key)) in LOOK_BACK
            mv, w = mover.best(sel.room_key, sel.final_duration, earlier=earlier, prev=prev,
                               piece_rank=len(poses.get(sel.room_key, [])), opening=i == 0,
                               closing=closing, kind=sel.kind, next_room=nxt,
                               enter_route=None if id(sel) in flown_in else enters.get(id(sel)),
                               after_exit=id(sel) in after_exit,
                               final=sel.room_key == last_room and i > 0,
                               follows_link=i > 0 and id(selected[i - 1]) in links,
                               then_forward=_leaves_flying(i), last_shot=False)   # (its coverage may follow)
        if mv is None and poses.get(sel.room_key) and sel.kind != "detail":
            w = [f"second_angle_dropped:{sel.room_key}"]      # no angle with room to breathe: the first keeps the time
        if mv is not None:
            mv.why["plan_s"] = round(time.perf_counter() - t_shot, 1)
        warnings += [x for x in w if x not in warnings]
        if mv is None:
            if not poses.get(sel.room_key) and sel.kind != "detail":
                w = [f"room_no_good_view:{sel.room_key}"]      # no shot with room to breathe of its own room
                warnings += [x for x in w if x not in warnings]
            spare[sel.room_key] = spare.get(sel.room_key, 0.0) + sel.final_duration
            continue
        poses.setdefault(sel.room_key, []).append((mv.first, mv.last))
        prev = mv.last
        moves.append((sel, mv))
        if sel.kind != "detail" and id(sel) not in links:
            room = sel.room_key
            covered[room] = np.union1d(covered.get(room, np.zeros(0, np.int64)), mover.walls_seen(room, mv))
            wide[room] = wide.get(room, 0) + 1
            first_s.setdefault(room, sel)
        last_of_room = i == len(selected) - 1 or selected[i + 1].room_key != sel.room_key
        if last_of_room and sel.room_key in first_s and id(sel) not in links \
                and canon(rtype.get(sel.room_key)) not in LOOK_BACK:
            prev = _cover_room(sel.room_key, i, prev)
            prev = _last_into_next(sel, i, prev, earlier)
            prev = _fly_on(sel.room_key, i, prev)
    if not moves:
        raise ValueError("no room allows a camera move")
    # a room is never shown by a close-up alone: without a room shot its close-up goes too
    room_filmed = {s.room_key for s, mv in moves if s.kind != "detail"}
    moves = [(s, mv) for s, mv in moves if s.room_key in room_filmed]
    # the closing length (allocate gave it to the last room's first shot) goes with the shot that now
    # closes the tour, when a coverage shot follows it
    # (not when planned at the 3D pace: a path played in another length plays at another speed - the
    # closing walk-in took a coverage shot's 4 s, ran at 3x and left the sync no solution, 2026-11-27)
    if moves and moves[-1][1].why.get("coverage_shot") and not getattr(mover, "pace_target", None):
        lead_ = first_s.get(moves[-1][0].room_key)
        if lead_ is not None and lead_.final_duration > moves[-1][0].final_duration:
            moves[-1][0].final_duration, lead_.final_duration = lead_.final_duration, moves[-1][0].final_duration
    # the tour's cap: the coverage shots showing the least new wall go first. Checked with room for
    # the beat sync, which only lengthens room shots afterwards (the Home Scan: 131 s past a 120 s cap)
    total_max = getattr(c3, "max_total_s", None) or 90.0
    if continuous and music_track:
        # continuous cuts: the sync only moves each cut within its handle - it added ~0.3 s a shot on
        # the green house (77 -> 79.8 s over 11 shots); SYNC_HEADROOM's 12% held 11 s back and cost
        # rooms their coverage angles (2026-11-27)
        cap = total_max - SYNC_DRIFT_S * len(moves)
    else:
        cap = SYNC_HEADROOM * total_max
    # 90 s is a soft limit (user 2026-11-27: "if the video tour needs an extra second or two we should
    # permit it"): shots shorten to the cap; a coverage angle or a flight is dropped only past it by
    # more than max_over_s
    cap_drop = cap + float(getattr(c3, "max_over_s", 0.0) or 0.0)
    # over the cap, shots longer than their kind's length (planner/lengths: hero / room) give time
    # back first, longest first, as far as the move's speed limits allow - a room shown whole by its
    # coverage shots matters more than one long take (user 2026-11-26: "the home office shot never
    # captured the whole room - showing most of each room has been a feature of the tours")
    from tour_worker.planner.lengths import length_range
    from tour_worker.planner3d.moves import FLIGHT, MIN_KIND_S, trial_speeds
    for s_, mv in sorted(moves, key=lambda m: -m[0].final_duration):
        over = sum(x.final_duration - handle for x, _ in moves) - cap
        if over <= 0:
            break
        if (mv.kind in FLIGHT or mv.why.get("coverage_shot") or s_.kind == "detail"
                or mv.why.get("seamless_in") or mv.why.get("seamless_out")):     # (one take: joined as planned)
            continue
        hi = length_range("hero" if s_.kind == "hero" else "room", c3.pacing)[1] * reach + handle
        lo = max(hi, MIN_KIND_S.get(mv.kind, 0.0) + handle, s_.final_duration - over)
        if lo >= s_.final_duration:
            continue
        if getattr(mover, "pace_target", None):
            # planned at the 3D pace: the way in goes with its time, the pace and the end it frames
            # kept (shortened in time alone a move sped up past the cut it was matched to)
            from tour_worker.planner3d.moves import trim_move_start
            trim_move_start(mv, lo / s_.final_duration)
            mv.why["trimmed_from_s"] = round(s_.final_duration, 2)
            s_.final_duration = round(lo, 2)
            continue
        ok = s_.final_duration
        for d in np.linspace(lo, s_.final_duration, 8):         # the shortest the speed limits allow
            turn, move_ = trial_speeds(mv.eye, mv.tgt, float(d), mv.zs, "linear" if continuous else mv.easing)
            if turn <= MAX_TURN_DPS * 0.95 and move_ <= MAX_MOVE_MPS * 0.95:
                ok = float(d)
                break
        if ok < s_.final_duration:
            mv.why["trimmed_from_s"] = round(s_.final_duration, 2)
            s_.final_duration = round(ok, 2)
    # then flights through doors give back time by starting nearer their door - the way there goes
    # with its time, the pace kept (shortened in time alone they sped up, and the speed band then cut
    # their start away: the study walk-in played from its door frame, 2026-11-27); never past
    # DOOR_LEAD before the room's door - before a room loses a coverage angle
    from tour_worker.planner3d.moves import trim_move_start
    for s_, mv in sorted(moves, key=lambda m: -m[0].final_duration):
        over = sum(x.final_duration - handle for x, _ in moves) - cap
        if over <= 0:
            break
        if mv.kind != "through":
            continue
        at = [space.room_at(q) for q in mv.eye]
        k = next((j for j in range(len(at)) if at[j] == s_.room_key), None)
        s_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(mv.eye, axis=0), axis=1))]
        if not k or s_arc[-1] < 1e-6:
            continue
        least = 1.0 - max(0.0, s_arc[k] - DOOR_LEAD) / s_arc[-1]
        # (the room's opening shot: never under a room shot's length - the tour's closing walk-in cut
        # to 4 s left the music sync no solution, 2026-11-27)
        kind_len = "hero" if (s_.kind == "hero" or mv is moves[-1][1]) else "room"   # (the closing shot: a hero's)
        floor = (length_range(kind_len, c3.pacing)[0] * reach + handle) / max(s_.final_duration, 1e-6)
        keep = max(least, floor, 1.0 - over / max(s_.final_duration, 1e-6))
        if keep < 0.98:
            trim_move_start(mv, keep)
            mv.why["trimmed_from_s"] = round(s_.final_duration, 2)
            s_.final_duration = round(s_.final_duration * keep, 2)
    for _, pair in sorted(extras, key=lambda e: e[0]):
        if sum(s_.final_duration - handle for s_, _ in moves) <= cap_drop:
            break
        if _own_scene_needed(moves, pair[1]):
            continue                                     # (the room's only scene besides its walk-in)
        moves = [m for m in moves if m[1] is not pair[1]]   # (the same move: the tuples differ)
        warnings.append(f"coverage_shot_dropped_for_length:{pair[0].room_key}")
    for chain in exits[::-1]:                     # then the flights on, from the tour's end, whole
        if sum(s_.final_duration - handle for s_, _ in moves) <= cap_drop:
            break
        gone = {id(mv) for _, mv in chain}
        k_after = next((n + 1 for n, (_, mv) in enumerate(moves) if mv is chain[-1][1]), None)
        moves = [m for m in moves if id(m[1]) not in gone]
        warnings.append(f"flight_dropped_for_length:{chain[0][0].room_key}")
        # the shot that followed the flight was chosen to carry it on: chosen again after what now
        # comes before it (the next room's first shot, entered as before)
        k_after = None if k_after is None else k_after - len(chain)
        if k_after is not None and 0 < k_after < len(moves):
            moves[k_after] = _rechoose(moves[k_after - 1], moves[k_after])
    filmed = {s.room_key for s, _ in moves}
    warnings += [f"room_skipped:{k}" for k in sorted({s.room_key for s in selected} - filmed)]
    for sel, _ in moves:              # a dropped angle gives its time to the room's shot: the same move, slower
        if sel.kind != "detail" and spare.get(sel.room_key):
            add = min(spare.pop(sel.room_key), c3.shot_max_s - sel.final_duration)
            sel.final_duration = round(sel.final_duration + max(0.0, add), 2)

    def _finish(moves_):
        """The shots of moves_ as they will play - fitted, synced to the music, in the speed band,
        captioned - as a plan, its rows and its own cut check (render/continuity): (plan, rows,
        cut rows, cut summary). Run again after a repair (below); it changes none of moves_."""
        plan_rows_: list[dict] = []
        warnings_ = list(warnings)
        shots: list[Shot] = []
        for n, (sel, mv) in enumerate(moves_, start=1):
            segs, err = fit_pair(mv.eye3, mv.tgt)
            if mv.kind != "detail":
                # the fitted curve's control points may overshoot its aims; the curve stays inside them,
                # so held under AIM_Z_MAX they keep a room shot from looking up (a long walk-in, 2026-11-27)
                from tour_worker.planner3d.moves import AIM_Z_MAX
                segs = [(e, np.c_[np.asarray(t)[:, :2], np.minimum(np.asarray(t)[:, 2], max(AIM_Z_MAX, float(np.max(mv.tgt[:, 2]))))])
                        for e, t in segs]
            bz = [BezierSegment(eye=np.round(space.to_scene(e), 4).tolist(), target=np.round(space.to_scene(t), 4).tolist())
                  for e, t in segs]
            role = sel.role                               # hero | detail (a room shot) | close
            shots.append(Shot(shot_id=f"s{n:02d}", order=n, asset_id="scene", room_key=sel.room_key,
                              room_type=rtype[sel.room_key], narrative_stage=stage[sel.room_key], role=role,
                              duration_s=round(sel.final_duration - (0.0 if mv.why.get("seamless_out") else handle), 3),
                              camera=CameraParams(type="bezier_path", easing=mv.easing, move=mv.kind, segments=bz,
                                                  hfov_deg=[HFOV, HFOV],
                                                  path_s=round(sel.final_duration, 3) if continuous else None),
                              transition_in=TransitionParams(type="cut", duration_s=0.0)))
            plan_rows_.append({"shot_id": f"s{n:02d}", "room_key": sel.room_key, "move": mv.kind,
                              "planned_s": round(sel.final_duration, 2), "score": round(mv.score, 3),
                              "piece": mv.piece.to_dict() if mv.piece is not None else None,
                              "fit_error_m": round(err, 3), **mv.why})

        music = None
        seam = [bool(mv.why.get("seamless_out")) for _, mv in moves_]
        has_seams = any(seam[:-1])
        if music_track and not has_seams:                 # the same beat sync, as an even retime
            from tour_worker.planner.music_sync import sync_to_music
            # the opening and closing shots sync as heroes (they hold); room shots may shorten to a hit (3 s floor); close-ups flex
            roles = ["hero" if s.role == "hero" or k in (0, len(shots) - 1) else s.role for k, s in enumerate(shots)]
            r = sync_to_music([s.duration_s for s in shots], music_track, c3, roles)
            for s, d in zip(shots, r.durations):
                s.duration_s = round(float(d), 4)
            music = r.music
            warnings_ += [x for x in [r.warning, *(r.extra_warnings or [])] if x]
        if continuous and shots:
            # the last shot's move ends on the tour's last frame (it settles there); the ends ease, the
            # cuts between run at a steady speed
            from tour_worker.planner.planner import set_continuous_easing
            shots[-1].camera.path_s = round(shots[-1].duration_s, 3)
            set_continuous_easing(shots)
            if getattr(c3, "speed_band", False):
                # one speed band, matched within 25% at every cut (item 4): 3D moves_ are only slowed
                from tour_worker.planner.speed_band import apply_speed_band_3d
                # a seamless join stays as planned: a piece joined at both ends is not slowed; joined at
                # its start, it is slowed from its end (its cut); joined at its end, from its start
                pinned = {}
                for k, (_, mv) in enumerate(moves_):
                    j_in, j_out = bool(mv.why.get("seamless_in")), bool(mv.why.get("seamless_out"))
                    if j_in and j_out:
                        pinned[k] = (0.0, 1.0)
                    elif j_in:
                        pinned[k] = (0.0, 0.0)
                    elif j_out:
                        # a walk-in joined on is not slowed at all (paced as planned, to the cut before
                        # it): its start is the walk-out and the door; else its end is kept
                        pinned[k] = (0.0, 1.0) if mv.kind == "through" else (1.0, 0.0)
                warnings_ += apply_speed_band_3d(shots, c3.pacing, cfg.fps, scene.get("up_axis"),
                                                keep_from={**_door_shares(moves_, space), **pinned})
                if has_seams:
                    _slow_joined(shots, moves_)
        if music_track and has_seams:
            # joined moves: the music last, on the lengths as they play (its moments: the real cuts
            # and the door crossings inside the joins)
            music, mw = _music_takes(shots, seam)
            warnings_ += mw

        # transitions: every interior shot change is a cut - the camera carries the move between parts
        # of the home (client review 2026-11-20, item 8); shots are built with cuts
        apply_captions(shots, cfg.output_room_labels is not None)
        # a walk through a door is the room's opening shot: its label - and the tour's room change -
        # comes where the camera enters the room, not where the shot starts (user 2026-11-27)
        t_tour = 0.0
        for s, row in zip(shots, plan_rows_):
            if s.camera.move in ("through", "bridge"):
                at = _enters_at(s, space, cfg.fps)
                if at is not None:
                    row["enters_room_at_s"] = at
                    row["enters_room_at_tour_s"] = round(t_tour + at, 2)
                    if s.caption:
                        s.caption_at_s = at
            t_tour += s.duration_s

        # the speed limits on the shots as they will play (after the sync) — interior shots only;
        # run before drone clip injection so zip(shots, plan_rows_) stays aligned
        for s, row in zip(shots, plan_rows_):
            turn, move = speeds_of([x.model_dump() for x in s.camera.segments], s.duration_s,
                                   getattr(s.camera.easing, "value", s.camera.easing), fps=cfg.fps, path_s=s.camera.path_s)
            row.update(played_s=s.duration_s, played_turn_dps=round(turn, 1), played_move_mps=round(move, 2))
            if turn > MAX_TURN_DPS + 0.05 or move > MAX_MOVE_MPS + 0.005:
                warnings_.append(f"camera_speed:{s.shot_id}")

        # inject drone video clips (entry_exterior before interior shots, rear_exterior after)
        if drone_clips:
            entry_clips = [c for c in drone_clips if c.get("role") == "entry_exterior"]
            rear_clips = [c for c in drone_clips if c.get("role") == "rear_exterior"]

            def _make_drone_shot(clip: dict, shot_id: str, narrative_stage: str) -> Shot:
                in_s = clip.get("in_point_s") or 0.0
                out_s = clip.get("out_point_s")
                duration_s = round(max(1.0, (out_s - in_s) if out_s is not None else 5.0), 3)
                cam = CameraParams(type="video_clip", clip_uri=clip["video_s3_url"],
                                   in_s=in_s if in_s > 0 else None,
                                   zoom_end=clip.get("zoom_end") or None)
                return Shot(shot_id=shot_id, order=0, asset_id="drone_clip",
                            room_key=clip.get("role", "exterior"), room_type="exterior",
                            narrative_stage=narrative_stage, role="hero",
                            duration_s=duration_s, camera=cam,
                            transition_in=TransitionParams(type="cut", duration_s=0.0))

            prepend: list[Shot] = []
            for k, clip in enumerate(entry_clips):
                prepend.append(_make_drone_shot(clip, f"d_e{k + 1:02d}", "opening"))
            if prepend and shots:
                # transition from last entry clip into first interior shot
                trans_type = entry_clips[-1].get("transition_to_interior", "cut")
                if trans_type == "fade":
                    shots[0].transition_in = TransitionParams(
                        type="fade", duration_s=round(min(FADE_S, 0.5 * shots[0].duration_s), 3))
                else:
                    shots[0].transition_in = TransitionParams(type="cut", duration_s=0.0)
            shots = prepend + shots

            for k, clip in enumerate(rear_clips):
                drone_shot = _make_drone_shot(clip, f"d_r{k + 1:02d}", "closing")
                if k == 0 and shots:
                    drone_shot.transition_in = TransitionParams(
                        type="fade", duration_s=round(min(FADE_S, 0.5 * shots[-1].duration_s), 3))
                shots.append(drone_shot)

            for n, s in enumerate(shots, start=1):
                s.order = n

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
            warnings=sorted(set(warnings_)),
            provenance=PlanProvenance(scoring_config_version=report.config.version,
                                      planner_config_version=f"{cfg.version}+{PLANNER3D_VERSION}"),
            music=music, voiceover=Voiceover(**cfg.voiceover) if cfg.voiceover else None,
            scene=Scene(uri=scene["uri"], sha256=scene["sha256"], format=scene["format"], up_axis=up,
                        json_uri=scene.get("json_uri"), json_sha256=scene.get("json_sha256"),
                        motion_uri=scene.get("motion_uri"), motion_sha256=scene.get("motion_sha256")))
        # QA (user 2026-11-27: "only 7 of 11 cuts in the limit - that will cause QA issues"): the plan
        # checks its own cuts as the render will (render/continuity.py) - each cut outside the limits is
        # named in the plan's warnings, with what broke
        cut_rows, cut_summary = [], None
        if continuous and len(shots) > 1:
            from tour_worker.render import continuity
            try:
                cut_rows = continuity.report(plan)
                cut_summary = continuity.summary(cut_rows)
                bad = [f"cut_out_of_limits:{c['cut']}:{c['from']}-{c['to']}:{'+'.join(c['breaks'])}"
                       for c in cut_rows if c.get("judged") and c.get("breaks")]
                if bad:
                    plan.warnings = sorted(set(plan.warnings) | set(bad))
            except Exception as e:                       # noqa: BLE001 - the check never fails a plan
                plan.warnings = sorted(set(plan.warnings) | {f"cut_check_failed:{type(e).__name__}"})
        return plan, plan_rows_, cut_rows, cut_summary

    def _stitch(ms):
        """One take wherever a walking route joins two moves (user 2026-11-27: enter a room, move
        round it, see its exit, walk through it into the next room and on, all in one shot; a cut
        - a transition - only where no route joins them): a bridge (RoomMoves.bridge) between
        each two, the pieces joined exactly - same place, look and pace (seamless_in / _out: they
        play their whole paths, the speed band leaves them, the cut check does not judge the join).
        Copies: ms is left as it is (repairs drop from it and stitch again)."""
        import dataclasses
        out: list = []
        for sel, mv in ms:
            mv = dataclasses.replace(mv, why={**mv.why, "base_id": id(mv)})
            if one_take and not take_built and out:
                psel, pmv = out[-1]
                br = mover.bridge(pmv, mv)
                if br is not None:
                    pmv.why["seamless_out"] = True
                    mv.why["seamless_in"] = True
                    # the next move's room - the room being left when that is a walk-in starting there
                    rk = psel.room_key if mv.kind == "through" else sel.room_key
                    dur = round(float(br.why["dur_s"]), 2)
                    bsel = dataclasses.replace(first_s.get(rk, sel), room_key=rk, role="detail", kind="room",
                                               is_hero=False, is_lead=False, final_duration=dur, raw_duration=dur)
                    br.why.update(seamless_in=True, seamless_out=True, base_id=None)
                    out.append((bsel, br))
            out.append((sel, mv))
        return out

    def _slow_joined(shots_, ms) -> None:
        """A joined walk-in (its walk-out and door kept whole: the band leaves it) entering faster than
        the cut before it allows is slowed by more time, not a shorter path - within a join the time
        is the planner's (the sync only places the real cuts) - at most JOIN_STRETCH_MAX."""
        from tour_worker.planner.speed_band import PLAN_RATIO, _scene_ends
        up = scene.get("up_axis")
        up = tuple(up) if up else (0.0, 0.0, 1.0)
        for k in range(1, len(shots_)):
            mv = ms[k][1]
            if not (mv.kind == "through" and mv.why.get("seamless_out") and not mv.why.get("seamless_in")):
                continue
            a, b = shots_[k - 1], shots_[k]
            out_v = _scene_ends(a.camera, a.duration_s, cfg.fps, up)[1]
            in_v = _scene_ends(b.camera, b.duration_s, cfg.fps, up)[0]
            if out_v > 1e-6 and in_v > PLAN_RATIO * out_v:
                f = min(JOIN_STRETCH_MAX, in_v / (PLAN_RATIO * out_v))
                b.duration_s = round(b.duration_s * f, 3)
                b.camera.path_s = round(b.duration_s, 3)
                mv.why["slowed_x"] = round(f, 2)

    def _played(ms) -> float:
        return sum(sel.final_duration - (0.0 if mv.why.get("seamless_out") else handle) for sel, mv in ms)

    def _music_takes(shots_, seam):
        """The music for a tour of takes (user 2026-11-27: room changes on the beat): the sync's
        moments go to the tour's real cuts and to each door crossing inside a take (where a room's
        label comes in); a crossing is moved onto its moment by the pace of the walk it is in (within
        TAKE_PACE_FLEX), a cut by its shot's length (within its handle). (music, warnings)."""
        from tour_worker.planner.music_sync import sync_to_music
        starts = np.r_[0.0, np.cumsum([s_.duration_s for s_ in shots_])]
        marks = []                                   # (tour time, kind, shot index, s into the shot)
        for k in range(1, len(shots_)):
            if not seam[k - 1]:
                marks.append((float(starts[k]), "cut", k, 0.0))
            elif shots_[k].room_key != shots_[k - 1].room_key and shots_[k].camera.move in ("bridge", "through"):
                at_ = _enters_at(shots_[k], space, cfg.fps)
                if at_ is not None and at_ > 0.5:
                    marks.append((float(starts[k]) + at_, "cross", k, at_))
        marks.sort()
        bounds = [m[0] for m in marks]
        segs = list(np.diff(np.r_[0.0, bounds, starts[-1]]))
        roles = ["hero"] + ["detail"] * (len(segs) - 2) + ["hero"] if len(segs) > 1 else ["hero"]
        r = sync_to_music([float(x) for x in segs], music_track, c3, roles)
        target = np.cumsum(r.durations)[:-1]
        for (t0, kind, k, off), want in zip(marks, target):
            now = float(np.r_[0.0, np.cumsum([s_.duration_s for s_ in shots_])][k]) + off
            shift = float(want) - now
            if kind == "cross":
                f = float(np.clip((off + shift) / off, 1 - TAKE_PACE_FLEX, 1 + TAKE_PACE_FLEX))
                shots_[k].duration_s = round(shots_[k].duration_s * f, 3)
                shots_[k].camera.path_s = round(shots_[k].duration_s, 3)
            else:
                prev_ = shots_[k - 1]
                lo_ = max(MIN_SHOT_S, (prev_.camera.path_s or prev_.duration_s) - 2 * handle)
                prev_.duration_s = round(float(np.clip(prev_.duration_s + shift, lo_, prev_.camera.path_s or 1e9)), 3)
        total = float(sum(s_.duration_s for s_ in shots_))
        music_ = r.music
        music_.end_s = round(total, 3)
        starts = np.r_[0.0, np.cumsum([s_.duration_s for s_ in shots_])]
        # each shot boundary's musical moment: a real cut its beat, a seamless join itself (not judged)
        beat_at = {k: float(w) for (_, kind, k, _), w in zip(marks, target) if kind == "cut"}
        music_.cut_beats_s = [round(beat_at.get(k, float(starts[k])), 3) for k in range(1, len(shots_))]
        return music_, [x for x in [r.warning, *(r.extra_warnings or [])] if x]

    # QA repair (user 2026-11-27: every cut within the limits): a cut that breaks with an extra angle
    # (a coverage shot) on either side loses that angle and the plan is finished again - the room
    # keeps the shots it has; a break with no extra angle in it stays named (cut_out_of_limits)
    repairs: list[str] = []
    for round_ in range(REPAIR_ROUNDS + 1):
        stitched = _stitch(moves)
        while one_take and extras and _played(stitched) > total_max + float(getattr(c3, "max_over_s", 0.0) or 0.0):
            # the walks between the moves count too: the coverage angles showing the least new wall go
            gone = min((e for e in extras if any(m[1] is e[1][1] for m in moves)), key=lambda e: e[0], default=None)
            if gone is None:
                break
            moves = [m for m in moves if m[1] is not gone[1][1]]
            repairs.append(f"coverage_shot_dropped_for_length:{gone[1][0].room_key}")
            stitched = _stitch(moves)
        plan, plan_rows, cut_rows, cut_summary = _finish(stitched)
        bad = [c for c in cut_rows if c.get("judged") and c.get("breaks")]
        if not bad or round_ == REPAIR_ROUNDS:
            break
        at = {f"s{n:02d}": n - 1 for n in range(1, len(stitched) + 1)}
        votes: dict[int, int] = {}
        for c in bad:
            for sid in (c["from"], c["to"]):
                n = at.get(sid)
                if (n is not None and stitched[n][1].why.get("coverage_shot")
                        and not _own_scene_needed(moves, next((m[1] for m in moves if id(m[1]) == stitched[n][1].why.get("base_id")), None))):
                    votes[n] = votes.get(n, 0) + 1
        if not votes:
            break
        n = max(votes, key=lambda k: (votes[k], -float(stitched[k][1].why.get("walls_new", 0.0) or 0.0)))
        repairs.append(f"coverage_dropped_breaks_cut:{stitched[n][0].room_key}:{'+'.join(sorted({b for c in bad if n in (at.get(c['from']), at.get(c['to'])) for b in c['breaks']}))}")
        drop = stitched[n][1].why.get("base_id")
        moves = [m for m in moves if id(m[1]) != drop]
    if repairs:
        plan.warnings = sorted(set(plan.warnings) | set(repairs))
    from tour_worker.planner.planner import _compute_plan_hash
    plan.provenance.plan_hash = _compute_plan_hash(plan.model_dump(mode="json"))
    rep = {
        "cuts": cut_summary,
        "scene": {"up": [round(float(x), 4) for x in space.up_scene], "up_source": space.frame.source,
                  "room_height_m": round(space.room_height, 2), "pieces": len(space.pieces),
                  "labels": labels is not None, "objects": len(space.objects),
                  "check_render": getattr(mover.occ, "backend", None) if mover.occ is not None else None},
        "rooms": [{"room_key": k, "room_type": rtype.get(k) or next((r.get("room_type") for kk, r in zip(keys, rooms) if kk == k), None),
                   "toured": k in rtype, "area_m2": round(space.room_area(k), 1),
                   "open_plan_with": sorted(space.spaces.get(k, {k}) - {k}),
                   "main_pieces": [p.to_dict() for p in space.main_pieces(k)],
                   "objects": sorted({o["class"] for o in space.room_objects(k)}),
                   "checks": {st: dict(c) for st, c in getattr(mover, "check_stats", {}).get(k, {}).items()}}
                  for k in space.room_keys],
        "sequence": [r for r in report.sequence],
        "shots": plan_rows,
        "room_changes": room_changes,
    }
    return plan, rep, report

