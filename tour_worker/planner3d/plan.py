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
LOOK_BACK = {"balcony", "patio", "outdoor_patio", "terrace", "deck", "porch", "garden", "backyard", "rooftop"}
MAX_WIDE = 3                  # wide shots a room at most (coverage: user 2026-11-16)
PLAN_BUDGET_S = 240.0         # s of planning before each shot settles for its best within LATE_CHECKS renders
SYNC_REACH = 0.85            # with music: shots planned at this share of their lengths (the sync lengthens)
SYNC_HEADROOM = 0.88          # the cap is met before the sync with this share of it
COVER_ENOUGH = 0.85           # a room whose walls are this covered gets no further shot
MAX_DETAILS = 0                # 3D close-ups off (user 2026-11-16, Home Scan: a painting rendered as a black
#                               blob, wall photos blurred - a scan rarely holds detail at 1 m); was 2
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
    mover.tour_shots = len(selected)               # the moving-in cap: a third of the tour's shots
    last_room = selected[-1].room_key if selected else None
    moves, plan_rows, prev = [], [], None
    poses: dict[str, list] = {}
    spare: dict[str, float] = {}
    covered: dict[str, np.ndarray] = {}           # per room: its wall cells the shots so far show
    wide: dict[str, int] = {}
    first_s: dict = {}
    extras: list = []                             # (walls_new, (sel, mv)) of the shots added for coverage

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
        return prev
    exits: list = []                              # flight chains (lists of (sel, mv)): dropped whole for length
    after_exit: set = set()                       # rooms whose first shot picks a matched cut at their door up
    flown_in: set = set()                         # rooms a through flight already entered
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

    def _leaves_flying(i: int) -> bool:
        """selected[i] is its room's last planned shot and a flight forward follows it."""
        nxt = selected[i + 1] if i + 1 < len(selected) else None
        if not (nxt is not None and nxt.room_key != selected[i].room_key and id(nxt) not in links
                and nxt.kind != "detail" and id(selected[i]) not in links):
            return False
        key = (selected[i].room_key, nxt.room_key)
        if key not in can_fly:                     # a dry run of the flight (geometry only, no renders)
            can_fly[key] = bool(_fly_on(selected[i].room_key, i, None, dry=True))
        return can_fly[key]

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
                                                 THROUGH_BEFORE, WALK_LEN)
        DOOR_GAP = EXIT_STOP + AFTER_EXIT[1] + 0.6
        nxt = selected[i + 1] if i + 1 < len(selected) else None
        if nxt is None or id(nxt) in links or nxt.kind == "detail":
            return prev
        dst = nxt.room_key
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
            end0 = (entry["s"] + THROUGH_AFTER[0], entry["s"] + THROUGH_AFTER[1])
            earlier0 = [p for k in sorted(space.spaces.get(dst, {dst})) for p in poses.get(k, [])]
            mv, w = mover.flight("through", dst, r, start0, end0, pace, handle, None if dry else prev, earlier0,
                                 aim=aim, piece=piece, end_room=dst, geometry_only=dry, start_room=room)
            if mv is None:                       # a bend past the door: look at one point in the room
                mv, w = mover.flight("through", dst, r, start0, end0, pace, handle, None if dry else prev, earlier0,
                                     aim=aim, piece=piece, end_room=dst, geometry_only=dry, start_room=room,
                                     look_fixed_s=entry["s"] + THROUGH_AFTER[1] + 1.0)
            if mv is not None:
                if dry:
                    return True
                dur = round(mv.why["dur_s"] + handle, 2)
                sel = dataclasses.replace(first_s[room], room_key=dst, role="detail", kind="room", is_hero=False,
                                          is_lead=False, final_duration=dur, raw_duration=dur)
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
                kind, end, end_room, aim_, piece_ = "through", (entry["s"] + THROUGH_AFTER[0], entry["s"] + THROUGH_AFTER[1]), dst, aim, piece
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
                                 start_room=room if j == 0 else None)
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
            sel = dataclasses.replace(first_s[room], room_key=rk, role="detail", kind="room", is_hero=False,
                                      is_lead=False, final_duration=dur, raw_duration=dur)
            chain.append((sel, mv))
            log["legs"].append({"kind": kind, "room": rk, "travel_m": mv.why["travel_m"], "dur_s": mv.why["dur_s"]})
            p_ = mv.last
        if dry:
            return bool(chain)
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

    for i, sel in enumerate(selected):
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
                               then_forward=_leaves_flying(i))
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
            prev = _fly_on(sel.room_key, i, prev)
    if not moves:
        raise ValueError("no room allows a camera move")
    # a room is never shown by a close-up alone: without a room shot its close-up goes too
    room_filmed = {s.room_key for s, mv in moves if s.kind != "detail"}
    moves = [(s, mv) for s, mv in moves if s.room_key in room_filmed]
    # the closing length (allocate gave it to the last room's first shot) goes with the shot that now
    # closes the tour, when a coverage shot follows it
    if moves and moves[-1][1].why.get("coverage_shot"):
        lead_ = first_s.get(moves[-1][0].room_key)
        if lead_ is not None and lead_.final_duration > moves[-1][0].final_duration:
            moves[-1][0].final_duration, lead_.final_duration = lead_.final_duration, moves[-1][0].final_duration
    # the tour's cap: the coverage shots showing the least new wall go first. Checked with room for
    # the beat sync, which only lengthens room shots afterwards (the Home Scan: 131 s past a 120 s cap)
    cap = SYNC_HEADROOM * (getattr(c3, "max_total_s", None) or 90.0)
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
        if mv.kind in FLIGHT or mv.why.get("coverage_shot") or s_.kind == "detail":
            continue
        hi = length_range("hero" if s_.kind == "hero" else "room", c3.pacing)[1] * reach + handle
        lo = max(hi, MIN_KIND_S.get(mv.kind, 0.0) + handle, s_.final_duration - over)
        if lo >= s_.final_duration:
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
    for _, pair in sorted(extras, key=lambda e: e[0]):
        if sum(s_.final_duration - handle for s_, _ in moves) <= cap:
            break
        moves = [m for m in moves if m[1] is not pair[1]]   # (the same move: the tuples differ)
        warnings.append(f"coverage_shot_dropped_for_length:{pair[0].room_key}")
    for chain in exits[::-1]:                     # then the flights on, from the tour's end, whole
        if sum(s_.final_duration - handle for s_, _ in moves) <= cap:
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

    shots: list[Shot] = []
    for n, (sel, mv) in enumerate(moves, start=1):
        segs, err = fit_pair(mv.eye3, mv.tgt)
        bz = [BezierSegment(eye=np.round(space.to_scene(e), 4).tolist(), target=np.round(space.to_scene(t), 4).tolist())
              for e, t in segs]
        role = sel.role                               # hero | detail (a room shot) | close
        shots.append(Shot(shot_id=f"s{n:02d}", order=n, asset_id="scene", room_key=sel.room_key,
                          room_type=rtype[sel.room_key], narrative_stage=stage[sel.room_key], role=role,
                          duration_s=round(sel.final_duration - handle, 3),
                          camera=CameraParams(type="bezier_path", easing=mv.easing, move=mv.kind, segments=bz,
                                              hfov_deg=[HFOV, HFOV],
                                              path_s=round(sel.final_duration, 3) if continuous else None),
                          transition_in=TransitionParams(type="cut", duration_s=0.0)))
        plan_rows.append({"shot_id": f"s{n:02d}", "room_key": sel.room_key, "move": mv.kind,
                          "planned_s": round(sel.final_duration, 2), "score": round(mv.score, 3),
                          "piece": mv.piece.to_dict() if mv.piece is not None else None,
                          "fit_error_m": round(err, 3), **mv.why})

    music = None
    if music_track:                                   # the same beat sync, as an even retime
        from tour_worker.planner.music_sync import sync_to_music
        # the opening and closing shots sync as heroes (they hold); room shots may shorten to a hit (3 s floor); close-ups flex
        roles = ["hero" if s.role == "hero" or k in (0, len(shots) - 1) else s.role for k, s in enumerate(shots)]
        r = sync_to_music([s.duration_s for s in shots], music_track, c3, roles)
        for s, d in zip(shots, r.durations):
            s.duration_s = round(float(d), 4)
        music = r.music
        warnings += [x for x in [r.warning, *(r.extra_warnings or [])] if x]
    if continuous and shots:
        # the last shot's move ends on the tour's last frame (it settles there); the ends ease, the
        # cuts between run at a steady speed
        from tour_worker.planner.planner import set_continuous_easing
        shots[-1].camera.path_s = round(shots[-1].duration_s, 3)
        set_continuous_easing(shots)
        if getattr(c3, "speed_band", False):
            # one speed band, matched within 25% at every cut (item 4): 3D moves are only slowed
            from tour_worker.planner.speed_band import apply_speed_band_3d
            warnings += apply_speed_band_3d(shots, c3.pacing, cfg.fps, scene.get("up_axis"))

    # transitions: every interior shot change is a cut - the camera carries the move between parts
    # of the home (client review 2026-11-20, item 8); shots are built with cuts
    apply_captions(shots, cfg.output_room_labels is not None)

    # the speed limits on the shots as they will play (after the sync) — interior shots only;
    # run before drone clip injection so zip(shots, plan_rows) stays aligned
    for s, row in zip(shots, plan_rows):
        turn, move = speeds_of([x.model_dump() for x in s.camera.segments], s.duration_s,
                               getattr(s.camera.easing, "value", s.camera.easing), fps=cfg.fps, path_s=s.camera.path_s)
        row.update(played_s=s.duration_s, played_turn_dps=round(turn, 1), played_move_mps=round(move, 2))
        if turn > MAX_TURN_DPS + 0.05 or move > MAX_MOVE_MPS + 0.005:
            warnings.append(f"camera_speed:{s.shot_id}")

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

