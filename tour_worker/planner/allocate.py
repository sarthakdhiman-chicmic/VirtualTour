from dataclasses import dataclass

from tour_worker.scoring.schemas import ScoringReport

from .config import PlannerConfig
from .scene_context import SceneContext


@dataclass
class SelectedShot:
    photo_id: str
    room_key: str
    role: str
    raw_duration: float
    order_idx: int = 0
    final_duration: float = 0.0
    is_hero: bool = False
    narrative_order: int = 0
    importance_rank: int = 0
    is_low_score: bool = False
    is_lead: bool = False           # the room's first shot (hero or not): trimming drops the room
    kind: str = "room"              # planner/lengths.py type: room | hero | detail (a close-up)


def natural_length(role: str, ctx: SceneContext | None, cfg: PlannerConfig) -> float:
    """A shot's own length from what its photo shows (richness 0..1): a normal shot 2.8-3.6 s,
    a hero 3.2-5.0 s - a hero with little in it gets the short end (review 2026-10-19: "the
    kitchen is a hero shot, but for what there is we are giving it too much time")."""
    lo, hi = cfg.hero_shot_s if role == "hero" else cfg.normal_shot_s
    r = ctx.richness if ctx is not None else 0.5
    return lo + r * (hi - lo)


def pace_factor(cfg: PlannerConfig) -> float:
    from .config import PACING
    return PACING.get(cfg.pacing, 1.0)


def room_shots(budget: float, n_photos: int, cfg: PlannerConfig, pace: float, richness: float) -> list[float]:
    """A room's time budget split into its shots: one shot per room_shot_s of budget (paced), as
    many as it has photos and max_detail_shots_per_room allow. The lead shot takes the larger
    share (more for a rich photo); every shot within shot_min_s..shot_max_s."""
    n = max(1, min(n_photos, 1 + cfg.max_detail_shots_per_room, int(budget // (cfg.room_shot_s * pace))))
    if n == 1:
        lengths = [budget]
    else:
        lead = budget * (0.40 + 0.15 * richness) if n > 2 else budget * (0.50 + 0.10 * richness)
        lengths = [lead] + [(budget - lead) / (n - 1)] * (n - 1)
    return [max(cfg.shot_min_s, min(cfg.shot_max_s, d)) for d in lengths]


def _is_close(report: ScoringReport, photo_id: str) -> bool:
    """A close-up (fixtures, hardware, finishes): the scoring's coverage_class "tight"."""
    p = next((x for x in report.photos if x.photo_id == photo_id), None)
    return bool(p is not None and p.motion is not None and p.motion.coverage_class == "tight")


def allocate_content(
    report: ScoringReport, cfg: PlannerConfig, contexts: dict[str, SceneContext]
) -> tuple[list[SelectedShot], list[str]]:
    """Shots and their lengths by shot type (planner/lengths.py, client standards 2026-11-13):
      - every room in narrative order; how many room shots it gets comes from its time budget
        (room_base_s x its screen-time multiplier - hero rooms 2x, exteriors 1.5x - x the pacing),
        as many as it has photos and max_detail_shots_per_room allow;
      - a hero room's shots are hero shots (5-8 s at Normal), others room shots (3.5-5 s); a
        room's first shot sits where its richness puts it in the range, later angles a little lower;
      - a close-up photo (scoring coverage "tight") is a detail shot (1.5-2.5 s) right after the
        room's first shot - at most one per room, never a room's only shot;
      - the opening and closing shots are hero length (the closing one at least final_shot_min_s);
      - under min_total_s room and hero shots are stretched (up to their range's ceiling), never
        padded; over max_total_s detail shots go first, then other angles, then whole rooms."""
    from .lengths import ceiling_s, length_range, shot_length
    warnings: list[str] = []
    pace = pace_factor(cfg)
    rooms_in_sequence = [r for k in report.sequence for r in report.rooms if r.room_key == k]
    low = {p.photo_id for p in report.photos if "low_score" in p.flags}
    exterior = {r.room_key for r in rooms_in_sequence if r.tier == "exterior"}
    shots: list[SelectedShot] = []
    for room in rooms_in_sequence:
        hero = room.screen_time_multiplier >= 2.0
        ctx = contexts.get(room.hero_photo_id)
        rich = ctx.richness if ctx is not None else 0.5
        boost = cfg.outdoor_boost if room.tier == "exterior" else 1.0
        budget = cfg.room_base_s * room.screen_time_multiplier * pace * boost
        photos = [room.hero_photo_id] + [p.photo_id for p in room.photos
                                         if p.photo_id != room.hero_photo_id and p.photo_id not in low]
        views = [pid for pid in photos if not _is_close(report, pid)] or photos[:1]
        closes = [pid for pid in photos if pid not in views][:1]
        n = max(1, min(len(views), 1 + cfg.max_detail_shots_per_room, int(budget // (cfg.room_shot_s * pace))))
        kind = "hero" if hero else "room"
        room_shots = []
        lead_s = 0.0
        for k, pid in enumerate(views[:n]):
            # a hero room filmed in one shot gets the top of the hero range: still clearly longer;
            # a room's later angles run 85% of its first (a room's shots never all match)
            if k == 0:
                r = max(rich, 0.85) if (hero and n == 1) else rich
                d = lead_s = shot_length(kind, cfg.pacing, r) * boost
            else:
                d = max(length_range("room", cfg.pacing)[0], 0.85 * lead_s / boost)
            room_shots.append(SelectedShot(
                photo_id=pid, room_key=room.room_key, role="hero" if (hero and k == 0) else "detail",
                raw_duration=d, is_hero=hero and k == 0,
                is_lead=k == 0, narrative_order=room.narrative_order, importance_rank=room.importance_rank,
                kind=kind))
        for pid in closes:
            room_shots.insert(1, SelectedShot(
                photo_id=pid, room_key=room.room_key, role="close",
                raw_duration=shot_length("detail", cfg.pacing, rich), narrative_order=room.narrative_order,
                importance_rank=room.importance_rank, kind="detail"))
        shots += room_shots
    if shots:                                     # the long establishing and closing shots
        shots[0].raw_duration = max(shots[0].raw_duration, shot_length("hero", cfg.pacing, rich_of(contexts, shots[0])))
        shots[0].role, shots[0].kind = "hero", "hero"
        if shots[-1].kind == "detail" and len(shots) > 1:     # a tour does not close on a close-up
            shots.pop()
        shots[-1].raw_duration = max(shots[-1].raw_duration, shot_length("hero", cfg.pacing, 0.5),
                                     cfg.final_shot_min_s * pace)
        if shots[-1].kind == "room":
            shots[-1].kind = "hero"
        shots[-1].role = "hero" if shots[-1].kind == "hero" else shots[-1].role   # it holds like one

    def total():
        return sum(s.raw_duration for s in shots)

    while total() > cfg.max_total_s and shots:
        drop = ([s for s in shots if s.kind == "detail"] or [s for s in shots if not s.is_lead])
        if drop:
            shots.remove(max(drop, key=lambda s: s.importance_rank))
        else:
            worst = max(shots, key=lambda s: s.importance_rank).room_key
            shots = [s for s in shots if s.room_key != worst]
        if "trimmed" not in warnings:
            warnings.append("trimmed")

    # a standard listing lands at 60 s or more; Fast pacing keeps its pace (its floor is 0.75 x 60):
    # room and hero shots stretch, up to their range's ceiling; detail shots stay short
    floor = cfg.min_total_s * min(1.0, pace)
    if shots and total() < floor:
        # every room / hero shot grows by the same factor (up to its range's ceiling, a boosted
        # exterior's x its boost): heroes stay longer than rooms, the tour does not even out
        caps = {id(s): ceiling_s(s.kind, cfg.pacing) * (cfg.outdoor_boost if s.room_key in exterior else 1.0)
                for s in shots if s.kind != "detail"}
        for _ in range(8):
            free = [s for s in shots if id(s) in caps and s.raw_duration < caps[id(s)] - 1e-6]
            short = floor - total()
            if short <= 1e-6 or not free:
                break
            f = 1.0 + short / sum(s.raw_duration for s in free)
            for s in free:
                s.raw_duration = min(caps[id(s)], s.raw_duration * f)

    for i, s in enumerate(shots):
        s.order_idx = i + 1
        cap = ceiling_s(s.kind, cfg.pacing) * (cfg.outdoor_boost if s.room_key in exterior else 1.0)
        s.final_duration = round(min(cap * (1.5 if i == len(shots) - 1 else 1.0), s.raw_duration), 2)
    return shots, warnings


def rich_of(contexts: dict, shot: SelectedShot) -> float:
    ctx = contexts.get(shot.photo_id)
    return ctx.richness if ctx is not None else 0.5


def allocate_shots(
    report: ScoringReport, cfg: PlannerConfig, contexts: dict[str, SceneContext] | None = None
) -> tuple[list[SelectedShot], list[str]]:
    if cfg.length_mode == "content":
        return allocate_content(report, cfg, contexts or {})
    warnings = []

    # 1. Initial selection: One hero shot per room
    shots: list[SelectedShot] = []
    used_photos = set()

    # Sort rooms by narrative order
    rooms_in_sequence = []
    for r_key in report.sequence:
        room = next((r for r in report.rooms if r.room_key == r_key), None)
        if room:
            rooms_in_sequence.append(room)

    # A room with one photo is a hero only when it is one of the hero rooms (the top
    # hero_room_count by importance); the rest play as normal-length shots, so a tour of
    # single-photo rooms mixes long and short shots instead of being all heroes (review 2026-10-19).
    hero_rooms = report.config.ranking.hero_room_count
    for room in rooms_in_sequence:
        hero = len(room.photos) > 1 or room.importance_rank <= hero_rooms
        factor = 1.0 if hero else cfg.detail_shot_factor
        shots.append(
            SelectedShot(
                photo_id=room.hero_photo_id,
                room_key=room.room_key,
                role="hero" if hero else "detail",
                raw_duration=cfg.base_shot_s * room.screen_time_multiplier * factor,
                is_hero=hero,
                is_lead=True,
                narrative_order=room.narrative_order,
                importance_rank=room.importance_rank,
            )
        )
        used_photos.add(room.hero_photo_id)

    # 2. Add detail shots while the shots cannot fill the target length even at shot_max_s
    # (filling toward target_total_s, not just min_total_s: with ~6 s shots a tour of heroes only
    # came out short, and its durations were pushed past shot_max_s)
    detail_counts = {r.room_key: 0 for r in report.rooms}
    fill_to = max(cfg.min_total_s, cfg.target_total_s)

    while len(shots) * cfg.shot_max_s < fill_to:
        # Find next best detail shot
        best_candidate = None
        best_room = None

        # Sort rooms by importance_rank (lower is better, assuming 1 is most important)
        # We will try to add a detail shot by visiting rooms in importance_rank order.
        rooms_by_importance = sorted(report.rooms, key=lambda r: r.importance_rank)

        # Two passes: first honoring max_detail_shots_per_room, then relaxing
        for pass_relax in [False, True]:
            # Two passes for low_score: first without low_score, then allowing low_score
            for pass_low_score in [False, True]:
                for room in rooms_by_importance:
                    if (
                        not pass_relax
                        and detail_counts[room.room_key] >= cfg.max_detail_shots_per_room
                    ):
                        continue

                    # Find highest ranked unused photo in this room
                    for p in room.photos:
                        if p.photo_id not in used_photos:
                            # Check low_score flag
                            is_low = False
                            for rp in report.photos:
                                if rp.photo_id == p.photo_id:
                                    is_low = "low_score" in rp.flags
                                    break

                            if is_low and not pass_low_score:
                                continue

                            best_candidate = p
                            best_room = room
                            break

                    if best_candidate:
                        break
                if best_candidate:
                    break
            if best_candidate:
                break

        if best_candidate and best_room:
            # Insert right after its room's hero (or existing details)
            # Find the last shot of this room
            insert_idx = len(shots)
            for i in range(len(shots) - 1, -1, -1):
                if shots[i].room_key == best_room.room_key:
                    insert_idx = i + 1
                    break

            is_low = False
            for rp in report.photos:
                if rp.photo_id == best_candidate.photo_id:
                    is_low = "low_score" in rp.flags
                    break

            shots.insert(
                insert_idx,
                SelectedShot(
                    photo_id=best_candidate.photo_id,
                    room_key=best_room.room_key,
                    role="detail",
                    raw_duration=cfg.base_shot_s * cfg.detail_shot_factor,
                    is_hero=False,
                    narrative_order=best_room.narrative_order,
                    importance_rank=best_room.importance_rank,
                    is_low_score=is_low,
                ),
            )
            used_photos.add(best_candidate.photo_id)
            detail_counts[best_room.room_key] += 1
        else:
            # out of photos: only a problem if even shot_max_s cannot reach the minimum
            if len(shots) * cfg.shot_max_s < cfg.min_total_s:
                warnings.append("below_min_total")
            break

    # 3. Trim if too many shots
    # while n_shots * shot_min_s > max_total_s
    while len(shots) * cfg.shot_min_s > cfg.max_total_s:
        # Drop detail shots from least important rooms first
        # (Importance rank: higher number = less important)
        dropped = False

        # Try finding a detail shot to drop
        detail_shots = [s for s in shots if not s.is_hero and not s.is_lead]
        if detail_shots:
            # Sort by importance_rank descending (least important first)
            # Tie breaker: we could drop the last detail shot of that room
            detail_shots.sort(key=lambda s: s.importance_rank, reverse=True)
            worst_detail = detail_shots[0]
            shots.remove(worst_detail)
            dropped = True
        else:
            # Drop whole rooms by lowest importance
            rooms_least_important = sorted(
                set(s.room_key for s in shots),
                key=lambda rk: next(s.importance_rank for s in shots if s.room_key == rk),
                reverse=True,
            )
            if rooms_least_important:
                worst_room_key = rooms_least_important[0]
                shots = [s for s in shots if s.room_key != worst_room_key]
                dropped = True

        if not dropped:
            break
        if "trimmed" not in warnings:
            warnings.append("trimmed")

    # Reassign sequential order indices
    for i, s in enumerate(shots):
        s.order_idx = i + 1

    # 4. Duration Allocation
    # Scale all raw durations so sum equals target, clamp each, redistribute residual

    raw_sum = sum(s.raw_duration for s in shots)
    target = max(cfg.min_total_s, min(cfg.target_total_s, raw_sum))
    # a total the shots cannot hold would push durations outside [shot_min_s, shot_max_s]
    target = min(max(target, len(shots) * cfg.shot_min_s), len(shots) * cfg.shot_max_s)

    for s in shots:
        s.final_duration = s.raw_duration

    for _ in range(20):
        current_sum = sum(s.final_duration for s in shots)
        if current_sum == 0:
            break

        ratio = target / current_sum
        for s in shots:
            s.final_duration *= ratio

        # Clamp and compute residual
        residual = 0.0
        clamped_count = 0
        unclamped = []

        for s in shots:
            if s.final_duration < cfg.shot_min_s:
                residual += s.final_duration - cfg.shot_min_s
                s.final_duration = cfg.shot_min_s
                clamped_count += 1
            elif s.final_duration > cfg.shot_max_s:
                residual += s.final_duration - cfg.shot_max_s
                s.final_duration = cfg.shot_max_s
                clamped_count += 1
            else:
                unclamped.append(s)

        if abs(residual) < 0.001 or len(unclamped) == 0:
            break

        # Apply residual to unclamped
        adj = residual / len(unclamped)
        for s in unclamped:
            s.final_duration += adj

    # Raise final shot
    if shots:
        final_shot = shots[-1]
        if final_shot.final_duration < cfg.final_shot_min_s:
            final_shot.final_duration = cfg.final_shot_min_s

    # Round to 0.01s
    for s in shots:
        s.final_duration = round(s.final_duration, 2)

    total_dur = sum(s.final_duration for s in shots)
    if total_dur < cfg.min_total_s or total_dur > cfg.max_total_s:
        warnings.append("total_out_of_range")

    return shots, warnings
