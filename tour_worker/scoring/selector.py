"""Group scored photos into rooms, pick each room's hero shot, rank rooms by importance, assign
screen-time multipliers and a narrative order (SOW 5.2C). Deterministic: ties are broken by
coverage, then lighting, then photo_id. No numpy/cv2 imports: usable in a service that only
receives PhotoScore JSON."""
from collections import defaultdict

from .config import ScoringConfig
from .rooms import stage_for, stage_index, tier_for, tier_index
from .schemas import PhotoScore, RankedPhoto, RoomResult, ScoringReport


def _sort_key(p: PhotoScore) -> tuple:
    cov = p.criteria["coverage"].score or 0.0
    lit = p.criteria["lighting"].score or 0.0
    return (-p.total, -cov, -lit, p.photo_id)


def build_rooms(scores: list[PhotoScore], cfg: ScoringConfig) -> list[RoomResult]:
    groups: dict[str, list[PhotoScore]] = defaultdict(list)
    for s in scores:
        groups[s.room_key].append(s)

    partial = []
    for key, photos in groups.items():
        photos = sorted(photos, key=_sort_key)
        room_type = next((p.room_type for p in sorted(photos, key=lambda p: p.photo_id) if p.room_type), None)
        partial.append((tier_for(room_type), stage_for(room_type), key, room_type, photos[0], photos))

    partial.sort(key=lambda t: (tier_index(t[0]), -t[4].total, t[2]))      # importance order
    rank_of = {t[2]: i for i, t in enumerate(partial, start=1)}
    narrative = sorted(partial, key=lambda t: (stage_index(t[1]), rank_of[t[2]]))
    order_of = {t[2]: i for i, t in enumerate(narrative, start=1)}

    rooms: list[RoomResult] = []
    for tier, stage, key, room_type, hero, photos in partial:
        rank = rank_of[key]
        if rank <= cfg.ranking.hero_room_count:
            mult = cfg.ranking.hero_multiplier
        elif tier == "exterior":
            mult = cfg.ranking.exterior_multiplier
        else:
            mult = 1.0
        rooms.append(RoomResult(
            room_key=key, room_type=room_type, tier=tier, importance_rank=rank,
            screen_time_multiplier=mult, narrative_stage=stage, narrative_order=order_of[key],
            hero_photo_id=hero.photo_id, hero_total=hero.total, hero_flags=hero.flags,
            photos=[RankedPhoto(photo_id=p.photo_id, total=p.total) for p in photos],
        ))
    return rooms


def build_report(scores: list[PhotoScore], cfg: ScoringConfig | None = None,
                 warnings: list[str] | None = None) -> ScoringReport:
    cfg = cfg or ScoringConfig()
    ordered = sorted(scores, key=lambda s: s.photo_id)        # input order must not change results
    rooms = build_rooms(ordered, cfg)
    sequence = [r.room_key for r in sorted(rooms, key=lambda r: r.narrative_order)]
    return ScoringReport(config=cfg, photos=ordered, rooms=rooms, sequence=sequence,
                         warnings=warnings or [])
