"""Shot lengths (client standards 2026-11-13, planner/lengths.py): room / hero / detail shots by
pacing, hero rooms clearly longer, a long opening and closing shot, 60-120 s tours."""
import cv2
import numpy as np
import pytest

from tour_worker.planner.allocate import allocate_shots
from tour_worker.planner.config import PlannerConfig
from tour_worker.scoring.pipeline import score_photos

# a listing like Turkey Trot: a family room in the living tier next to the living room, a primary
# bedroom and two other bedrooms, exteriors at both ends
ROOMS = ["front_exterior", "entry", "family_room", "living_room", "living_room", "kitchen", "kitchen",
         "primary_bedroom", "bedroom", "bedroom", "bathroom", "patio", "pool_area"]


def _photo(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((640, 960, 3)) * 255).astype(np.uint8), (0, 0), 3)
    for _ in range(6):
        x, y = int(rng.integers(0, 900)), int(rng.integers(0, 600))
        cv2.rectangle(img, (x, y), (x + 60, y + 40), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture(scope="module")
def report():
    # one room per photo except the repeated living_room / kitchen / bedroom types (room_key per type)
    return score_photos([(f"p{i:02d}", _photo(i), rt) for i, rt in enumerate(ROOMS)])


def _room_time(shots):
    out = {}
    for s in shots:
        out[s.room_key] = out.get(s.room_key, 0.0) + s.final_duration
    return out


def _in_range(s, pacing, last=False):
    from tour_worker.planner.lengths import ceiling_s, floor_s
    top = ceiling_s(s.kind, pacing) * (1.5 if last else 1.0)
    return floor_s(s.kind, pacing) - 1e-6 <= s.final_duration <= top + 1e-6


def test_hero_rooms_get_clearly_more_time(report):
    cfg = PlannerConfig(min_total_s=0.0)          # no stretching: the lengths as planned
    shots, _ = allocate_shots(report, cfg, {})
    t = _room_time(shots)
    mult = {r.room_key: r.screen_time_multiplier for r in report.rooms}
    primary = next(r.room_key for r in report.rooms if r.room_type == "primary_bedroom")
    family = next(r.room_key for r in report.rooms if r.room_type == "family_room")
    assert mult[primary] == 2.0 and mult[family] == 1.0        # the primary bedroom, not the family room
    inner = [k for k in t if k not in (shots[0].room_key, shots[-1].room_key)]   # not the long ends
    normal = [t[k] for k in inner if mult[k] == 1.0]
    room_shots = [s.final_duration for s in shots if s.kind == "room"]
    for k in inner:
        if mult[k] == 2.0:
            assert t[k] >= 1.15 * max(normal)
            lead = next(s for s in shots if s.room_key == k)
            assert lead.kind == "hero" and lead.final_duration >= 5.0 and lead.final_duration >= max(room_shots) + 2.0


def test_shots_follow_the_length_table_with_long_opening_and_closing_shots(report):
    cfg = PlannerConfig()
    shots, _ = allocate_shots(report, cfg, {})
    assert all(_in_range(s, "normal", last=k == len(shots) - 1) for k, s in enumerate(shots))
    assert shots[0].kind == "hero" and shots[-1].kind == "hero" and shots[-1].final_duration >= cfg.final_shot_min_s
    assert 60.0 - 1e-6 <= sum(s.final_duration for s in shots) <= 120.0
    lengths = sorted({round(s.final_duration, 1) for s in shots})
    assert len(lengths) >= 4                                   # the tour does not even out


@pytest.mark.parametrize("slow,fast", [("slow", "normal"), ("normal", "fast")])
def test_pacing_scales_the_tour(report, slow, fast):
    a, _ = allocate_shots(report, PlannerConfig(pacing=slow, min_total_s=0.0), {})
    b, _ = allocate_shots(report, PlannerConfig(pacing=fast, min_total_s=0.0), {})
    assert sum(s.final_duration for s in a) > sum(s.final_duration for s in b)
    assert all(_in_range(s, slow, k == len(a) - 1) for k, s in enumerate(a))
    assert all(_in_range(s, fast, k == len(b) - 1) for k, s in enumerate(b))


def test_a_small_listing_is_stretched_toward_60_s_evenly_by_factor_never_past_a_ceiling():
    from tour_worker.planner.lengths import ceiling_s
    small = score_photos([(f"p{i}", _photo(i), rt) for i, rt in enumerate(["exterior", "kitchen", "bedroom"])])
    shots, _ = allocate_shots(small, PlannerConfig(), {})
    assert all(s.final_duration <= ceiling_s(s.kind, "normal") * 1.5 + 1e-6 for s in shots)
    cap = sum(ceiling_s(s.kind, "normal") * (1.5 if k == len(shots) - 1 else 1) for k, s in enumerate(shots))
    assert sum(s.final_duration for s in shots) <= min(60.0, cap) + 0.1


def test_a_close_up_photo_is_a_short_detail_shot_after_the_rooms_first_shot(report):
    """A photo the scoring sees as a close-up (coverage "tight": a fixture, a finish) plays as a
    1.5-2.5 s detail shot (role "close") right after its room's first shot."""
    kitchen = [p.photo_id for p in report.photos if p.room_type == "kitchen"]
    rep = report.model_copy(deep=True)
    for p in rep.photos:
        if p.motion is not None:
            p.motion.coverage_class = "tight" if p.photo_id == kitchen[1] else "wide"
    shots, _ = allocate_shots(rep, PlannerConfig(min_total_s=0.0), {})
    ks = [s for s in shots if s.room_key == next(s.room_key for s in shots if s.photo_id == kitchen[0])]
    assert [s.kind for s in ks][:2] == ["hero", "detail"] and ks[1].role == "close" and ks[1].photo_id == kitchen[1]
    assert 1.5 - 1e-6 <= ks[1].final_duration <= 2.5 + 1e-6


def test_the_living_room_is_the_hero_even_when_a_family_room_scores_higher():
    """The demo: the family room (living tier) outscored the living room and took its 2x."""
    rep = score_photos([(f"p{i}", _photo(i), rt) for i, rt in enumerate(["family_room", "living_room", "kitchen"])])
    mult = {r.room_type: r.screen_time_multiplier for r in rep.rooms}
    assert mult["living_room"] == 2.0 and mult["family_room"] == 1.0


def test_the_sync_extends_room_and_hero_shots_but_never_shortens_them():
    """Client 2026-11-13: when a cut needs to land on a beat, extend to the next beat rather than
    shortening a room shot; close-ups are the flexible ones."""
    from tour_worker.planner.music_sync import length_bounds
    cfg = PlannerConfig()
    for wide in (False, True):
        lo, hi = length_bounds([8.0, 4.0, 2.0], ["hero", "detail", "close"], cfg, wide=wide)
        assert lo[0] >= 8.0 - 1e-6 and lo[1] >= 4.0 - 1e-6 and hi[0] > 8.0 and hi[1] > 4.0
        assert lo[2] < 2.0 and hi[2] > 2.0
    lo, hi = length_bounds([8.0], ["hero"], PlannerConfig(music_shot_abs_max_s=6.0))
    assert lo[0] <= hi[0]                                         # never an empty window
