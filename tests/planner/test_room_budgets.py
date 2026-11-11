"""Room budgets (client review 2026-11-05): hero rooms 2x, exteriors 1.5x, 3-8 s shots, a long
opening and closing shot, Slow / Normal / Fast pacing, 60-120 s tours."""
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


def test_hero_rooms_get_twice_a_normal_rooms_time(report):
    cfg = PlannerConfig(min_total_s=0.0)          # no stretching: the budgets as planned
    shots, _ = allocate_shots(report, cfg, {})
    t = _room_time(shots)
    mult = {r.room_key: r.screen_time_multiplier for r in report.rooms}
    primary = next(r.room_key for r in report.rooms if r.room_type == "primary_bedroom")
    family = next(r.room_key for r in report.rooms if r.room_type == "family_room")
    assert mult[primary] == 2.0 and mult[family] == 1.0        # the primary bedroom, not the family room
    inner = [k for k in t if k not in (shots[0].room_key, shots[-1].room_key)]   # not the long ends
    normal = [t[k] for k in inner if mult[k] == 1.0]
    for k in inner:
        if mult[k] == 2.0:
            assert t[k] >= 1.8 * max(normal)


def test_shots_are_3_to_8_s_with_long_opening_and_closing_shots(report):
    cfg = PlannerConfig()
    shots, _ = allocate_shots(report, cfg, {})
    assert all(3.0 - 1e-6 <= s.final_duration <= 8.0 + 1e-6 for s in shots)
    assert shots[0].final_duration >= cfg.establishing_min_s and shots[-1].final_duration >= cfg.final_shot_min_s
    assert 60.0 - 1e-6 <= sum(s.final_duration for s in shots) <= 120.0


@pytest.mark.parametrize("slow,fast", [("slow", "normal"), ("normal", "fast")])
def test_pacing_scales_the_tour(report, slow, fast):
    a, _ = allocate_shots(report, PlannerConfig(pacing=slow, min_total_s=0.0), {})
    b, _ = allocate_shots(report, PlannerConfig(pacing=fast, min_total_s=0.0), {})
    assert sum(s.final_duration for s in a) > sum(s.final_duration for s in b)
    assert all(3.0 - 1e-6 <= s.final_duration <= 8.0 + 1e-6 for s in a + b)


def test_a_small_listing_is_stretched_toward_60_s_but_never_past_8_s_a_shot():
    small = score_photos([(f"p{i}", _photo(i), rt) for i, rt in enumerate(["exterior", "kitchen", "bedroom"])])
    shots, _ = allocate_shots(small, PlannerConfig(), {})
    assert all(s.final_duration <= 8.0 + 1e-6 for s in shots)
    assert sum(s.final_duration for s in shots) == pytest.approx(min(60.0, 8.0 * len(shots)), abs=0.1)


def test_the_living_room_is_the_hero_even_when_a_family_room_scores_higher():
    """The demo: the family room (living tier) outscored the living room and took its 2x."""
    rep = score_photos([(f"p{i}", _photo(i), rt) for i, rt in enumerate(["family_room", "living_room", "kitchen"])])
    mult = {r.room_type: r.screen_time_multiplier for r in rep.rooms}
    assert mult["living_room"] == 2.0 and mult["family_room"] == 1.0


def test_a_hero_shot_keeps_its_time_through_the_sync():
    from tour_worker.planner.music_sync import length_bounds
    cfg = PlannerConfig()
    lo, hi = length_bounds([8.0, 4.0], ["hero", "detail"], cfg, wide=True)
    assert lo[0] >= 0.9 * 8.0 - 1e-6 and lo[1] < 0.9 * 4.0       # heroes never shrink past 0.9x
    lo, hi = length_bounds([8.0], ["hero"], PlannerConfig(music_shot_abs_max_s=6.0))
    assert lo[0] <= hi[0]                                         # never an empty window
