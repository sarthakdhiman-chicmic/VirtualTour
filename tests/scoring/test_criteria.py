"""tour_worker.scoring: each criterion ranks good photos above bad ones, says when it cannot
measure, and the selector's rooms, heroes and order do not depend on the input order."""
import random

import cv2
import numpy as np
import pytest

from tests.perceptual_fixtures import _photo
from tour_worker.scoring.config import ScoringConfig
from tour_worker.scoring.coverage import score_coverage
from tour_worker.scoring.depth_of_field import score_depth_of_field
from tour_worker.scoring.lighting import score_lighting
from tour_worker.scoring.pipeline import score_photos

CFG = ScoringConfig()


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _room(seed=0):
    """A synthetic room: lit walls, a floor, a window, furniture edges and some texture."""
    rng = np.random.default_rng(seed)
    img = np.full((600, 900, 3), 150, np.uint8)
    img[420:] = (95, 110, 125)                                    # floor
    img[:70] = (215, 215, 210)                                    # ceiling
    cv2.line(img, (0, 420), (899, 420), (60, 60, 60), 3)          # wall / floor edge
    cv2.line(img, (0, 70), (899, 70), (120, 120, 120), 2)         # wall / ceiling edge
    for x in (140, 460, 760):
        cv2.line(img, (x, 70), (x, 420), (70, 70, 70), 3)         # corners, frames
    cv2.rectangle(img, (520, 120), (720, 330), (235, 230, 220), -1)   # window
    cv2.rectangle(img, (200, 330), (420, 470), (60, 50, 45), -1)      # sofa
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


# ---- lighting --------------------------------------------------------------------------------

def test_lighting_prefers_a_well_exposed_photo():
    room = _room()
    good = score_lighting(_gray(room) / 255.0, CFG.lighting).score
    dark = score_lighting((_gray(room) // 6) / 255.0, CFG.lighting).score
    blown = score_lighting(np.clip(_gray(room).astype(int) * 3, 0, 255) / 255.0, CFG.lighting).score
    assert good > 0.5 and good > dark and good > blown


@pytest.mark.parametrize("value", [0, 255])
def test_a_black_or_white_frame_scores_near_zero_on_lighting(value):
    assert score_lighting(np.full((300, 400), value / 255.0), CFG.lighting).score < 0.2


# ---- depth of field --------------------------------------------------------------------------

def test_the_more_blurred_the_lower_the_depth_of_field_score():
    """(Heavier blur, sigma ~4 here, leaves too few edges to measure: score None, like a blank
    wall - the criterion is then dropped, not scored low. A known limit of the edge measure.)"""
    room = _gray(_room())
    scores = [score_depth_of_field(cv2.GaussianBlur(room, (0, 0), s) if s else room, None,
                                   CFG.depth_of_field).score for s in (0, 1.2, 2, 3)]
    assert None not in scores and scores == sorted(scores, reverse=True) and scores[0] > 0.8


def test_a_flat_frame_has_no_depth_of_field_score():
    r = score_depth_of_field(np.full((300, 400), 128, np.uint8), None, CFG.depth_of_field)
    assert r.score is None and r.note


def test_a_depth_map_with_nan_falls_back_to_thirds_and_says_so():
    room = _gray(_room())
    depth = np.tile(np.linspace(0, 1, room.shape[0])[:, None], (1, room.shape[1]))
    depth[10, 10] = np.nan
    r = score_depth_of_field(room, depth, CFG.depth_of_field)
    plain = score_depth_of_field(room, None, CFG.depth_of_field)
    assert r.score == plain.score and "not finite" in r.note and r.details["bands_used"] >= 2


# ---- coverage --------------------------------------------------------------------------------

def test_a_wider_lens_covers_more_and_no_exif_is_not_guessed():
    g = _gray(_room())
    wide = score_coverage(g, 16.0, CFG.coverage)
    tele = score_coverage(g, 85.0, CFG.coverage)
    none = score_coverage(g, None, CFG.coverage)
    assert wide.score is not None and tele.score is not None and wide.score > tele.score
    assert none.details.get("focal_35mm") is None


def test_a_featureless_frame_has_no_coverage_score():
    assert score_coverage(np.full((300, 400), 128, np.uint8), 24.0, CFG.coverage).score is None


# ---- selector --------------------------------------------------------------------------------

ROOMS = ["front_exterior", "living_room", "living_room", "kitchen", "primary_bedroom", "bathroom"]


def _encoded(i):
    img = _room(i) if i % 2 else _photo()
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_rooms_heroes_and_order_do_not_depend_on_the_input_order():
    inputs = [(f"p{i}", _encoded(i), rt) for i, rt in enumerate(ROOMS)]
    a = score_photos(inputs)
    shuffled = inputs[:]
    random.Random(4).shuffle(shuffled)
    b = score_photos(shuffled)
    assert a.model_dump() == b.model_dump()
    assert len(a.rooms) == 5 and a.sequence[0] == "front_exterior"         # the tour opens outside
    living = next(r for r in a.rooms if r.room_key == "living_room")
    best = max((p for p in a.photos if p.room_key == "living_room"), key=lambda p: (p.total, p.photo_id))
    assert living.hero_photo_id == best.photo_id and len(living.photos) == 2
    assert sorted(r.importance_rank for r in a.rooms) == [1, 2, 3, 4, 5]
