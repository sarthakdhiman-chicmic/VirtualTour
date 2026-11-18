"""A bed too big for the frame from a tight room's free floor (green house v15's primary bedroom: a
1 m aisle at the bed's foot): it counts as framed when most of its top and its headboard side are in
view - from the foot, not from beside it."""
import numpy as np

from tour_worker.planner3d.moves import EYE, bed_seen, cap_pitch
from tour_worker.planner3d.space import Piece


def _bed():
    # 2.0 m wide (u, with nightstands 3.0) x 2.1 m long (v), its head at +v
    return Piece(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.5, 1.05, 0.6, 6.3, "bed")


def _aim(eye, piece, k=0.35):
    t = np.array([*(piece.centre + k * piece.half_short * piece.v), piece.top])
    return cap_pitch(np.asarray(eye)[None], t[None], 12.0)[0]


def test_seen_from_the_foot_with_its_head():
    b = _bed()
    eye = np.array([0.0, -2.0])                    # 0.95 m off the foot, in the aisle
    share, head = bed_seen(eye, _aim(eye, b), b, (1, 1), EYE)
    assert share >= 0.7 and head


def test_not_from_beside_it_or_facing_away():
    b = _bed()
    eye = np.array([1.9, 0.0])                     # beside it, by the nightstand
    share, head = bed_seen(eye, _aim(eye, b), b, (1, 1), EYE)
    assert share < 0.7 or not head
    eye = np.array([0.0, -1.6])
    away = np.array([0.0, -4.0, 1.0])
    assert bed_seen(eye, away, b, (1, 1), EYE)[0] == 0.0
