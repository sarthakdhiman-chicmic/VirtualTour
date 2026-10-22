"""scoring/motion.py: saliency targets and the coverage matrix (room size x spread band)."""

import numpy as np
import pytest

from tour_worker.scoring.config import MotionConfig
from tour_worker.scoring.motion import coverage_class, motion_signals, room_size

CFG = MotionConfig()


@pytest.mark.parametrize(
    "room, size",
    [
        ("living_room", "large"),
        ("Kitchen", "large"),
        ("aerial", "large"),
        ("dining_room", "medium"),
        ("primary_bedroom", "medium"),
        (None, "medium"),
        ("wine_cellar", "medium"),
        ("bedroom", "small"),
        ("powder_room", "small"),
        ("hallway", "small"),
    ],
)
def test_room_size(room, size):
    assert room_size(room, CFG) == size


@pytest.mark.parametrize(
    "spread, size, cls",
    [
        (0.30, "large", "medium"),
        (0.40, "large", "wide"),
        (0.45, "large", "wide"),
        (0.30, "medium", "tight"),
        (0.40, "medium", "medium"),
        (0.45, "medium", "wide"),
        (0.30, "small", "tight"),
        (0.40, "small", "tight"),
        (0.45, "small", "medium"),
    ],
)
def test_matrix(spread, size, cls):
    assert coverage_class(spread, size, None, CFG)[0] == cls


def test_exif_field_of_view_bumps_one_class():
    assert coverage_class(0.40, "medium", 0.9, CFG)[0] == "wide"  # very wide lens
    assert coverage_class(0.40, "medium", 0.1, CFG)[0] == "tight"  # long lens
    assert coverage_class(0.45, "large", 0.9, CFG)[0] == "wide"  # already the top
    assert coverage_class(0.40, "medium", 0.5, CFG)[0] == "medium"  # in between: no change


def test_signals_on_a_photo_like_image():
    rng = np.random.default_rng(1)
    img = np.full((640, 960), 0.5, np.float32)
    img[300:420, 600:760] = 0.95  # one bright subject, right of centre
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    m = motion_signals(np.clip(img, 0, 1), "living_room", None, CFG)
    assert m is not None
    assert m.side == "right" and m.peak_x > 0.5
    assert 0.0 < m.spread < 1.0 and m.room_size == "large"
    assert len(m.grid) == round(CFG.grid_w * 480 * 640 / 960 / 480) and len(m.grid[0]) == CFG.grid_w
    assert max(max(r) for r in m.grid) == 1.0


def test_flat_frame_has_no_signals():
    assert motion_signals(np.full((400, 600), 0.5, np.float32), "living_room", None, CFG) is None
