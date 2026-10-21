"""tour_worker.motion.masks: mask loading, feathering and the belongs-to-its-bbox check."""
import os
import tempfile

import cv2
import numpy as np
import pytest

from tests.perceptual_fixtures import CURTAIN_BBOX, SH, SW, _mask
from tour_worker.motion.masks import load_feather


def test_feather_is_zero_at_the_mask_edge_and_full_inside():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.png")
        cv2.imwrite(p, _mask())
        f, _ = load_feather(p, SW, SH, (CURTAIN_BBOX[0] * SW, CURTAIN_BBOX[1] * SH,
                                        CURTAIN_BBOX[2] * SW, CURTAIN_BBOX[3] * SH))
    m = _mask() > 0
    assert f[~m].max() == 0.0
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert f[edge].max() < 0.1
    assert f[m].max() == pytest.approx(1.0)


def test_mask_of_a_different_aspect_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.png")
        cv2.imwrite(p, np.full((500, 500), 255, np.uint8))
        assert load_feather(p, SW, SH, (0, 0, SW, SH)) == (None, "aspect")


def test_a_mask_that_belongs_to_another_object_is_rejected():
    """A curtain detection whose stored mask is a different object (seen in production data)."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.png")
        other = np.zeros((SH, SW), np.uint8)
        other[500:600, 700:900] = 255                      # far from the curtain bbox
        cv2.imwrite(p, other)
        box = (CURTAIN_BBOX[0] * SW, CURTAIN_BBOX[1] * SH, CURTAIN_BBOX[2] * SW, CURTAIN_BBOX[3] * SH)
        assert load_feather(p, SW, SH, box) == (None, "mismatch")
        # a correct mask with a few stray pixels elsewhere is kept, and the strays never move
        good = _mask()
        good[600:604, 900:904] = 255
        cv2.imwrite(p, good)
        f, reason = load_feather(p, SW, SH, box)
        assert reason == "" and f[600:604, 900:904].max() == 0.0
