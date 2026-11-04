"""tour_worker.scoring.scorer: criteria combined into one score; a photo that cannot be judged
never outranks a real one."""
import cv2
import numpy as np
import pytest

from tour_worker.scoring.config import ScoringConfig
from tour_worker.scoring.imaging import photo_from_bytes
from tour_worker.scoring.scorer import score_photo
from tests.perceptual_fixtures import _photo


def _score(img):
    ok, data = cv2.imencode(".jpg", img)
    return score_photo(photo_from_bytes("p", data.tobytes(), room_type="living_room"), ScoringConfig())


@pytest.mark.parametrize("img", [
    (np.full((800, 1200, 3), 128) + np.random.default_rng(0).normal(0, 2, (800, 1200, 3))).clip(0, 255).astype(np.uint8),
    np.full((4, 4, 3), 128, np.uint8),
    np.zeros((800, 1200, 3), np.uint8),
], ids=["flat_grey", "tiny", "black"])
def test_a_photo_that_cannot_be_judged_is_low_and_flagged(img):
    s = _score(img)
    assert len(s.missing_criteria) >= 2
    assert s.total <= ScoringConfig().unmeasurable_cap and {"unmeasurable", "low_score"} <= set(s.flags)
    real = _score(_photo())
    assert real.total > s.total and not real.missing_criteria
