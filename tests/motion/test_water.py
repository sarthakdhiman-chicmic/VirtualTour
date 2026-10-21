"""tour_worker.motion.water: water_guard."""

import numpy as np


def test_thin_distant_water_keeps_most_of_its_surface_moving():
    """Review 2026-10-19: a lake band lost most of its surface to an edge band sized for the near
    water; the held band now narrows with the ripples' own (perspective) reach."""
    from tour_worker.motion.water import water_guard
    img = np.full((800, 1200, 3), (140, 120, 90), np.uint8)
    feather = np.zeros((800, 1200), np.float32)
    feather[400:440, 50:1150] = 1.0                       # a 40 px lake band
    w, _ = water_guard(img, feather, reach=10.0)
    assert (w[400:440] > 0.5).mean() > 0.45
