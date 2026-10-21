"""tour_worker.motion.foliage: the pot stays still."""

import numpy as np


def test_a_potted_plants_pot_stays_still_and_the_plant_sways_from_its_rim():
    """Review 2026-10-21: the plant mask covers the pot, and the pot swayed with the plant."""
    from tour_worker.motion.foliage import plant_weight
    rng = np.random.default_rng(4)
    img = np.full((400, 300, 3), (200, 205, 210), np.uint8)
    leaves = (np.clip(rng.normal(0, 40, (400, 300)), -80, 80)[..., None] + [40, 140, 60]).clip(0, 255)
    mask = np.zeros((400, 300), np.float32)
    mask[40:260, 60:240] = 1.0                                       # foliage
    mask[260:380, 100:200] = 1.0                                     # the pot below it
    img[40:260, 60:240] = leaves[40:260, 60:240].astype(np.uint8)
    img[260:380, 100:200] = (90, 120, 170)                           # a terracotta pot
    img[80:110, 120:150] = (230, 230, 245)                           # a white flower up in it
    w, pot, pot_top = plant_weight(img, mask, 4.0)
    assert w[262:380, 100:200].max() == 0.0                          # the pot is still
    assert pot_top is not None and 255 <= pot_top <= 265             # sway from the rim
    assert w[60:200, 80:220].mean() > 0.8                            # the plant moves ...
    assert w[85:105, 125:145].mean() > 0.8                           # ... flowers too
