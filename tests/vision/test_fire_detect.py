"""tour_worker.vision.fire_detect: lit fires inside fireplaces."""

import cv2
import numpy as np

from tests.perceptual_fixtures import _fireplace_scene, _with_firebox


def test_lit_fire_is_found_inside_the_fireplace_and_warm_stone_is_not():
    from tour_worker.vision.fire_detect import flame_mask
    img, fp = _fireplace_scene(lit=True)
    fm = flame_mask(img, fp) > 0
    assert fm[225:290, 240:360].mean() > 0.15                             # the flames
    assert fm[80:170].sum() == 0 and fm[:, :230].sum() == 0               # not the stone surround
    img, fp = _fireplace_scene(lit=False)
    assert flame_mask(img, fp).sum() == 0                                 # unlit: nothing


def test_fires_behind_glass_are_found_with_the_firebox():
    """Downing's gas fires: the glass lifts the blacks around the flames, so the dark-ring test
    failed them; with the firebox known it decides instead."""
    img, fp = _fireplace_scene(lit=True)
    img[180:320, 230:370] = np.clip(img[180:320, 230:370].astype(int) + 30, 0, 255).astype(np.uint8)  # glass haze
    assert _with_firebox(img, fp)[225:290, 240:360].mean() > 0.15


def test_unlit_logs_marble_and_a_throw_in_front_are_not_fire():
    """S3 review: unlit orange logs, an orange surround, a throw on the table in front."""
    img, fp = _fireplace_scene(lit=False)
    for x in range(245, 355, 22):                                    # orange-brown logs, no glow
        cv2.ellipse(img, (x + 8, 290), (12, 8), 0, 0, 360, (40, 110, 175), -1)
    img[90:110, 190:410] = (60, 140, 220)                            # orange marble band on top
    img[322:340, 250:350] = (50, 150, 240)                           # a throw just in front
    fp[322:340, 250:350] = True
    assert _with_firebox(img, fp).sum() == 0


def test_firelight_on_the_lintel_is_not_a_flame():
    img, fp = _fireplace_scene(lit=True)
    img[184:190, 235:365] = (60, 170, 250)                           # a glowing strip under the lintel
    fm = _with_firebox(img, fp)
    assert fm[225:290, 240:360].mean() > 0.15 and fm[182:192].sum() == 0
