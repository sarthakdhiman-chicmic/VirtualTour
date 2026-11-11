"""tour_worker.vision.segment: panoptic map -> regions, scene summary, class maps."""

import numpy as np
import pytest


def test_segments_to_regions_splits_stuff_and_maps_classes():
    from tour_worker.vision.segment import segments_to_regions
    seg = np.zeros((200, 300), np.int32)
    seg[20:180, 10:60] = 1        # curtain panel left
    seg[20:180, 240:290] = 1      # curtain panel right (the same "curtain" segment)
    seg[100:190, 120:180] = 2     # potted plant
    seg[0:3, 0:3] = 3             # a wall: not a class we animate
    info = [{"id": 1, "label_id": 10, "score": 0.99}, {"id": 2, "label_id": 20, "score": 0.95},
            {"id": 3, "label_id": 30, "score": 0.99}]
    from tour_worker.vision.segment import MODELS
    regs = segments_to_regions(seg, info, {10: "curtain", 20: "plant", 30: "wall"}, (300, 200),
                               class_map=MODELS["m2f_ade20k"]["class_map"])
    assert sorted(r["class_name"] for r in regs) == ["curtain", "curtain", "potted_plant"]
    assert all(r["space"] == [300, 200] and r["mask"].shape == (200, 300) for r in regs)
    left = [r for r in regs if r["class_name"] == "curtain" and r["bbox"][0] < 100][0]
    assert left["bbox"] == [10, 20, 60, 180] and left["mask"][100, 30] == 255 and left["mask"][100, 260] == 0
    assert [r["object_id"] for r in regs] == ["seg_000", "seg_001", "seg_002"]


def test_scene_summary_lists_every_class_with_its_place():
    from tour_worker.vision.segment import scene_summary
    seg = np.zeros((100, 200), np.int32)
    seg[60:100, 40:160] = 1                    # a bed, lower middle
    seg[0:30, :] = 2                           # ceiling across the top
    seg[10:50, 170:200] = 3                    # a window, right
    seg[0:1, 0:1] = 4                          # a speck: below 0.2%, left out
    info = [{"id": 1, "label_id": 0, "score": 0.9}, {"id": 2, "label_id": 1, "score": 0.95},
            {"id": 3, "label_id": 2, "score": 0.8}, {"id": 4, "label_id": 3, "score": 0.9}]
    labels = {0: "bed", 1: "ceiling-merged", 2: "window-other", 3: "clock"}
    out = scene_summary(seg, info, labels, (200, 100))
    c = out["classes"]
    assert set(c) == {"bed", "ceiling-merged", "window-other"}
    assert c["bed"]["area"] == pytest.approx(0.24) and c["bed"]["bbox"] == [0.2, 0.6, 0.8, 1.0]
    assert c["bed"]["centroid"] == pytest.approx([0.4975, 0.795], abs=1e-3)
    assert c["window-other"]["centroid"][0] > 0.85


def test_the_class_map_uses_mask2formers_own_label_spelling():
    """Mask2Former's ADE20K labels differ from OneFormer's ("blind", "swimming pool", "palm")."""
    from tour_worker.vision.segment import MODELS, segments_to_regions
    seg = np.zeros((200, 300), np.int32)
    seg[20:180, 10:60] = 1
    seg[100:190, 120:180] = 2
    seg[60:140, 200:280] = 3
    seg[0:60, 80:110] = 4
    info = [{"id": i, "label_id": i, "score": 0.9} for i in (1, 2, 3, 4)]
    labels = {1: "blind", 2: "swimming pool", 3: "fireplace", 4: "palm"}
    regs = segments_to_regions(seg, info, labels, (300, 200), class_map=MODELS["m2f_ade20k"]["class_map"])
    assert sorted(r["class_name"] for r in regs) == ["blinds", "fireplace", "tree", "water"]


def _pool_photo():
    """A pool: aqua water in the middle, a grey stone coping band around it with grout joints."""
    img = np.full((240, 320, 3), (150, 150, 150), np.uint8)          # grey coping (BGR)
    for x in range(40, 280, 12):
        img[40:200, x] = (60, 60, 60)                                # grout joints: strong edges
    img[70:170, 70:250] = (190, 170, 40)                            # aqua water (BGR)
    mask = np.zeros((240, 320), np.uint8)
    mask[40:200, 40:280] = 255                                       # the raw mask takes the coping too
    return img, mask


def test_the_water_refinement_keeps_the_surface_and_freezes_the_coping():
    from tour_worker.vision.refine import refine_water_surface
    img, raw = _pool_photo()
    surface, coping = refine_water_surface(img, raw)
    assert surface[120, 160] == 255 and surface[50, 50] == 0          # water in, coping out
    assert coping[50, 50] == 255 and coping[120, 160] == 0
    assert ((surface > 0) | (coping > 0)).sum() == (raw > 0).sum()   # a split of the raw mask


def _spa_photo():
    """A spa above a pool: a solid dark tile wall between the two waters, a submerged step in the
    pool (dark in patches: ripples break it up) and a green tree reflection on the surface."""
    rng = np.random.default_rng(3)
    img = np.full((300, 400, 3), (190, 170, 40), np.uint8)              # aqua water (BGR)
    img[:20] = (170, 175, 180)                                         # deck, outside the mask
    img[20:28, 40:360] = (205, 210, 215)                               # the spa's pale coping
    img[28:84, 40:360] = (200, 180, 50)                                 # spa water
    img[84:90, 40:360] = (205, 210, 215)                               # the coping on top of the wall
    img[90:140, 40:360] = (75, 60, 50)                                  # the spa's dark tile wall
    img[90:140, 40:360:16] = (110, 110, 110)                            # grout lines
    step = np.zeros((300, 400), bool)
    step[200:260, 120:260] = rng.random((60, 140)) < 0.5                # submerged step, patchy
    img[step] = (80, 70, 55)
    img[160:190, 280:360] = (40, 120, 30)                               # tree reflection (green)
    mask = np.zeros((300, 400), np.uint8)
    mask[20:, 40:360] = 255                                            # spa, wall and pool: one mask
    return img, mask


def test_the_water_refinement_freezes_a_spa_wall_and_keeps_submerged_parts_and_reflections():
    from tour_worker.vision.refine import refine_water_surface
    img, raw = _spa_photo()
    surface, structure = refine_water_surface(img, raw)
    assert (structure[95:135, 60:340] > 0).mean() > 0.95                # the wall above the waterline
    assert (surface[205:255, 125:255] > 0).mean() > 0.95                # the step under it moves
    assert (surface[165:185, 285:355] > 0).all()                        # the reflection moves
    assert (surface[40:78, 60:340] > 0).mean() > 0.95                   # spa water too
    assert (surface[150:290, 60:340] > 0).mean() > 0.95


def test_dark_reflections_and_dark_water_keep_moving():
    """A lake: dark trees on the shore and their dark reflection under them, the sky's reflection
    further down. The reflection mirrors what stands above it, so it is water, not a wall."""
    from tour_worker.vision.refine import refine_water_surface
    img = np.full((300, 400, 3), (200, 150, 110), np.uint8)             # sky-blue water (BGR)
    img[:100] = (40, 80, 50)                                            # dark trees on the shore
    img[100:170] = (60, 75, 55)                                         # their dark reflection
    mask = np.zeros((300, 400), np.uint8)
    mask[100:] = 255
    surface, _ = refine_water_surface(img, mask)
    assert (surface[105:165, 10:390] > 0).mean() > 0.95
    # a dark pool, darker than its few bright highlights, with pale stone above: dark water
    img = np.full((300, 400, 3), (205, 210, 215), np.uint8)             # pale deck
    img[100:] = (70, 60, 45)                                            # dark water
    img[250:270, 100:300] = (220, 200, 120)                             # a bright highlight
    surface, _ = refine_water_surface(img, mask)
    assert (surface[110:240, 20:380] > 0).mean() > 0.95


def test_pale_sunlit_water_at_the_edge_is_not_coping():
    from tour_worker.vision.refine import refine_water_surface
    img = np.full((300, 400, 3), (170, 175, 180), np.uint8)             # grey deck
    img[60:] = (230, 190, 60)                                           # saturated blue water
    img[60:200, 300:] = (235, 225, 190)                                 # pale, sunlit shallow end
    mask = np.zeros((300, 400), np.uint8)
    mask[60:] = 255
    surface, _ = refine_water_surface(img, mask)
    assert (surface[70:190, 310:390] > 0).mean() > 0.95


def _fireplace_photo(lit=True):
    """A stone surround with a dark firebox; a bright orange flame inside it when lit."""
    img = np.full((240, 320, 3), (170, 175, 180), np.uint8)          # pale stone surround
    img[90:200, 100:220] = (20, 20, 25)                              # dark firebox
    if lit:
        img[130:190, 140:180] = (20, 120, 240)                       # orange flame (BGR)
        img[150:180, 150:170] = (200, 240, 255)                      # near-white core
    mask = np.zeros((240, 320), np.uint8)
    mask[60:220, 70:250] = 255                                       # the raw fireplace: surround and all
    return img, mask


def test_a_lit_fireplace_becomes_its_flame_core_and_an_unlit_one_is_dropped():
    from tour_worker.vision.segment import refine_regions
    def region(cls, mask):
        ys, xs = np.nonzero(mask)
        return {"object_id": "seg_000", "class_name": cls, "confidence": 0.9, "space": [320, 240],
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "mask": mask}
    img, raw = _fireplace_photo(lit=True)
    out = refine_regions(img, [region("fireplace", raw)])
    assert [r["class_name"] for r in out] == ["fire"] and out[0]["object_id"] == "fire_000"
    flame = out[0]["mask"]
    ys, xs = np.nonzero(flame)
    assert len(ys) >= 30                                             # the orange tongues move ...
    assert ys.min() >= 90 and ys.max() < 200 and xs.min() >= 100 and xs.max() < 220   # ... inside the firebox
    assert flame[70, 80] == 0 and flame[100, 110] == 0              # the mantel and the cavity do not
    x0, y0, x1, y1 = out[0]["bbox"]
    assert 130 <= x0 and x1 <= 190 and y1 <= 200
    img, raw = _fireplace_photo(lit=False)
    assert refine_regions(img, [region("fireplace", raw)]) == []   # unlit: no flames to animate
    img, raw = _pool_photo()
    water = refine_regions(img, [region("water", raw)])[0]
    assert water["mask"][50, 50] == 0 and water["bbox"][0] >= 60   # the coping left the water region
