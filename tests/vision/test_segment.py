"""tour_worker.vision.segment: panoptic map -> regions, scene summary, class maps."""

import numpy as np
import pytest


def test_segments_to_regions_splits_stuff_and_maps_classes():
    from tour_worker.vision.segment import segments_to_regions
    seg = np.zeros((200, 300), np.int32)
    seg[20:180, 10:60] = 1        # curtain panel left
    seg[20:180, 240:290] = 1      # curtain panel right (same COCO "curtain" segment)
    seg[100:190, 120:180] = 2     # potted plant
    seg[0:3, 0:3] = 3             # a wall: not a class we animate
    info = [{"id": 1, "label_id": 10, "score": 0.99}, {"id": 2, "label_id": 20, "score": 0.95},
            {"id": 3, "label_id": 30, "score": 0.99}]
    from tour_worker.vision.segment import MODELS
    regs = segments_to_regions(seg, info, {10: "curtain", 20: "potted plant", 30: "wall-other-merged"}, (300, 200),
                               class_map=MODELS["coco"]["class_map"])
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


def test_ade20k_class_map_covers_our_classes_and_fireplaces():
    from tour_worker.vision.segment import MODELS, segments_to_regions
    seg = np.zeros((200, 300), np.int32)
    seg[20:180, 10:60] = 1
    seg[100:190, 120:180] = 2
    seg[60:140, 200:280] = 3
    info = [{"id": 1, "label_id": 18, "score": 0.9}, {"id": 2, "label_id": 17, "score": 0.9},
            {"id": 3, "label_id": 49, "score": 0.9}]
    labels = {18: "curtain", 17: "plant", 49: "fireplace"}
    regs = segments_to_regions(seg, info, labels, (300, 200), class_map=MODELS["ade20k"]["class_map"])
    assert sorted(r["class_name"] for r in regs) == ["curtain", "fireplace", "potted_plant"]
