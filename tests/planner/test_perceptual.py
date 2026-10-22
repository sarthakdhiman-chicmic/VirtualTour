"""Perceptual motion: region selection (planner/perceptual.py), plan 1.4 wiring, and the
detections helper (reuse a completed detect job, else run detect, never write anything)."""

import math

import pytest

from tour_worker.planner.config import PlannerConfig
from tour_worker.planner.perceptual import plan_motion_regions
from tour_worker.planner.schemas import CameraParams, Keyframe

CFG = PlannerConfig()
W, H = 1536, 1024  # under detect's 1600 px limit: bboxes are photo pixels


def cam(end_cx=0.5, end_zoom=1.012, start_cx=0.5, start_zoom=1.2):
    return CameraParams(
        type="keyframe_path",
        move="reveal_x",
        keyframes=[
            Keyframe(t=0, cx=start_cx, cy=0.5, zoom=start_zoom),
            Keyframe(t=1, cx=end_cx, cy=0.5, zoom=end_zoom),
        ],
    )


def det(i, cls, bbox, conf=0.9):
    return {
        "object_id": f"obj_{i:03d}",
        "class_name": cls,
        "confidence": conf,
        "bbox": bbox,
        "mask_url": f"s3://b/listings/x/masks/obj_{i:03d}.png",
    }


CURTAIN = det(0, "curtain", [200, 100, 500, 900])
PLANT = det(1, "potted_plant", [1100, 500, 1350, 1000])


def test_maps_classes_to_effects_with_anchor_and_amplitude():
    regions = plan_motion_regions(
        "L", "s01", cam(), [CURTAIN, PLANT, det(2, "sofa", [0, 0, 900, 900])], W, H, 1920, 1080, CFG
    )
    by = {r.label: r for r in regions}
    assert set(by) == {"curtain", "potted_plant"}
    assert by["curtain"].effect == "fabric_sway" and by["curtain"].anchor == "top"
    assert by["potted_plant"].effect == "foliage_sway" and by["potted_plant"].anchor == "bottom"
    # strength is set on screen: 7 px at the widest view (here zoom 1.012 of a 1536 px photo)
    scale = 1920 / (1536 / 1.012)
    assert by["curtain"].amplitude_px * scale == pytest.approx(7.0, abs=0.01)
    assert by["curtain"].bbox == [
        round(200 / W, 5),
        round(100 / H, 5),
        round(500 / W, 5),
        round(900 / H, 5),
    ]


def test_filters_low_confidence_small_and_wrong_pixel_space():
    dets = [
        det(0, "curtain", [200, 100, 500, 900], conf=0.3),  # not confident
        det(1, "curtain", [10, 10, 40, 40]),  # too small
        det(2, "curtain", [200, 100, 2600, 900]),
    ]  # bbox outside the photo
    assert plan_motion_regions("L", "s01", cam(), dets, W, H, 1920, 1080, CFG) == []


def test_off_camera_objects_are_dropped():
    # camera starts and ends tight on the right side: a curtain on the far left is never seen
    c = cam(start_cx=0.75, start_zoom=1.3, end_cx=0.7, end_zoom=1.25)
    assert plan_motion_regions("L", "s01", c, [CURTAIN], W, H, 1920, 1080, CFG) == []


def test_at_most_max_regions_largest_first_and_stable_seeds():
    # 6 since 2026-10-19 (every lake piece and tree of a photo); the cap still holds
    dets = [det(i, "curtain", [100 + 250 * (i % 5), 100 + 300 * (i // 5), 300 + 250 * (i % 5),
                               300 + 300 * (i // 5) + 10 * i]) for i in range(8)]
    a = plan_motion_regions("L", "s01", cam(), dets, W, H, 1920, 1080, CFG)
    b = plan_motion_regions("L", "s01", cam(), dets, W, H, 1920, 1080, CFG)
    assert len(a) == CFG.perceptual_max_regions == 6
    assert [r.id for r in a][:2] == ["obj_007", "obj_006"]                  # largest first
    assert [r.seed for r in a] == [r.seed for r in b]
    assert len({r.seed for r in a}) == len(a)
    other = plan_motion_regions("L", "s02", cam(), dets, W, H, 1920, 1080, CFG)
    assert [r.seed for r in other] != [r.seed for r in a]


# segment_photos (the worker's tour_segment job): tests/domains/virtual_tour/test_segmentation.py


def test_oneformer_regions_use_their_own_pixel_space():
    # a 6240x4160 photo segmented at 2048x1365: bbox normalised by the region's space
    r = det(0, "curtain", [400, 100, 700, 800])
    r["space"] = [2048, 1365]
    reg = plan_motion_regions("L", "s01", cam(), [r], 6240, 4160, 1920, 1080, CFG)[0]
    assert reg.bbox == [
        round(400 / 2048, 5),
        round(100 / 1365, 5),
        round(700 / 2048, 5),
        round(800 / 1365, 5),
    ]
    tree = det(1, "tree", [0, 0, 900, 1000])
    tree["space"] = [2048, 1365]
    water = det(2, "water", [100, 900, 1900, 1360])     # a third of the frame: near, full strength
    water["space"] = [2048, 1365]
    regs = {
        x.label: x
        for x in plan_motion_regions("L", "s01", cam(), [tree, water], 6240, 4160, 1920, 1080, CFG)
    }
    assert regs["tree"].effect == "canopy_flutter" and regs["tree"].anchor == "none"
    assert regs["water"].effect == "calm_water" and regs["water"].anchor == "none"
    # strength follows the PHOTO width (the reviewed prototype), capped: 0.0063 x 6240 -> 20 max
    assert regs["water"].amplitude_px == 20.0
    # trees: 10 px on screen (6240 px photo at zoom 1.012 -> ~0.31 output px per source px)
    assert regs["tree"].amplitude_px * 1920 / (6240 / 1.012) == pytest.approx(
        CFG.perceptual_screen_target_px["canopy_flutter"], abs=0.01)
    assert regs["tree"].freq_hz == 1.1


def test_water_strength_follows_its_apparent_distance():
    # near water (a pool filling the lower frame) moves at full strength, whatever its width
    narrow, wide = det(0, "water", [100, 600, 400, 1000]), det(1, "water", [100, 600, 1400, 1000])
    amps = [
        plan_motion_regions("L", "s01", cam(), [d], 1536, 1024, 1920, 1080, CFG)[0].amplitude_px
        for d in (narrow, wide)
    ]
    assert amps[0] == amps[1] == round(0.0063 * 1536, 3)
    # a mid band moves less than the near pool (boosted a little for its distance); a thin band
    # near the horizon (a far lake) still moves, at least as much as reads on screen (review
    # 2026-10-19: distant water was left still)
    mid = plan_motion_regions(
        "L", "s01", cam(), [det(0, "water", [100, 600, 1400, 800])], 1536, 1024, 1920, 1080, CFG
    )[0]
    h = 200 / 1024
    boost = 1 + CFG.perceptual_far_boost * (CFG.perceptual_far_height - h) / (
        0.7 * CFG.perceptual_far_height)
    assert mid.amplitude_px == pytest.approx(0.0063 * 1536 * min(1.0, boost * h / 0.25), abs=0.01)
    assert mid.amplitude_px <= amps[0]                     # never more than the near water
    far = det(0, "water", [100, 400, 1400, 450])
    (lake,) = plan_motion_regions("L", "s01", cam(), [far], 1536, 1024, 1920, 1080, CFG)
    assert 0 < lake.amplitude_px < mid.amplitude_px


def test_bboxes_of_large_photos_are_in_detects_1600px_space():
    # a 6240x4160 photo is detected at 1600x1067: a bbox there must map to the same relative spot
    big = det(0, "curtain", [400, 100, 700, 800])
    r = plan_motion_regions("L", "s01", cam(), [big], 6240, 4160, 1920, 1080, CFG)[0]
    k = 1600 / 6240
    assert r.bbox == [
        round(400 / (6240 * k), 5),
        round(100 / (4160 * k), 5),
        round(700 / (6240 * k), 5),
        round(800 / (4160 * k), 5),
    ]
    # a 6240 px photo shows ~0.3 output px per source px: ~22 source px for 7 px on screen
    assert r.amplitude_px * 1920 / (6240 / 1.012) == pytest.approx(7.0, abs=0.01)
    small = plan_motion_regions("L", "s01", cam(), [big], 1536, 1024, 1920, 1080, CFG)[0]
    assert small.bbox == [
        round(400 / 1536, 5),
        round(100 / 1024, 5),
        round(700 / 1536, 5),
        round(800 / 1024, 5),
    ]


def test_screen_strength_is_capped_by_the_object_and_tiny_ones_are_dropped():
    # a narrow curtain (60 px): capped at 8% of its width; 4.8 px source = ~6 px on screen, kept
    narrow = det(0, "curtain", [600, 100, 660, 900])
    r = plan_motion_regions("L", "s01", cam(), [narrow], W, H, 1920, 1080, CFG)
    assert r and r[0].amplitude_px == pytest.approx(0.08 * 60)
    # narrower still (45 px: ~4.6 px peak): its sway would read under 2 px, so it is not planned
    thinner = det(0, "curtain", [600, 100, 645, 900])
    assert plan_motion_regions("L", "s01", cam(), [thinner], W, H, 1920, 1080, CFG) == []


def test_curtains_and_plants_get_opposite_sway_phases():
    regs = plan_motion_regions("L", "s01", cam(), [CURTAIN, PLANT], W, H, 1920, 1080, CFG)
    by = {r.label: r for r in regs}
    diff = (by["potted_plant"].phase - by["curtain"].phase) % (2 * math.pi)
    assert diff == pytest.approx(math.pi, abs=1e-4)
    assert by["curtain"].freq_hz == by["potted_plant"].freq_hz      # stays opposite all shot


def test_lit_fires_get_the_fire_effect():
    fire = det(0, "fire", [700, 600, 820, 700])
    reg = plan_motion_regions(
        "L", "s01", cam(), [fire], W, H, 1920, 1080, CFG, room_type="living_room"
    )
    assert reg and reg[0].effect == "fire_flicker" and reg[0].anchor == "bottom"


def test_fire_strength_follows_the_flames_own_height():
    """User 2026-10-21: scale the flames to their own size (a small insert gently, a big log
    fire more), not one on-screen strength for every fire."""
    small = det(0, "fire", [700, 650, 760, 690])                 # 40 px high flames
    big = det(1, "fire", [600, 550, 900, 700])                   # 150 px high flames
    a = plan_motion_regions("L", "s01", cam(), [small], W, H, 1920, 1080, CFG)[0].amplitude_px
    b = plan_motion_regions("L", "s01", cam(), [big], W, H, 1920, 1080, CFG)[0].amplitude_px
    frac = CFG.perceptual_amplitude["fire_flicker"][0]
    assert a == pytest.approx(frac * 40, abs=0.05) and b == pytest.approx(frac * 150, abs=0.05)


def test_ade20k_plants_sway_indoors_and_stay_still_outdoors():
    plant = det(0, "plant", [1100, 500, 1350, 1000])
    def plan(room):
        return plan_motion_regions(
            "L", "s01", cam(), [plant], W, H, 1920, 1080, CFG, room_type=room
        )

    inside, outside = plan("living_room"), plan("patio")
    assert inside and inside[0].effect == "foliage_sway"
    assert outside == []                                   # shrubs and hedges stay still
    shrub = det(1, "plant_outdoor", [1100, 500, 1350, 1000])  # sky in the photo (worker)
    assert plan_motion_regions("L", "s01", cam(), [shrub], W, H, 1920, 1080, CFG) == []


def test_small_and_partly_framed_water_still_moves():
    """Review 2026-10-19: lake pieces of 0.4% of the photo, or 41% in frame, were left still."""
    thin = det(0, "water", [600, 500, 760, 540])                 # 0.4% of a 1536x1024 photo
    assert plan_motion_regions("L", "s01", cam(), [thin], 1536, 1024, 1920, 1080, CFG)
    tiny_curtain = det(1, "curtain", [600, 500, 760, 540])
    assert not plan_motion_regions("L", "s01", cam(), [tiny_curtain], 1536, 1024, 1920, 1080, CFG)


def test_fire_regions_from_the_segment_job_plan_as_fire_flicker():
    """A lit fire as tour_segment's fireplace pass returns it (moved from the backend's
    test_segmentation: the planner half)."""
    fire = {"object_id": "fire_000", "class_name": "fire", "confidence": 1.0,
            "bbox": [900, 700, 1150, 860], "space": [2048, 1365],
            "mask_url": "s3://b/p/img_1/fire_000.png"}
    cam_ = CameraParams(type="keyframe_path", move="reveal_x", keyframes=[
        Keyframe(t=0, cx=0.5, cy=0.5, zoom=1.2), Keyframe(t=1, cx=0.5, cy=0.5, zoom=1.012)])
    regions = plan_motion_regions("L", "s01", cam_, [fire], 2048, 1365, 1920, 1080,
                                  PlannerConfig(), room_type="living_room")
    assert [(r.label, r.effect, r.mask_uri) for r in regions] == [
        ("fire", "fire_flicker", fire["mask_url"])]
