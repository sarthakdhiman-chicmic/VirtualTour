"""scoring -> build_shot_plan with the cinematic engine: allocation limits, plan 1.3, hash."""

import cv2
import numpy as np
import pytest

from tour_worker.planner.config import PlannerConfig
from tour_worker.planner.planner import _compute_plan_hash, build_shot_plan
from tour_worker.scoring.pipeline import score_photos

ROOMS = [
    "exterior",
    "living_room",
    "living_room",
    "kitchen",
    "kitchen",
    "dining_room",
    "bedroom",
    "bedroom",
    "primary_bathroom",
    None,
]


def _photo(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((640, 960, 3)) * 255).astype(np.uint8), (0, 0), 3)
    for _ in range(6):  # some structure: lines and blocks
        x, y = int(rng.integers(0, 900)), int(rng.integers(0, 600))
        cv2.rectangle(
            img, (x, y), (x + 60, y + 40), tuple(int(v) for v in rng.integers(0, 255, 3)), -1
        )
        cv2.line(
            img, (int(rng.integers(0, 960)), 0), (int(rng.integers(0, 960)), 639), (20, 20, 20), 3
        )
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture(scope="module")
def report():
    return score_photos([(f"p{i:02d}", _photo(i), rt) for i, rt in enumerate(ROOMS)])


def _assets():
    return {
        f"p{i:02d}": {"uri": f"p{i:02d}.jpg", "width": 960, "height": 640}
        for i in range(len(ROOMS))
    }


def test_cinematic_plan(report):
    cfg = PlannerConfig()
    plan = build_shot_plan(report, "t", _assets(), cfg=cfg)
    assert plan.plan_version == "1.3" and plan.mode == "cinematic"
    assert all(
        cfg.shot_min_s - 1e-6 <= s.duration_s <= cfg.shot_max_s + 1e-6 or s is plan.shots[-1]
        for s in plan.shots
    )
    assert all(s.camera.type in ("keyframe_path", "compound") and s.camera.move for s in plan.shots)
    d = plan.model_dump(mode="json")
    assert d["provenance"]["plan_hash"] == _compute_plan_hash(d)
    assert all(p.motion is not None for p in report.photos)


def test_presets_engine_still_available(report):
    plan = build_shot_plan(report, "t", _assets(), cfg=PlannerConfig(motion_engine="presets"))
    assert plan.plan_version == "1.2" and all(s.camera.type == "ken_burns" for s in plan.shots)


def test_infeasible_target_never_breaks_the_shot_cap(report):
    cfg = PlannerConfig(target_total_s=200.0, max_total_s=300.0)
    plan = build_shot_plan(report, "t", _assets(), cfg=cfg)
    assert max(s.duration_s for s in plan.shots) <= max(cfg.shot_max_s, cfg.final_shot_min_s) + 1e-6


def test_perceptual_regions_make_plan_14(report):
    curtain = {
        "object_id": "obj_000",
        "class_name": "curtain",
        "confidence": 0.9,
        "bbox": [380, 150, 580, 500],
        "mask_url": "s3://b/listings/x/masks/obj_000.png",
    }
    dets = {f"p{i:02d}": [curtain] for i in range(len(ROOMS))}
    dets["p09"] = None  # detect failed for one photo
    plan = build_shot_plan(report, "t", _assets(), cfg=PlannerConfig(), detections_by_photo=dets)
    assert plan.plan_version == "1.4"
    with_regions = [s for s in plan.shots if s.motion_regions]
    assert with_regions  # exteriors are included now (trees, water)
    assert all(r.effect == "fabric_sway" for s in with_regions for r in s.motion_regions)
    d = plan.model_dump(mode="json")
    assert d["provenance"]["plan_hash"] == _compute_plan_hash(d)
    if any(s.asset_id == "p09" for s in plan.shots):
        assert any(w.startswith("perceptual_unavailable:") for w in plan.warnings)
    off = build_shot_plan(
        report, "t", _assets(), cfg=PlannerConfig(perceptual_motion=False), detections_by_photo=dets
    )
    assert off.plan_version == "1.3" and not any(s.motion_regions for s in off.shots)


def test_single_photo_rooms_hero_only_when_top_ranked(report):
    """A room with one photo is not automatically a hero: only the top hero_room_count rooms are,
    the rest are normal-length shots (review 2026-10-19: mixed shot lengths)."""
    from tour_worker.planner.allocate import allocate_shots

    cfg = PlannerConfig()
    top = report.config.ranking.hero_room_count
    shots, _ = allocate_shots(report, cfg)
    lead = {s.room_key: s for s in shots if s.is_lead}
    for room in report.rooms:
        s = lead.get(room.room_key)
        if s is None:
            continue
        expect_hero = len(room.photos) > 1 or room.importance_rank <= top
        assert s.is_hero is expect_hero
        assert s.role == ("hero" if expect_hero else "detail")
    singles = [r for r in report.rooms if len(r.photos) == 1]
    assert any(not lead[r.room_key].is_hero for r in singles if r.room_key in lead)
