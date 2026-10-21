import json
import pytest

from modules.plan_io import parse_plan
from run_tour import run_tour
from modules.storage import LocalMapStorage

def build_valid_plan_dict(easing="ease_in_out", plan_version="1.2", preset="push_in_gentle"):
    import hashlib
    data = {
        "plan_version": plan_version,
        "listing_id": "test_easing",
        "mode": "cinematic",
        "preset": "cinematic_tour",
        "output": {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "codec": "h264"
        },
        "total_duration_s": 5.0,
        "shots": [
            {
                "shot_id": "shot_001",
                "order": 1,
                "asset_id": "asset_1",
                "image_uri": "s3://dummy/img1.jpg",
                "source": {"width": 1920, "height": 1080},
                "room_key": "living_room",
                "room_type": "living_room",
                "narrative_stage": "intro",
                "role": "hero",
                "duration_s": 5.0,
                "camera": {
                    "type": "ken_burns",
                    "easing": easing,
                    "preset": preset,
                    "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                    "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.06}
                },
                "transition_in": {"type": "cut", "duration_s": 0.0}
            }
        ],
        "warnings": [],
        "provenance": {
            "scoring_config_version": "1.0",
            "planner_config_version": "1.0",
            "plan_hash": ""
        }
    }
    copy_d = dict(data)
    del copy_d["provenance"]["plan_hash"]
    c_json = json.dumps(copy_d, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
    data["provenance"]["plan_hash"] = hashlib.sha256(c_json.encode('utf-8')).hexdigest()
    return data

# =========================================================================
# 1. EASING UNIT TESTS
# =========================================================================
def test_easing_v12_accepts_all_easings():
    for easing in ["linear", "ease_in", "ease_out", "ease_in_out"]:
        plan_dict = build_valid_plan_dict(easing=easing, plan_version="1.2")
        plan = parse_plan(plan_dict)
        assert plan.shots[0].camera.easing == easing
        assert plan.shots[0].camera.preset == "push_in_gentle"

def test_easing_v10_v11_rejects_non_default():
    for ver in ["1.0", "1.1"]:
        # default ease_in_out should pass
        plan_dict = build_valid_plan_dict(easing="ease_in_out", plan_version=ver)
        plan = parse_plan(plan_dict)
        assert plan.shots[0].camera.easing == "ease_in_out"

        # non-default easings should be rejected with invalid_plan
        for non_default in ["linear", "ease_in", "ease_out"]:
            bad_dict = build_valid_plan_dict(easing=non_default, plan_version=ver)
            with pytest.raises(ValueError, match="invalid_plan: non-default easing"):
                parse_plan(bad_dict)

def test_easing_unknown_rejected_as_unsupported_easing():
    bad_dict = build_valid_plan_dict(easing="cubic_bezier", plan_version="1.2")
    with pytest.raises(ValueError, match="unsupported_easing"):
        parse_plan(bad_dict)

    # Test via run_tour interface
    storage = LocalMapStorage({})
    res = run_tour({"plan": bad_dict}, storage, "/tmp")
    assert res["status"] == "failed"
    assert res["error"]["code"] == "unsupported_easing"
