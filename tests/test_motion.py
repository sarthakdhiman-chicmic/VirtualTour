import os
import json
import pytest
import numpy as np
import cv2

from modules.plan_io import Plan, parse_plan
from run_tour import run_tour
from modules.storage import LocalMapStorage
from tools.motion_analyzer import analyze_shot_frames
from tools.motion_presets import preset_to_camera, PRESETS
from tools.golden_vectors import generate_golden_vectors
from tools.motion_lab import run_lab_single, check_black_edges

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

# =========================================================================
# 2. ANALYZER GROUND-TRUTH TESTS
# =========================================================================
def create_synthetic_frames(zoom_total=0.0, shift_x=0.0, shift_y=0.0, easing="linear", num_frames=122):
    np.random.seed(123)
    canvas_w, canvas_h = 2400, 1600
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y_grad = np.linspace(40, 200, canvas_h)[:, None]
    x_grad = np.linspace(40, 200, canvas_w)[None, :]
    canvas = ((y_grad + x_grad) / 2).astype(np.uint8)

    for _ in range(400):
        rx = np.random.randint(0, canvas_w - 100)
        ry = np.random.randint(0, canvas_h - 100)
        rw = np.random.randint(20, 150)
        rh = np.random.randint(20, 150)
        color = np.random.randint(0, 255)
        canvas[ry:ry+rh, rx:rx+rw] = color

    frames = []
    out_w, out_h = 332, 187
    for f in range(num_frames):
        t = f / max(1, num_frames - 1)
        t_prime = t if easing == "linear" else (1.0 - (1.0 - t)**2)
        z = 1.0 + zoom_total * t_prime
        cx = 0.5 + shift_x * t_prime
        cy = 0.5 + shift_y * t_prime

        base_w = 0.6 / z
        base_h = 0.6 * (out_h / out_w) / z
        x0 = int(round((cx - base_w / 2) * canvas_w))
        y0 = int(round((cy - base_h / 2) * canvas_h))
        x1 = int(round((cx + base_w / 2) * canvas_w))
        y1 = int(round((cy + base_h / 2) * canvas_h))

        crop = canvas[y0:y1, x0:x1]
        resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    return frames

def test_analyzer_pure_zoom():
    frames = create_synthetic_frames(zoom_total=0.10, shift_x=0.0, shift_y=0.0, easing="linear")
    res = analyze_shot_frames(frames)
    assert abs(res["zoom_total"] - 0.10) <= 0.02
    assert abs(res["shift_x"]) <= 0.015
    assert abs(res["shift_y"]) <= 0.015
    prof = res["cum_speed"]
    assert abs(prof[0] - 0.25) <= 0.07
    assert abs(prof[1] - 0.50) <= 0.07
    assert abs(prof[2] - 0.75) <= 0.07

def test_analyzer_pure_slide():
    frames = create_synthetic_frames(zoom_total=0.0, shift_x=-0.05, shift_y=0.0, easing="linear")
    res = analyze_shot_frames(frames)
    assert abs(res["zoom_total"]) <= 0.02
    assert abs(res["shift_y"]) <= 0.015

def test_analyzer_ease_out_profile():
    frames = create_synthetic_frames(zoom_total=0.15, shift_x=0.0, shift_y=0.0, easing="ease_out")
    res = analyze_shot_frames(frames)
    prof = res["cum_speed"]
    target_prof = [0.438, 0.750, 0.938]
    assert abs(prof[0] - target_prof[0]) <= 0.07
    assert abs(prof[1] - target_prof[1]) <= 0.07
    assert abs(prof[2] - target_prof[2]) <= 0.07

# =========================================================================
# 3. PRESET FALLBACKS & 9:16 DOWNGRADE
# =========================================================================
def test_preset_fallbacks_and_downgrade():
    # 9:16 with max_zoom=1.0 forces horizontal shift presets to downgrade
    cam, fb = preset_to_camera("push_slide", 1080, 1920, 1920, 1080, max_zoom=1.0)
    assert fb == "downgrade_push_in_gentle"
    assert cam["preset"] == "push_in_gentle"

    # scale_0.75
    cam_75, fb_75 = preset_to_camera("push_slide", 1080, 1920, 1920, 1080, max_zoom=1.10)
    assert fb_75 == "scale_0.75"

    # scale_0.5
    cam_50, fb_50 = preset_to_camera("push_slide", 1080, 1920, 1920, 1080, max_zoom=1.06)
    assert fb_50 == "scale_0.5"

    # static
    cam_stat, fb_stat = preset_to_camera("slide", 1080, 1920, 1920, 1080, max_zoom=0.99)
    assert fb_stat == "static"
    assert cam_stat["preset"] == "static"

# =========================================================================
# 4. GOLDEN VECTORS DETERMINISM
# =========================================================================
def test_golden_vectors_determinism():
    v1 = generate_golden_vectors()
    v2 = generate_golden_vectors()

    assert len(v1) <= 48
    assert json.dumps(v1, sort_keys=True) == json.dumps(v2, sort_keys=True)

    fallbacks = {item["fallback"] for item in v1}
    assert {"none", "scale_0.75", "scale_0.5", "downgrade_push_in_gentle", "static"}.issubset(fallbacks)

# =========================================================================
# 5. PRESETS ON TEXTURED IMAGES & BLACK-EDGE CHECKS
# =========================================================================
def _generate_textured_image(width: int, height: int, path: str):
    np.random.seed(42)
    # Bright textured canvas (luminance > 40 everywhere)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 180
    for _ in range(300):
        rx = np.random.randint(0, max(1, width - 100))
        ry = np.random.randint(0, max(1, height - 100))
        rw = np.random.randint(30, 150)
        rh = np.random.randint(30, 150)
        color = [np.random.randint(50, 240) for _ in range(3)]
        cv2.rectangle(canvas, (rx, ry), (min(width, rx + rw), min(height, ry + rh)), color, -1)
    cv2.imwrite(path, canvas)

def test_render_textured_aspects_and_black_edge(tmp_path):
    aspects = {
        "16:9": (1920, 1080),
        "3:2": (1500, 1000),
        "4:3": (1600, 1200)
    }
    
    # Test push_in_gentle across 16:9, 3:2, 4:3
    for aspect_name, (w, h) in aspects.items():
        img_file = str(tmp_path / f"textured_{aspect_name.replace(':', '_')}.jpg")
        _generate_textured_image(w, h, img_file)
        
        res = run_lab_single(img_file, "push_in_gentle", scale=0.25)
        assert res["checks"]["black_edge_passed"], f"Black edge check failed on {aspect_name}"
        assert res["checks"]["min_luminance"] > 10.0
        assert res["checks"]["zoom_ok"], f"Zoom error out of tolerance on {aspect_name}: {res['errors']['zoom']}"
        assert res["checks"]["shift_x_ok"], f"Shift X error out of tolerance on {aspect_name}: {res['errors']['shift_x']}"
        assert res["checks"]["shift_y_ok"], f"Shift Y error out of tolerance on {aspect_name}: {res['errors']['shift_y']}"
        assert res["checks"]["profile_ok"], f"Profile error out of tolerance on {aspect_name}: {res['errors']['profile_max']}"

def test_render_easing_zoom_profiles(tmp_path):
    # Pure zoom test: linear (push_in_gentle) vs ease_out (push_in_strong)
    img_file = str(tmp_path / "textured_16_9.jpg")
    _generate_textured_image(1920, 1080, img_file)

    # 1. Linear (push_in_gentle)
    res_linear = run_lab_single(img_file, "push_in_gentle", scale=0.25)
    assert res_linear["checks"]["profile_ok"], f"Linear profile max error {res_linear['errors']['profile_max']} > 0.07"

    # 2. Ease Out (push_in_strong)
    res_ease_out = run_lab_single(img_file, "push_in_strong", scale=0.25)
    assert res_ease_out["checks"]["profile_ok"], f"Ease out profile max error {res_ease_out['errors']['profile_max']} > 0.07"
    assert res_ease_out["checks"]["zoom_ok"], f"Ease out zoom error {res_ease_out['errors']['zoom']} > 0.03"
    assert res_ease_out["checks"]["black_edge_passed"]
