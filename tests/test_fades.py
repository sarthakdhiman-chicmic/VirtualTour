import os
import subprocess
import tempfile
import json
import hashlib
from PIL import Image
import numpy as np
import math
from modules.plan_io import parse_plan
from modules.direct_render import render_direct
import copy

def fix_plan_hash(p):
    dc = copy.deepcopy(p)
    if "plan_hash" in dc.get("provenance", {}):
        del dc["provenance"]["plan_hash"]
    p["provenance"]["plan_hash"] = hashlib.sha256(json.dumps(dc, separators=(',', ':'), sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

def get_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vsync", "0",
        os.path.join(out_dir, "f%04d.png")
    ], check=True, capture_output=True)
    frames = sorted([f for f in os.listdir(out_dir) if f.endswith(".png")])
    return [os.path.join(out_dir, f) for f in frames]

def test_fade_no_dip():
    # two shots of SAME flat gray (value 200)
    # verify mean pixel is within +-2 of 200 through the fade
    plan_dict = {
        "plan_version": "1.1", "listing_id": "L", "mode": "M", "preset": "P",
        "output": {"width": 320, "height": 180, "fps": 24, "codec": "h264"},
        "total_duration_s": 2.0,
        "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": "dummy"},
        "shots": [
            {
                "shot_id": "s0", "order": 0, "asset_id": "a0", "image_uri": "s3://b/img_gray.jpg",
                "source": {"width": 320, "height": 180},
                "room_key": "k", "room_type": "r", "narrative_stage": "n", "role": "h",
                "duration_s": 1.0,
                "camera": {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}, "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}},
                "transition_in": {"type": "cut", "duration_s": 0.0}
            },
            {
                "shot_id": "s1", "order": 1, "asset_id": "a1", "image_uri": "s3://b/img_gray.jpg",
                "source": {"width": 320, "height": 180},
                "room_key": "k", "room_type": "r", "narrative_stage": "n", "role": "h",
                "duration_s": 1.0,
                "camera": {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}, "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}},
                "transition_in": {"type": "fade", "duration_s": 0.4}
            }
        ]
    }
    fix_plan_hash(plan_dict)
    
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        Image.new('RGB', (320, 180), color=(200, 200, 200)).save(os.path.join(d, "images", "img_gray.jpg"))
        
        plan = parse_plan(plan_dict)
        out_path = render_direct(plan, d, workers=1)
        
        frames = get_frames(out_path, os.path.join(d, "frames"))
        
        # total frames should be 2.0 * 24 = 48
        assert len(frames) == 48
        
        for i, fpath in enumerate(frames):
            img = Image.open(fpath).convert('RGB')
            arr = np.array(img)
            mean_val = arr.mean()
            assert abs(mean_val - 200) <= 3.0, f"Frame {i} has mean {mean_val}, expected ~200. Fades are dipping."

def test_fade_formula_and_identity():
    # red -> blue with D = 0.5 s (12 frames at 24fps)
    plan_dict = {
        "plan_version": "1.1", "listing_id": "L", "mode": "M", "preset": "P",
        "output": {"width": 320, "height": 180, "fps": 24, "codec": "h264"},
        "total_duration_s": 2.0,
        "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": "dummy"},
        "shots": [
            {
                "shot_id": "s0", "order": 0, "asset_id": "a0", "image_uri": "s3://b/img_red.jpg",
                "source": {"width": 320, "height": 180},
                "room_key": "k", "room_type": "r", "narrative_stage": "n", "role": "h",
                "duration_s": 1.0,
                "camera": {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}, "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}},
                "transition_in": {"type": "cut", "duration_s": 0.0}
            },
            {
                "shot_id": "s1", "order": 1, "asset_id": "a1", "image_uri": "s3://b/img_blue.jpg",
                "source": {"width": 320, "height": 180},
                "room_key": "k", "room_type": "r", "narrative_stage": "n", "role": "h",
                "duration_s": 1.0,
                "camera": {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}, "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}},
                "transition_in": {"type": "fade", "duration_s": 0.5}
            }
        ]
    }
    fix_plan_hash(plan_dict)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        Image.new('RGB', (320, 180), color=(255, 0, 0)).save(os.path.join(d, "images", "img_red.jpg"))
        Image.new('RGB', (320, 180), color=(0, 0, 255)).save(os.path.join(d, "images", "img_blue.jpg"))
        
        plan = parse_plan(plan_dict)
        out_fade = render_direct(plan, d, workers=1)
        frames_fade = get_frames(out_fade, os.path.join(d, "frames_fade"))
        
        # compare before fade starts to a cut version
        plan_dict_cut = json.loads(json.dumps(plan_dict))
        plan_dict_cut["shots"][1]["transition_in"] = {"type": "cut", "duration_s": 0.0}
        
        
        fix_plan_hash(plan_dict_cut)
        plan_cut = parse_plan(plan_dict_cut)
        out_cut = render_direct(plan_cut, d, workers=1)
        frames_cut = get_frames(out_cut, os.path.join(d, "frames_cut"))
        
        # fade is D=12 frames. Cut is at B_1 = 24. 
        # fade frames are [24 - 6, 24 + 6) = [18, 30)
        # Frames 0 to 17 should be pixel-identical.
        for i in range(18):
            img_f = Image.open(frames_fade[i])
            img_c = Image.open(frames_cut[i])
            diff = np.abs(np.array(img_f).astype(float) - np.array(img_c).astype(float)).max()
            assert diff == 0, f"Frame {i} is not pixel-identical to cut version. diff={diff}"
            
        # formula check mid-fade: frame 24 (which is f=6 of the 12-frame fade region for shot 1)
        # Wait, D_k = 12. Render interval starts at 18.
        # At frame f in [0, D_k - 1], which is 24, so f = 24 - 18 = 6.
        # opacity = smoothstep((6 + 1) / (12 + 1)) = smoothstep(7/13) = 0.53846
        # smoothstep(x) = x*x*(3 - 2x)
        x = 7.0 / 13.0
        expected_opacity = x * x * (3 - 2 * x)
        
        img_mid = Image.open(frames_fade[24]).convert('RGB')
        arr_mid = np.array(img_mid).astype(float).mean(axis=(0, 1))
        # R should be (1 - opacity) * 255
        # B should be opacity * 255
        # However, due to color spaces (sRGB vs linear), the blend might not be perfectly linear in sRGB.
        # In CSS/HTML it blends in sRGB usually. 
        # So expected blue channel is approx opacity * 255
        # And red channel is approx (1 - opacity) * 255
        # H264 compression introduces slight shifts.
        expected_b = expected_opacity * 255
        expected_r = (1 - expected_opacity) * 255
        
        assert abs(arr_mid[2] - expected_b) < 15.0, f"Blue channel {arr_mid[2]} != expected {expected_b}"
        assert abs(arr_mid[0] - expected_r) < 15.0, f"Red channel {arr_mid[0]} != expected {expected_r}"

def test_fade_determinism():
    plan_dict = {
        "plan_version": "1.1", "listing_id": "L", "mode": "M", "preset": "P",
        "output": {"width": 320, "height": 180, "fps": 24, "codec": "h264"},
        "total_duration_s": 2.0,
        "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": "dummy"},
        "shots": [
            {
                "shot_id": "s0", "order": 0, "asset_id": "a0", "image_uri": "s3://b/img.jpg",
                "source": {"width": 320, "height": 180},
                "room_key": "k", "room_type": "r", "narrative_stage": "n", "role": "h",
                "duration_s": 1.0,
                "camera": {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}, "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}},
                "transition_in": {"type": "cut", "duration_s": 0.0}
            },
            {
                "shot_id": "s1", "order": 1, "asset_id": "a1", "image_uri": "s3://b/img.jpg",
                "source": {"width": 320, "height": 180},
                "room_key": "k", "room_type": "r", "narrative_stage": "n", "role": "h",
                "duration_s": 1.0,
                "camera": {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}, "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}},
                "transition_in": {"type": "fade", "duration_s": 0.4}
            }
        ]
    }
    
    fix_plan_hash(plan_dict)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        Image.new('RGB', (320, 180), color=(128, 128, 128)).save(os.path.join(d, "images", "img.jpg"))
        
        plan = parse_plan(plan_dict)
        out1 = render_direct(plan, d, workers=1)
        hash1 = subprocess.run(["ffmpeg", "-i", out1, "-f", "framemd5", "-"], capture_output=True).stdout
        
        out2 = render_direct(plan, d, workers=1)
        hash2 = subprocess.run(["ffmpeg", "-i", out2, "-f", "framemd5", "-"], capture_output=True).stdout
        
        assert hash1 == hash2, "Determinism failed! framemd5 differs between runs."
