import random
import pytest
from modules.plan_io import Plan
from modules.timeline import compute_timeline

def make_random_plan(num_shots, with_fades=True):
    shots = []
    durations = [random.uniform(1.0, 5.0) for _ in range(num_shots)]
    total_duration = sum(durations)
    
    fps = random.choice([24, 25, 30, 60])
    
    for i in range(num_shots):
        dur = durations[i]
        trans_type = "cut"
        trans_dur = 0.0
        
        if with_fades and i > 0 and random.random() > 0.5:
            trans_type = "fade"
            # D <= 0.4 * min(duration of this shot, duration of the previous shot)
            # D: multiple of 0.01 s, 0.2 <= D <= 1.5
            max_d = min(1.5, 0.4 * dur, 0.4 * durations[i-1])
            if max_d >= 0.2:
                # Random multiple of 0.01
                d_steps = int((max_d - 0.2) / 0.01)
                trans_dur = 0.2 + random.randint(0, d_steps) * 0.01
            else:
                trans_type = "cut"
        
        shot = {
            "shot_id": f"s{i}", "order": i, "asset_id": "a", "image_uri": "s3://b/i",
            "source": {"width": 1920, "height": 1080},
            "room_key": "x", "room_type": "y", "narrative_stage": "z", "role": "h",
            "duration_s": dur,
            "camera": {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}, "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}},
            "transition_in": {"type": trans_type, "duration_s": trans_dur}
        }
        shots.append(shot)
        
    return Plan(**{
        "plan_version": "1.1", "listing_id": "l", "mode": "m", "preset": "p",
        "output": {"width": 1920, "height": 1080, "fps": fps, "codec": "h264"},
        "total_duration_s": total_duration,
        "shots": shots,
        "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": "dummy"}
    })

def test_timeline_properties():
    for _ in range(100):
        num_shots = random.randint(1, 10)
        plan = make_random_plan(num_shots, with_fades=True)
        timeline = compute_timeline(plan)
        
        fps = plan.output.fps
        total_frames = round(plan.total_duration_s * fps)
        
        # Array to count visible shots per frame
        visibility = [0] * total_frames
        
        # Fade regions tracker to check overlap
        fade_regions = [0] * total_frames
        
        for i, t in enumerate(timeline):
            # check B_k unchanged logic internally: B_k = nominal_start
            
            # total frame count unchanged: covered implicitly by checking visibility array up to total_frames
            
            # check odd D_k split
            assert t["head"] + t["tail"] + (t["nominal_end"] - t["nominal_start"]) == t["render_frames"]
            
            for f in range(t["render_from"], t["render_from"] + t["render_frames"]):
                if 0 <= f < total_frames:
                    visibility[f] += 1
                else:
                    # frames rendered outside [0, total_frames-1]? Should not happen
                    assert False, f"Frame {f} out of bounds"
            
            # Record fade regions for overlap check
            # Fade region for shot i is [render_from, render_from + fade_frames)
            if t["fade_frames"] > 0:
                for f in range(t["render_from"], t["render_from"] + t["fade_frames"]):
                    if 0 <= f < total_frames:
                        fade_regions[f] += 1
                        
        # Check invariants
        for f in range(total_frames):
            assert visibility[f] >= 1, f"Frame {f} has no visible shots"
            assert visibility[f] <= 2, f"Frame {f} has >2 visible shots"
            if visibility[f] == 2:
                assert fade_regions[f] > 0, f"Frame {f} has 2 visible shots but not in a fade region"
            assert fade_regions[f] <= 1, f"Frame {f} has overlapping fade regions"
            
        # First shot starts at 0 and has 0 head
        assert timeline[0]["nominal_start"] == 0
        assert timeline[0]["head"] == 0
        
        # Last shot ends at total_frames
        assert timeline[-1]["nominal_end"] == total_frames
        assert timeline[-1]["tail"] == 0
