import time
import os
import tempfile
import urllib.parse
from modules.plan_io import Plan
from modules.remotion_render import render

def make_synthetic_plan(shots=30, dur=90, scale=0.5):
    # generate a plan
    plan_dict = {
        "plan_version": "1.0", "listing_id": "bench", "mode": "cinematic", "preset": "bench",
        "output": {"width": 3840, "height": 2160, "fps": 30, "codec": "h264"},
        "total_duration_s": dur,
        "shots": [], "warnings": [],
        "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": "bench"}
    }
    
    shot_dur = dur / shots
    for i in range(shots):
        plan_dict["shots"].append({
            "shot_id": f"s{i}", "order": i, "asset_id": "a", "image_uri": f"s3://b/img{i}.jpg",
            "source": {"width": 6000, "height": 4000},
            "room_key": "x", "room_type": "y", "narrative_stage": "z", "role": "hero",
            "duration_s": shot_dur,
            "camera": {
                "type": "ken_burns", "easing": "ease_in_out",
                "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.1}
            },
            "transition_in": {"type": "cut", "duration_s": 0.0}
        })
        
    return plan_dict

def run_bench():
    # 30-shot / 90s plan at scale 0.5
    print("Running 90s bench at scale 0.5...")
    plan90 = Plan(**make_synthetic_plan(30, 90, 0.5))
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        from PIL import Image
        for i in range(30):
            Image.new('RGB', (600, 400), color='blue').save(os.path.join(d, "images", f"img{i}.jpg"))
            
        t0 = time.time()
        render(plan90, d, scale=0.5)
        t1 = time.time()
        print(f"90s at scale 0.5 took {t1-t0:.2f}s")
        print(f"Frames per second: {90*30 / (t1-t0):.2f}")

    # 10s slice at full 4K
    print("Running 10s slice at full 4K...")
    plan10 = Plan(**make_synthetic_plan(4, 10, 1.0))
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        from PIL import Image
        for i in range(4):
            Image.new('RGB', (6000, 4000), color='blue').save(os.path.join(d, "images", f"img{i}.jpg"))
            
        t0 = time.time()
        render(plan10, d, scale=1.0)
        t1 = time.time()
        
        real_time = t1-t0
        print(f"10s at full 4K took {real_time:.2f}s")
        print(f"Frames per second: {10*30 / real_time:.2f}")
        
        cpu_count = os.cpu_count()
        gpu_used = False # Could check for nvenc if we enabled it
        print(f"CPU count: {cpu_count}, GPU used: {gpu_used}")
        est_90 = real_time * 9
        print(f"ESTIMATE for 90s at 4K: {est_90:.2f}s")
        print(f"SOW Target: <180s. Difference: {est_90 - 180:.2f}s")

if __name__ == "__main__":
    run_bench()
