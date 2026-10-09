import time
import os
import tempfile
import urllib.parse
import resource
from bench import make_synthetic_plan
from modules.plan_io import Plan
from modules.remotion_render import render

def run_sweep():
    # 10s slice at full 4K
    print("Running concurrency sweep on 10s slice at full 4K...")
    plan_dict = make_synthetic_plan(4, 10, 1.0)
    plan_dict["output"]["width"] = 3840
    plan_dict["output"]["height"] = 2160
    plan10 = Plan(**plan_dict)
    
    concurrencies = [2, 4, 8, 12, 16, 20]
    
    for c in concurrencies:
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "images"))
            from PIL import Image
            for i in range(4):
                Image.new('RGB', (6000, 4000), color='blue').save(os.path.join(d, "images", f"img{i}.jpg"))
                
            print(f"\n--- Concurrency {c} ---")
            ru_maxrss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            t0 = time.time()
            try:
                render(plan10, d, scale=1.0, concurrency=c)
                t1 = time.time()
                real_time = t1-t0
                ru_maxrss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                
                print(f"Time: {real_time:.2f}s")
                print(f"FPS: {10*30 / real_time:.2f}")
                print(f"Peak RAM (KB): {max(ru_maxrss_after, ru_maxrss_before)}")
            except Exception as e:
                print(f"Failed: {e}")

if __name__ == "__main__":
    run_sweep()
