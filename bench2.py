import time
import os
import tempfile
from bench import make_synthetic_plan
from modules.plan_io import Plan
from modules.remotion_render import render

def run_bench():
    # 10 shots, 0.6s fades vs cuts
    print("Benchmarking 10 shots with cuts vs 0.6s fades (Concurrency = 12)")
    
    # Base plan parameters: 10 shots, ~3s per shot, total ~30s
    plan_dict_cuts = make_synthetic_plan(10, 30, 1.0)
    for i in range(10):
        plan_dict_cuts["shots"][i]["duration_s"] = 3.0
        plan_dict_cuts["shots"][i]["transition_in"] = {"type": "cut", "duration_s": 0.0}
    
    # The total duration is roughly 30s.
    plan_dict_cuts["total_duration_s"] = 30.0
    
    import copy
    plan_dict_fades = copy.deepcopy(plan_dict_cuts)
    for i in range(1, 10):
        plan_dict_fades["shots"][i]["transition_in"] = {"type": "fade", "duration_s": 0.6}
        
    plan_cuts = Plan(**plan_dict_fades) # wait, I just re-instantiated plan_dict_cuts
    plan_cuts = Plan(**plan_dict_cuts)
    plan_fades = Plan(**plan_dict_fades)
    
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        from PIL import Image
        for i in range(10):
            Image.new('RGB', (1920, 1080), color='blue').save(os.path.join(d, "images", f"img{i}.jpg"))
            
        print("Rendering CUTS...")
        t0 = time.time()
        render(plan_cuts, d, scale=1.0, concurrency=12)
        t_cuts = time.time() - t0
        print(f"Cuts Time: {t_cuts:.2f}s")
        
        print("Rendering FADES...")
        t0 = time.time()
        render(plan_fades, d, scale=1.0, concurrency=12)
        t_fades = time.time() - t0
        print(f"Fades Time: {t_fades:.2f}s")
        
        overhead = ((t_fades - t_cuts) / t_cuts) * 100
        print(f"Overhead: {overhead:.2f}%")

if __name__ == "__main__":
    run_bench()
