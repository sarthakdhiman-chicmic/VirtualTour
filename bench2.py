import time
import os
import tempfile
import urllib.parse
from bench import make_synthetic_plan
from modules.plan_io import Plan
from modules.remotion_render import render
from modules.images import fetch_and_prepare
from modules.storage import LocalMapStorage

def run_bench2():
    print("Running 10s slice at full 4K with 1.2x pre-scaling...")
    plan_dict = make_synthetic_plan(4, 10, 1.0)
    plan_dict["output"]["width"] = 3840
    plan_dict["output"]["height"] = 2160
    # The synthetic plan uses s3://b/img{i}.jpg
    # We will create fake large images in a local dir, and use LocalMapStorage to fetch them
    
    with tempfile.TemporaryDirectory() as src_dir:
        from PIL import Image
        for i in range(4):
            Image.new('RGB', (6000, 4000), color='blue').save(os.path.join(src_dir, f"img{i}.jpg"))
            
        storage = LocalMapStorage(src_dir)
        
        with tempfile.TemporaryDirectory() as workdir:
            plan10 = Plan(**plan_dict)
            
            t_start = time.time()
            fetch_and_prepare(plan10, storage, workdir)
            t_prep = time.time()
            
            render(plan10, workdir, scale=1.0, concurrency=8)
            t_render = time.time()
            
            print(f"Prep time: {t_prep - t_start:.2f}s")
            print(f"Render time: {t_render - t_prep:.2f}s")
            print(f"Total time: {t_render - t_start:.2f}s")

if __name__ == "__main__":
    run_bench2()
