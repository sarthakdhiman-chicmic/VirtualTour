import argparse
import json
import tempfile
import os
import time
from modules.plan_io import parse_plan
from modules.storage import LocalMapStorage, S3Storage
from modules.images import fetch_and_prepare
from modules.probe import verify

def run_tour(request, storage, workdir, scale=1.0):
    start_time = time.time()
    try:
        plan_dict = request.get("plan", {})
        try:
            plan = parse_plan(plan_dict)
        except ValueError as e:
            msg = str(e)
            if "unsupported_camera_type" in msg:
                return {"status": "failed", "error": {"code": "unsupported_camera_type", "message": msg}}
            if "unsupported_easing" in msg:
                return {"status": "failed", "error": {"code": "unsupported_easing", "message": msg}}
            return {"status": "failed", "error": {"code": "invalid_plan", "message": str(e)}}
            
        image_urls = request.get("image_urls", {})
        warnings = fetch_and_prepare(plan, storage, workdir, image_urls)

        # direct renderer (OpenCV + ffmpeg, modules/direct_render.py)
        from modules.direct_render import render_direct
        music_path = None
        if plan.music is not None:                 # plan 1.5: fetch the track
            music_path = os.path.join(workdir, "music", os.path.basename(plan.music.uri) or "track")
            os.makedirs(os.path.dirname(music_path), exist_ok=True)
            try:
                storage.get(plan.music.uri, music_path)
            except Exception:
                warnings.append("music_fetch_failed")
                music_path = None
        out_path = render_direct(plan, workdir, warnings=warnings, music_path=music_path)

        verify(out_path, plan)
        
        # Upload
        output_prefix = request.get("output_prefix", "s3://dummy/out/")
        master_uri = output_prefix + "master.mp4"
        storage.put(out_path, master_uri)
        
        render_seconds = time.time() - start_time
        
        return {
            "status": "completed",
            "video_uri": master_uri,
            "duration_s": plan.total_duration_s,
            "width": plan.output.width,
            "height": plan.output.height,
            "fps": plan.output.fps,
            "codec": plan.output.codec,
            "render_seconds": render_seconds,
            "plan_hash": plan.provenance.plan_hash if hasattr(plan, "provenance") else None,
            "warnings": warnings,
            "error": None
        }
    except RuntimeError as e:
        msg = str(e)
        code = msg.split(":")[0] if ":" in msg else msg
        return {"status": "failed", "error": {"code": code, "message": msg}}
    except Exception as e:
        return {"status": "failed", "error": {"code": "unknown", "message": str(e)}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    
    with open(args.plan) as f:
        plan_data = json.load(f)
        
    request = {
        "job_id": "cli",
        "job_type": "tour_render",
        "plan": plan_data,
        "output_prefix": "s3://dummy/out/"
    }
    
    from modules.storage import LocalMapStorage, S3Storage, HttpStorage, CompositeStorage
    local_storage = LocalMapStorage(args.images_dir)
    local_storage.put = lambda path, uri: __import__('shutil').copy(path, os.path.join(args.out, "master.mp4"))
    
    storage = CompositeStorage(s3=S3Storage(), http=HttpStorage(), local=local_storage)
    
    with tempfile.TemporaryDirectory() as workdir:
        res = run_tour(request, storage, workdir, args.scale)
        print(json.dumps(res, indent=2))
