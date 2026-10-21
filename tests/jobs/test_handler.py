import os
import json
import tempfile
import pytest
from handler import handler

def test_handler_invalid_job_type():
    res = handler({"input": {"job_type": "invalid"}})
    assert res["status"] == "failed"
    assert res["error"]["code"] == "invalid_job_type"

def test_handler_local_storage(monkeypatch):
    import hashlib
    # Mock os.environ to use local storage
    monkeypatch.setenv("WORKER_STORAGE", "local")
    
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("IMAGES_DIR", d)
        
        # Write dummy image
        with open(os.path.join(d, "img.jpg"), "wb") as f:
            f.write(b"dummy")
            
        plan_dict = {
            "plan_version": "1.0", "listing_id": "L1", "mode": "cinematic", "preset": "P1",
            "output": {"width": 320, "height": 180, "fps": 30, "codec": "h264"},
            "total_duration_s": 2.0,
            "shots": [
                {
                    "shot_id": "s1", "order": 1, "asset_id": "a", "image_uri": "s3://bucket/img.jpg",
                    "source": {"width": 1920, "height": 1080},
                    "room_key": "x", "room_type": "y", "narrative_stage": "z", "role": "hero",
                    "duration_s": 2.0,
                    "camera": {
                        "type": "ken_burns", "easing": "ease_in_out",
                        "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                        "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.0}
                    },
                    "transition_in": {"type": "cut", "duration_s": 0.0}
                }
            ],
            "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": ""}
        }
        
        # Calculate valid hash
        d_copy = dict(plan_dict)
        d_copy["provenance"] = dict(d_copy["provenance"])
        del d_copy["provenance"]["plan_hash"]
        canon = json.dumps(d_copy, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
        valid_hash = hashlib.sha256(canon.encode('utf-8')).hexdigest()
        plan_dict["provenance"]["plan_hash"] = valid_hash
        
        event = {
            "input": {
                "job_type": "tour_render",
                "plan": plan_dict,
                "output_prefix": "s3://bucket/out/",
            }
        }
        
        res = handler(event)
        
        # it should error because dummy image is not valid image or because of missing image logic, but it should reach run_tour
        # Wait, if we use a real image, it will work. Let's make a real image.
        from PIL import Image
        img = Image.new('RGB', (1920, 1080), color='white')
        img.save(os.path.join(d, "img.jpg"))
        
        res = handler(event)
        
        if res.get("status") == "failed" or res.get("error") is not None:
            pytest.fail(f"Handler returned error: {res.get('error')}")
        
        assert res.get("status") == "completed"
        from tour_worker.contract.jobs import TourRenderResult
        TourRenderResult.model_validate(res)
        assert res["video_uri"] == "s3://bucket/out/master.mp4" and os.path.exists(os.path.join(d, "master.mp4"))
