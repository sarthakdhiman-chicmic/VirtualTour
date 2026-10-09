import json
import hashlib
import tempfile
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import pytest
from PIL import Image
from handler import handler

def test_handler_http_storage(monkeypatch):
    monkeypatch.setenv("WORKER_STORAGE", "local")
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("IMAGES_DIR", d)
        # Create dummy image
        img = Image.new('RGB', (1920, 1080), color='blue')
        img_path = os.path.join(d, "img.jpg")
        img.save(img_path)
        
        # Start HTTP server
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=d, **kwargs)
        
        server = HTTPServer(('127.0.0.1', 0), Handler)
        port = server.server_port
        
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        
        try:
            http_url = f"http://127.0.0.1:{port}/img.jpg"
            
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
            
            d_copy = dict(plan_dict)
            d_copy["provenance"] = dict(d_copy["provenance"])
            del d_copy["provenance"]["plan_hash"]
            canon = json.dumps(d_copy, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
            plan_dict["provenance"]["plan_hash"] = hashlib.sha256(canon.encode('utf-8')).hexdigest()
            
            event = {
                "input": {
                    "job_type": "tour_render",
                    "plan": plan_dict,
                    "image_urls": {
                        "s3://bucket/img.jpg": http_url
                    }
                }
            }
            
            # Monkeypatch the check in storage.py to allow our local http server for this test
            import modules.storage
            orig_get = modules.storage.HttpStorage.get
            
            def mock_get(self, uri, dest):
                if uri.startswith("http://127.0.0.1"):
                    uri = uri # allow it
                else:
                    if not uri.startswith("https://"):
                        raise ValueError("Only https:// is supported")
                
                import urllib.request
                req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
                try:
                    with urllib.request.urlopen(req, timeout=15) as response:
                        with open(dest, 'wb') as f:
                            while True:
                                chunk = response.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                except Exception as e:
                    raise RuntimeError(f"HTTP fetch failed for {uri}: {e}")
                    
            modules.storage.HttpStorage.get = mock_get
            orig_comp_get = modules.storage.CompositeStorage.get
            
            def mock_comp_get(self, uri, dest):
                if uri.startswith("http://127.0.0.1"):
                    self.http.get(uri, dest)
                else:
                    orig_comp_get(self, uri, dest)
                    
            modules.storage.CompositeStorage.get = mock_comp_get
            try:
                res = handler(event)
                assert res.get("status") == "completed", f"Failed: {res.get('error')}"
            finally:
                modules.storage.HttpStorage.get = orig_get
                modules.storage.CompositeStorage.get = orig_comp_get
        finally:
            server.shutdown()
