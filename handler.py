import os
import tempfile
from run_tour import run_tour
from modules.storage import S3Storage, HttpStorage, CompositeStorage, LocalMapStorage

def handler(event):
    input_data = event.get("input", {})
    if input_data.get("job_type") != "tour_render":
        return {"error": "invalid_job_type: job_type must be tour_render"}
    
    if os.environ.get("WORKER_STORAGE") == "local":
        storage = CompositeStorage(s3=S3Storage(), http=HttpStorage(), local=LocalMapStorage(os.environ.get("IMAGES_DIR", "/tmp")))
    else:
        storage = CompositeStorage(s3=S3Storage(), http=HttpStorage())
        
    with tempfile.TemporaryDirectory() as workdir:
        res = run_tour(input_data, storage, workdir)
        if res.get("status") == "failed":
            err = res.get("error", {})
            return {"error": f"{err.get('code', 'unknown')}: {err.get('message', 'Unknown error')}"}
        return res

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
