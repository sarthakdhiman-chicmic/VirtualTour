import os
import tempfile
from run_tour import run_tour
from modules.storage import S3Storage, HttpStorage, CompositeStorage, LocalMapStorage

def handler(event):
    input_data = event.get("input", {})
    job_type = input_data.get("job_type")
    if job_type not in ("tour_render", "tour_segment"):
        return {"error": "invalid_job_type: job_type must be tour_render or tour_segment"}

    if os.environ.get("WORKER_STORAGE") == "local":
        storage = CompositeStorage(s3=S3Storage(), http=HttpStorage(), local=LocalMapStorage(os.environ.get("IMAGES_DIR", "/tmp")))
    else:
        storage = CompositeStorage(s3=S3Storage(), http=HttpStorage())
        
    with tempfile.TemporaryDirectory() as workdir:
        if job_type == "tour_segment":
            # OneFormer masks before planning (modules/segment_job.py); masks go to S3, JSON back
            from modules.segment_job import run_segment
            res = run_segment(input_data, CompositeStorage(s3=S3Storage(), http=HttpStorage()), workdir)
        else:
            res = run_tour(input_data, storage, workdir)
        if res.get("status") == "failed":
            err = res.get("error", {})
            return {"error": f"{err.get('code', 'unknown')}: {err.get('message', 'Unknown error')}"}
        return res

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
