"""RunPod serverless entry point (RunPod starts this file). All the work is in
tour_worker.jobs.router: one result or {"status": "failed", "error": {"code", "message"}}."""
from tour_worker.jobs.router import run_job


def handler(event):
    return run_job((event or {}).get("input", {}))


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
