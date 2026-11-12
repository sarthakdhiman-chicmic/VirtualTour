"""Is this GPU pod ready to run the tour worker? One command, one report.

  python -m tools.runpod.preflight
      [--s3-prefix s3://bucket/some/prefix/]   write, read back and delete a small test object there
      [--scene s3://bucket/key.ply]            the scene file exists and can be read (HEAD only)
      [--segment]                              also check tour_segment (transformers, the weights)
      [--tests]                                run the GPU and 3D test suites with pytest
      [--json out.json]                        also write the report as JSON

Checks: ffmpeg / ffprobe; the GPU, CUDA, torch and gsplat versions; one small gsplat render (the
first gsplat call also builds its CUDA kernels if the install did not ship them - that is timed
here, so the first real job does not pay it); S3 access; optionally the segmentation model.
Exit 0 when every check that ran passed, else 1. Nothing here prints credentials or signed URLs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS: list[dict] = []


def check(name):
    def wrap(fn):
        def run(*a, **k):
            t = time.perf_counter()
            try:
                detail = fn(*a, **k) or {}
                ok = True
            except Exception as e:                      # noqa: BLE001 - every failure is reported
                detail, ok = {"error": f"{type(e).__name__}: {e}"}, False
            r = {"check": name, "ok": ok, "seconds": round(time.perf_counter() - t, 2), **detail}
            RESULTS.append(r)
            extras = ", ".join(f"{k}={v}" for k, v in detail.items())
            print(f"[{'ok' if ok else 'FAIL':4s}] {name:22s} {extras}", flush=True)
            return ok, detail
        return run
    return wrap


@check("ffmpeg")
def ffmpeg():
    out = {}
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is not on PATH (tools/runpod/setup.sh installs it)")
        v = subprocess.run([tool, "-hide_banner", "-version"], capture_output=True, text=True, timeout=20)
        out[tool] = v.stdout.split("\n", 1)[0].split(" ")[2] if v.stdout else "?"
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
    if "libx264" not in enc:
        raise RuntimeError("ffmpeg has no libx264 encoder")
    out["libx264"] = True
    out["libx265"] = "libx265" in enc
    return out


@check("gpu")
def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} sees no CUDA GPU")
    p = torch.cuda.get_device_properties(0)
    from tour_worker.render.splat_shot import _nvidia_driver
    return {"name": p.name, "vram_gb": round(p.total_memory / 2**30, 1), "count": torch.cuda.device_count(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "driver": _nvidia_driver()}


def _tiny_scene(n=20_000, seed=0):
    """A random cloud of coloured splats in front of the camera (no repo test fixtures needed)."""
    import numpy as np
    from tour_worker.scene.splat_io import SplatModel
    rng = np.random.default_rng(seed)
    means = (rng.normal(size=(n, 3)) * [0.6, 0.6, 0.4] + [0, 3, 1.2]).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    sh = np.zeros((n, 4, 3), np.float32)
    sh[:, 0] = rng.uniform(-1.5, 1.5, (n, 3))
    return SplatModel(means=means, scales=np.full((n, 3), 0.03, np.float32), quats=quats,
                      opacities=np.full(n, 0.8, np.float32), sh=sh, sh_degree=1, source_format="ply",
                      stats={})


@check("gsplat")
def gsplat_render():
    import importlib.metadata as md
    import numpy as np
    from tour_worker.render.splat_shot import GsplatRenderer
    from tour_worker.scene.camera import intrinsics, look_at
    m = _tiny_scene()
    t = time.perf_counter()
    r = GsplatRenderer(m)
    V, K = look_at([0, 0, 1.2], [0, 3, 1.2]), intrinsics(70, 1920, 1080)
    a = r.render(V, K, 1920, 1080)                      # first call: may build the CUDA kernels
    first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(5):
        b = r.render(V, K, 1920, 1080)
    per = (time.perf_counter() - t) / 5
    if not np.isfinite(a).all() or a.max() <= 0:
        raise RuntimeError("gsplat rendered an empty or non-finite frame")
    return {"gsplat": md.version("gsplat"), "first_call_s": round(first, 2),
            "frame_1080p_ms": round(per * 1000, 1), "deterministic": a.tobytes() == b.tobytes()}


def _s3_split(uri):
    if not uri.startswith("s3://"):
        raise ValueError("expected an s3:// URI")
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


@check("s3 write/read")
def s3_rw(prefix):
    import boto3
    bucket, key = _s3_split(prefix.rstrip("/") + f"/preflight-{uuid.uuid4().hex[:12]}.txt")
    s3 = boto3.client("s3")
    body = f"tour-worker preflight {socket.gethostname()} {time.time()}".encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    try:
        if s3.get_object(Bucket=bucket, Key=key)["Body"].read() != body:
            raise RuntimeError("read back a different body")
    finally:
        s3.delete_object(Bucket=bucket, Key=key)
    return {"bucket": bucket}


@check("scene")
def scene(uri):
    import boto3
    bucket, key = _s3_split(uri)
    h = boto3.client("s3").head_object(Bucket=bucket, Key=key)
    return {"key": key, "size_mb": round(h["ContentLength"] / 1e6, 1),
            "sha256_meta": (h.get("Metadata") or {}).get("sha256", "-")[:12]}


@check("segment model")
def segment():
    import importlib.metadata as md
    from tour_worker.config import get_settings
    from tour_worker.vision.segment import MODELS, ensure_model, model_id
    s = get_settings()
    d = s.segment_dir or os.path.join(s.models_dir, MODELS[s.segment_model]["dir"])
    ensure_model(d, s.segment_model)                    # downloads once, checks the sha256
    return {"model": model_id(s.segment_model), "dir": d, "transformers": md.version("transformers")}


@check("tests")
def tests():
    suites = ["tests/render/test_splat_gpu.py", "tests/render/test_splat_cpu.py", "tests/jobs/test_render_3d.py",
              "tests/contract", "tests/scene"]
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", *suites], cwd=REPO,
                       capture_output=True, text=True, timeout=1800)
    tail = [l for l in p.stdout.strip().splitlines() if l.strip()][-1:] or ["(no output)"]
    if p.returncode != 0:
        print(p.stdout[-4000:], p.stderr[-2000:], sep="\n")
        raise RuntimeError(tail[0])
    if "skipped" in tail[0] and "test_splat_gpu" in p.stdout:
        raise RuntimeError("the GPU tests were skipped: " + tail[0])
    return {"summary": tail[0]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3-prefix")
    ap.add_argument("--scene")
    ap.add_argument("--segment", action="store_true")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    print(f"tour-worker preflight on {socket.gethostname()} (python {platform.python_version()})", flush=True)
    ffmpeg()
    gpu_ok, _ = gpu()
    if gpu_ok:
        gsplat_render()
    if a.s3_prefix:
        s3_rw(a.s3_prefix)
    if a.scene:
        scene(a.scene)
    if a.segment:
        segment()
    if a.tests:
        tests()
    ok = all(r["ok"] for r in RESULTS)
    print(f"\n{'READY' if ok else 'NOT READY'}: {sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks passed")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"host": socket.gethostname(), "ok": ok, "checks": RESULTS}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
