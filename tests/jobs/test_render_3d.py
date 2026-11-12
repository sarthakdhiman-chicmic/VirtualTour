"""tour_render on a plan 1.6 (3D) tour, end to end on the CPU backend: the scene is fetched once
through the sha256 cache, both shots are rasterised, composed (a fade) and encoded like photo shots."""
import json
import os
import subprocess

import numpy as np
import pytest

import tour_worker.jobs.render as render_job
from tests.scene_fixtures import FACE_RGB, box_room, write_splat
from tour_worker.contract.plan import compute_plan_hash
from tour_worker.scene.cache import sha256_file

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "plans", "plan_1_6_box_room.json")


class DiskStorage:
    def __init__(self, files):
        self.files, self.gets, self.puts = files, [], {}

    def get(self, uri, dest):
        self.gets.append(uri)
        with open(self.files[uri], "rb") as s, open(dest, "wb") as d:
            d.write(s.read())

    def put(self, path, uri):
        self.puts[uri] = path
        with open(path, "rb") as s, open(os.path.join(os.path.dirname(path), "stored_" + uri.split("/")[-1]), "wb") as d:
            d.write(s.read())


def _plan(scene_file, sha):
    p = json.load(open(FIXTURE))
    p["output"].update(width=160, height=90, fps=12)
    # two short shots inside the 4 x 4 x 2.5 box room (floor z=0), a fade between them
    p["shots"] = p["shots"][:2]
    for s, d in zip(p["shots"], (1.0, 1.0)):
        s["duration_s"] = d
    p["shots"][0]["camera"]["segments"] = [{"eye": [[-1, 0, 1.4], [-0.8, 0, 1.4], [-0.6, 0, 1.4], [-0.4, 0, 1.4]],
                                            "target": [[3, 0, 1.2]] * 4}]
    p["shots"][1]["camera"]["segments"] = [{"eye": [[0, -1, 1.4], [0, -0.8, 1.4], [0, -0.6, 1.4], [0, -0.4, 1.4]],
                                            "target": [[0, 3, 1.2]] * 4}]
    p["shots"][1]["transition_in"] = {"type": "fade", "duration_s": 0.25}
    p["total_duration_s"] = 2.0
    p["scene"] = {"uri": "s3://scenes/box_room.splat", "sha256": sha, "format": "splat"}
    p["provenance"]["plan_hash"] = compute_plan_hash(p)
    return p


@pytest.fixture
def cpu_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("SPLAT_BACKEND", "cpu")
    monkeypatch.setenv("SPLAT_SUPERSAMPLE", "1.0")
    monkeypatch.setenv("SCENE_CACHE_DIR", str(tmp_path / "cache"))
    render_job._SCENE.clear()
    yield
    render_job._SCENE.clear()


def _frame(path, t, w=160, h=90):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3) / 255.0


def test_a_3d_tour_renders_end_to_end_on_the_cpu_backend(cpu_backend, tmp_path):
    scene = tmp_path / "box_room.splat"
    write_splat(scene, box_room())
    sha = sha256_file(str(scene))
    storage = DiskStorage({"s3://scenes/box_room.splat": str(scene)})
    work = tmp_path / "w1"
    work.mkdir()
    out = render_job.run_tour({"plan": _plan(scene, sha), "output_prefix": "s3://out/t/"}, storage, str(work))
    assert out["status"] == "completed", out
    master = str(work / "stored_master.mp4")
    assert abs(_frame(master, 0.2)[45, 80] - FACE_RGB["+x"]).max() < 0.08     # shot 1 looks at the +x wall
    assert abs(_frame(master, 1.8)[45, 80] - FACE_RGB["+y"]).max() < 0.08     # shot 2 at the +y wall
    assert storage.gets == ["s3://scenes/box_room.splat"]


def test_the_scene_is_fetched_once_and_a_wrong_file_is_scene_invalid(cpu_backend, tmp_path):
    scene = tmp_path / "box_room.splat"
    write_splat(scene, box_room())
    sha = sha256_file(str(scene))
    storage = DiskStorage({"s3://scenes/box_room.splat": str(scene)})
    for k in (1, 2):
        work = tmp_path / f"w{k}"
        work.mkdir()
        assert render_job.run_tour({"plan": _plan(scene, sha), "output_prefix": "s3://o/"}, storage, str(work))["status"] == "completed"
    assert len(storage.gets) == 1                                  # the second render reused it
    render_job._SCENE.clear()
    bad = _plan(scene, "ef" * 32)
    work = tmp_path / "w3"
    work.mkdir()
    out = render_job.run_tour({"plan": bad, "output_prefix": "s3://o/"}, storage, str(work))
    assert out["status"] == "failed" and out["error"]["code"] == "scene_invalid"


def test_without_a_gpu_the_default_backend_fails_cleanly(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SPLAT_BACKEND", raising=False)
    render_job._SCENE.clear()
    import torch
    if torch.cuda.is_available():
        pytest.skip("a CUDA GPU is present")
    scene = tmp_path / "box_room.splat"
    write_splat(scene, box_room())
    storage = DiskStorage({"s3://scenes/box_room.splat": str(scene)})
    work = tmp_path / "w"
    work.mkdir()
    out = render_job.run_tour({"plan": _plan(scene, sha256_file(str(scene))), "output_prefix": "s3://o/"}, storage, str(work))
    assert out["status"] == "failed" and out["error"]["code"] == "scene_render_failed"


def test_without_scene_cache_dir_the_scene_is_not_kept_on_disk(cpu_backend, monkeypatch, tmp_path):
    """The default (SCENE_CACHE_DIR unset): the scene is fetched into the job's folder and deleted
    once loaded; the loaded scene stays in memory, so a second render does not fetch it again."""
    monkeypatch.delenv("SCENE_CACHE_DIR")
    scene = tmp_path / "box_room.splat"
    write_splat(scene, box_room())
    sha = sha256_file(str(scene))
    storage = DiskStorage({"s3://scenes/box_room.splat": str(scene)})
    for i in range(2):
        work = tmp_path / f"job{i}"
        work.mkdir()
        out = render_job.run_tour({"plan": _plan(scene, sha), "output_prefix": "s3://o/"}, storage, str(work))
        assert out["status"] == "completed", out
        assert not [p for p in work.rglob("*") if p.suffix in (".splat", ".ply")]
    assert storage.gets == ["s3://scenes/box_room.splat"]
    assert not (tmp_path / "cache").exists()
