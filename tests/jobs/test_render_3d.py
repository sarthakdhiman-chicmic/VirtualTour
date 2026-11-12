"""tour_render on a plan 1.6 (3D) tour, end to end on the CPU backend: the scene is fetched once
through the sha256 cache, both shots are rasterised, composed (a fade) and encoded like photo shots."""
import json
import math
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


def test_parallel_3d_encoders_give_the_same_bytes_as_one(cpu_backend, monkeypatch, tmp_path):
    """3D chunks are encoded by parallel threads sharing the one renderer: 1 encoder or 3, the
    master is byte-identical (captions on, so the shared label cache is exercised too)."""
    from tour_worker.config import reset_settings
    scene = tmp_path / "box_room.splat"
    write_splat(scene, box_room())
    sha = sha256_file(str(scene))
    p = _plan(scene, sha)
    for s, d in zip(p["shots"], (5.0, 5.0)):                # 10 s at 12 fps: 3 chunks of 48 frames
        s["duration_s"] = d
    p["total_duration_s"] = 10.0
    p["output"]["room_labels"] = {"font": "sacramento", "scale": 0.8, "placement": "bottom_left"}
    p["shots"][0]["caption"], p["shots"][1]["caption"] = "Living Room", "Kitchen"
    p["provenance"]["plan_hash"] = compute_plan_hash(p)
    masters = []
    for n in ("1", "3"):
        monkeypatch.setenv("PREWARP_WORKERS", n)
        reset_settings()
        render_job._SCENE.clear()
        storage = DiskStorage({"s3://scenes/box_room.splat": str(scene)})
        work = tmp_path / f"enc{n}"
        work.mkdir()
        out = render_job.run_tour({"plan": p, "output_prefix": "s3://o/"}, storage, str(work))
        assert out["status"] == "completed", out
        assert out["timings"]["chunks"] == 3 and out["timings"]["render_workers"] == int(n)
        masters.append(open(work / "stored_master.mp4", "rb").read())
    reset_settings()
    assert masters[0] == masters[1]


def _striped_room():
    """The box room with a magenta stripe on the +x wall (shot 1 looks at it): something to see move."""
    m = box_room()
    p = m.means
    band = (p[:, 0] > 1.95) & (np.abs(p[:, 1]) < 0.3) & (p[:, 2] > 0.9) & (p[:, 2] < 1.5)
    m.sh[band, 0] = (np.array([1.0, 0.0, 1.0]) - 0.5) / 0.28209479
    return m, band


def _wall_motion(model, band, scene_sha, path, sha_override=None):
    """The stripe slides 9 cm sideways: a frequency-0 term with phase pi/2 is a constant offset."""
    from tour_worker.scene.motion3d import TERMS, MotionField
    idx = np.flatnonzero(band).astype(np.int64)
    n = len(idx)
    dirs = np.zeros((n, TERMS, 3), np.float32); dirs[:, :, 1] = 1.0
    amp = np.zeros((n, TERMS), np.float32); amp[:, 0] = 0.09
    phase = np.full((n, TERMS), math.pi / 2, np.float32)
    f = MotionField(idx, dirs, amp, np.zeros((n, TERMS), np.float32), phase, sha_override or scene_sha,
                    len(model), {"curtain": n})
    f.save(path)
    return sha256_file(path)


def test_3d_motion_moves_the_frames_deterministically_and_a_bad_file_only_warns(cpu_backend, tmp_path):
    from tour_worker.scene.splat_io import load_scene
    scene = tmp_path / "box_room.splat"
    room, _ = _striped_room()
    write_splat(scene, room)
    sha = sha256_file(str(scene))
    model = load_scene(str(scene))                 # as the worker loads it (indices refer to this)
    band = (model.means[:, 0] > 1.95) & (np.abs(model.means[:, 1]) < 0.3) & (model.means[:, 2] > 0.9) & (model.means[:, 2] < 1.5)
    good = tmp_path / "motion.npz"
    msha = _wall_motion(model, band, sha, str(good))
    bad = tmp_path / "motion_other.npz"
    bsha = _wall_motion(model, band, sha, str(bad), sha_override="cd" * 32)
    files = {"s3://scenes/box_room.splat": str(scene), "s3://scenes/motion.npz": str(good),
             "s3://scenes/motion_other.npz": str(bad)}

    def render(tag, motion=None):
        p = _plan(scene, sha)
        if motion:
            p["scene"]["motion_uri"], p["scene"]["motion_sha256"] = motion
        p["provenance"]["plan_hash"] = compute_plan_hash(p)
        work = tmp_path / tag
        work.mkdir()
        out = render_job.run_tour({"plan": p, "output_prefix": "s3://o/"}, DiskStorage(files), str(work))
        assert out["status"] == "completed", out
        return out, str(work / "stored_master.mp4")

    _, still = render("still")
    out, moving = render("moving", ("s3://scenes/motion.npz", msha))
    assert out["timings"]["motion_splats"] > 0
    _, moving2 = render("moving2", ("s3://scenes/motion.npz", msha))
    assert open(moving, "rb").read() == open(moving2, "rb").read()             # deterministic
    diff = np.abs(_frame(moving, 0.8) - _frame(still, 0.8)).max()
    assert diff > 0.2                                                          # the stripe moved
    _, still2 = render("still2")                                               # cached renderer: motion taken off
    assert open(still2, "rb").read() == open(still, "rb").read()
    out, _ = render("bad", ("s3://scenes/motion_other.npz", bsha))
    assert "scene_motion_skipped" in out["warnings"] and "another scene" in out["timings"]["motion_skipped"]
