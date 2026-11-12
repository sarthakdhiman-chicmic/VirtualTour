"""Render memory (2026-10-27: a 4K tour took a 15 GB machine down twice). The worker count follows
the output size and the memory this job may use - the host's, or less inside a cgroup limit - and
caps a configured count too; the local heat guard never asks for more memory than is free."""
import importlib.util
import os

from tour_worker import cpus

from tour_worker.render import frames

GB = 2**30


def test_a_worker_costs_more_at_4k(monkeypatch):
    assert 1.0 * GB < frames.worker_mem_bytes(1920, 1080) < 1.5 * GB
    assert frames.worker_mem_bytes(1080, 1920) == frames.worker_mem_bytes(1920, 1080)
    assert 3.5 * GB < frames.worker_mem_bytes(3840, 2160) < 4.5 * GB
    # the optional H.265 copy adds an x265 (measured 0.43 GB at 1080p, 1.35 GB at 4K)
    assert 0.4 * GB < frames.worker_mem_bytes(1920, 1080, hevc=True) - frames.worker_mem_bytes(1920, 1080) < 0.5 * GB
    assert 1.3 * GB < frames.worker_mem_bytes(3840, 2160, hevc=True) - frames.worker_mem_bytes(3840, 2160) < 1.4 * GB


def test_the_worker_count_fits_the_memory_budget(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)))
    monkeypatch.setattr(cpus, "_quota", lambda: None)            # this host's own CPU quota aside
    monkeypatch.setattr(frames, "_available_memory", lambda: 16 * GB)
    monkeypatch.setattr(frames, "_cgroup_headroom", lambda: None)
    assert frames.render_workers(None, 1920, 1080) == 7                 # 16 GB x 0.6 / 1.2 GB
    assert frames.render_workers(None, 3840, 2160) == 2                 # memory bounds it
    # a configured count (the heat guard sets PREWARP_WORKERS=7) is capped as well: uncapped, it
    # started 7 4K workers (~26 GB)
    assert frames.render_workers(7, 3840, 2160) == 2
    assert frames.render_workers(3, 1920, 1080) == 3
    monkeypatch.setattr(frames, "_cgroup_headroom", lambda: 5 * GB)     # 5 GB left inside a limit
    assert frames.render_workers(None, 1920, 1080) == 2
    monkeypatch.setattr(frames, "_available_memory", lambda: 1 * GB)    # never zero
    assert frames.render_workers(None, 3840, 2160) == 1


def test_the_cgroup_headroom_is_the_tightest_limit_on_the_way_up(monkeypatch, tmp_path):
    root = tmp_path / "cg"
    scope = root / "user.slice" / "job.scope"
    scope.mkdir(parents=True)
    (root / "user.slice" / "memory.max").write_text(str(10 * GB))
    (root / "user.slice" / "memory.current").write_text(str(9 * GB))   # 1 GB left above
    (scope / "memory.max").write_text(str(4 * GB))
    (scope / "memory.current").write_text(str(1 * GB))                  # 3 GB left in the scope
    real_open, real_join = open, os.path.join

    def fake_open(path, *a, **kw):
        if path == "/proc/self/cgroup":
            return real_open(tmp_path / "self_cgroup", *a, **kw)
        return real_open(path, *a, **kw)

    (tmp_path / "self_cgroup").write_text("0::/user.slice/job.scope\n")
    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(os.path, "join", lambda a, *p: real_join(str(root), *p) if a == "/sys/fs/cgroup" else real_join(a, *p))
    assert frames._cgroup_headroom() == 1 * GB


def test_the_heat_guard_never_caps_above_free_memory(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("safe_run", os.path.join(os.path.dirname(__file__), "..", "..",
                                                                         "tools", "safe_run.py"))
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    real_open = open
    meminfo = tmp_path / "meminfo"

    def fake_open(path, *a, **kw):
        return real_open(meminfo, *a, **kw) if path == "/proc/meminfo" else real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    meminfo.write_text("MemTotal: 16000000 kB\nMemAvailable: 4194304 kB\n")          # 4 GB free
    assert sr.memory_cap("9G") == "2560M"                                # 4 GB - 1.5 GB headroom
    meminfo.write_text("MemTotal: 64000000 kB\nMemAvailable: 50331648 kB\n")         # 48 GB free
    assert sr.memory_cap("9G") == "9216M"                                # what was asked


def test_the_mls_copy_costs_a_second_encoder():
    from tour_worker.render.frames import worker_mem_bytes
    assert worker_mem_bytes(1920, 1080, mls=True) > worker_mem_bytes(1920, 1080)
    assert worker_mem_bytes(1920, 1080, hevc=True, mls=True) > worker_mem_bytes(1920, 1080, hevc=True)
