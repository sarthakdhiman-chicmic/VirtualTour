"""Per-stage timings for every job result (client review 2026-11-05: "a 4K render ... from a cold
cache, with timings for each stage"). A job times its stages with a Stopwatch and returns
`timings`: seconds per stage, the total, and the host it ran on (CPUs, memory, GPU, render
workers), so a run on a RunPod A100 describes itself."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from contextlib import contextmanager

from tour_worker import progress
from tour_worker.cpus import usable_cpus


class Stopwatch:
    def __init__(self):
        self._t0 = self._lap = time.perf_counter()
        self.stages: dict[str, float] = {}
        self.info: dict = {}

    def lap(self, name: str | None = None, report: bool = True) -> None:
        """Close a stage that ran since the previous lap (name None: start a new lap, timing nothing).
        report: log it as done (tour_worker.progress); a loop's per-item laps pass False."""
        now = time.perf_counter()
        if name:
            self.add(name, now - self._lap)
            rep = progress.current()
            if report and rep is not None:
                rep.stage_end(name, now - self._lap)
        self._lap = now

    @contextmanager
    def stage(self, name: str, detail: str = ""):
        """Time a stage; a stage entered again adds to its total. Logged as it starts and ends."""
        rep = progress.current()
        if rep is not None:
            rep.stage_start(name, detail)
        t = time.perf_counter()
        failed = True
        try:
            yield
            failed = False
        finally:
            self.add(name, time.perf_counter() - t)
            if rep is not None:
                rep.stage_end(name, time.perf_counter() - t, failed=failed)

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = round(self.stages.get(name, 0.0) + seconds, 3)

    def result(self) -> dict:
        return {"stages": dict(self.stages), "total_s": round(time.perf_counter() - self._t0, 3),
                "host": host_info(), **self.info}


def host_info() -> dict:
    """The machine as this job sees it: CPUs it may use, memory, GPU (nvidia-smi, if any)."""
    info = {"cpus": usable_cpus()}
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                info["memory_gb"] = round(int(line.split()[1]) / 2**20, 1)
                break
    except OSError:
        pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            if out:
                info["gpu"] = out[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    pod = os.environ.get("RUNPOD_POD_ID")
    if pod:
        info["runpod_pod"] = pod
    return info
