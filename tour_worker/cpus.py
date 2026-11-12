"""How many CPUs this process can really use.

os.sched_getaffinity counts the CPUs the process may be scheduled on, but a container is often
limited by a CPU quota instead: a RunPod pod sees all 128 host CPUs while its cgroup allows ~16.
Sizing thread pools or encoder processes by 128 there oversubscribes the quota many times over
(everything crawls, and each x264 process holds ~0.5 GB). usable_cpus() is the smaller of the
affinity and the cgroup quota (v2 cpu.max or v1 cpu.cfs_quota_us); on a machine without a quota
it is exactly the affinity count, as before.
"""
from __future__ import annotations

import functools
import math
import os


def _quota() -> float | None:
    """The cgroup CPU quota in CPUs, or None when there is none."""
    try:                                                     # cgroup v2
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max" and int(p) > 0:
            return int(q) / int(p)
    except (OSError, ValueError):
        pass
    try:                                                     # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and p > 0:
            return q / p
    except (OSError, ValueError):
        pass
    return None


@functools.lru_cache(maxsize=1)
def usable_cpus() -> int:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    q = _quota()
    if q is not None:
        n = min(n, max(1, math.ceil(q)))
    return max(1, n)
