"""Run a heavy local job (segmentation + render) without overloading this machine.

  python3 tools/safe_run.py [--cpus 0,2,4,6] [--max-temp 82] [--resume-temp 70] [--memory 8G] -- <cmd...>

Why: the dev box hard-reset twice during full renders (no OOM, no kernel error; the journal just
stops while Chrome/encoder/workers load every core). 8 busy threads take the CPU from 57 to 87 C
in 20 s. So the job is
  - pinned to a few CPUs (taskset; inherited by every child: PyTorch, ffmpeg, render
    workers), one hardware thread per physical core to spread the heat;
  - told to use that many threads (OMP/MKL/OpenBLAS, PREWARP_WORKERS);
  - run in a systemd user scope with a memory limit;
  - FROZEN (cgroup.freeze) when the CPU package reaches --max-temp, and thawed below
    --resume-temp: the job pauses, it is not killed.
Temperatures and freezes are logged once per 10 s to out/safe_run.log (survives a reboot).
"""
import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time

# outside the repo (the worker folder holds code only): SAFE_RUN_LOG, else ~/.cache/tour-worker/
LOG = os.environ.get("SAFE_RUN_LOG") or os.path.join(os.path.expanduser("~"), ".cache", "tour-worker",
                                                     "safe_run.log")


def pkg_temp_path() -> str | None:
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if open(os.path.join(z, "type")).read().strip() == "x86_pkg_temp":
                return os.path.join(z, "temp")
        except OSError:
            pass
    return None


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, file=sys.stderr, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())                     # on disk even if the machine resets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", default="0,2,4,6")
    ap.add_argument("--max-temp", type=float, default=82.0)
    ap.add_argument("--resume-temp", type=float, default=70.0)
    ap.add_argument("--memory", default="8G")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    n = len(a.cpus.split(","))
    env = dict(os.environ, OMP_NUM_THREADS=str(n), MKL_NUM_THREADS=str(n), OPENBLAS_NUM_THREADS=str(n),
               PREWARP_WORKERS=str(max(1, n - 1)))
    unit = f"safe-run-{os.getpid()}"
    inner = ["taskset", "-c", a.cpus] + cmd
    scoped = shutil.which("systemd-run") is not None
    full = (["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}", "-p", f"MemoryMax={a.memory}"]
            + inner) if scoped else inner
    temp_path = pkg_temp_path()
    log(f"start cpus={a.cpus} max={a.max_temp}C resume={a.resume_temp}C mem={a.memory} cmd={' '.join(cmd)}")
    proc = subprocess.Popen(full, env=env, start_new_session=True)

    def freeze_file():
        hits = glob.glob(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@*.service/**/{unit}.scope/"
                         "cgroup.freeze", recursive=True)
        return hits[0] if hits else None

    def set_frozen(on: bool) -> None:
        f = freeze_file() if scoped else None
        if f:
            open(f, "w").write("1" if on else "0")
        else:                                     # no cgroup: stop / continue the process group
            os.killpg(proc.pid, signal.SIGSTOP if on else signal.SIGCONT)

    def stop(*_):
        set_frozen(False)
        os.killpg(proc.pid, signal.SIGTERM)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    frozen, last, peak, paused_s = False, 0.0, 0.0, 0.0
    while proc.poll() is None:
        time.sleep(0.5)             # this CPU can jump ~20 C within a second
        if temp_path is None:
            continue
        t = int(open(temp_path).read()) / 1000
        peak = max(peak, t)
        if not frozen and t >= a.max_temp:
            set_frozen(True)
            frozen = True
            log(f"FREEZE at {t:.0f}C")
        elif frozen and t <= a.resume_temp:
            set_frozen(False)
            frozen = False
            log(f"resume at {t:.0f}C")
        if frozen:
            paused_s += 0.5
        if time.time() - last >= 10:
            last = time.time()
            log(f"temp {t:.0f}C{' (frozen)' if frozen else ''}")
    log(f"exit {proc.returncode}: peak {peak:.0f}C, paused {paused_s:.0f}s")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
