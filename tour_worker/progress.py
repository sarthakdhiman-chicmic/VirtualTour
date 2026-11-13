"""Readable job progress in the worker's log: one block of lines per job, made to be followed with
`tail -f` (tools/runpod/start.sh logs) - plain lines, no carriage-return redraws.

    12:04:10 ┌─ job 3f9a1c · tour_render
    12:04:10 │   3840×2160 · 66.7 s tour · 12 shots · 3D scene
    12:04:10 │ ▶ scene_fetch
    12:04:16 │ ✓ scene_fetch              5.6 s
    12:04:43 │   [██████░░░░░░░░░░░░░░]  30%    9/30 chunks · 47 s · ETA 1m50s
    12:07:20 └─ ✓ COMPLETED in 190.1 s
               scene_fetch    █▌                     5.6 s    3%

A job's reporter lives in a context variable set around the job (JobStore._execute, or
router.run_job for callers without one), so Stopwatch stages and progress calls made on the job's
thread find it; elsewhere (tests, tools, threads of a pool) every call is a no-op. Reporting never
changes a job's result and never raises into it.
"""
from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager

log = logging.getLogger("tour_worker.progress")

BAR = 20                      # progress bar cells
STEP_PCT = 10                 # a progress line every 10 % ...
STEP_S = 15.0                 # ... or every 15 s, whichever comes first
_EIGHTHS = " ▏▎▍▌▋▊▉"

_CURRENT: contextvars.ContextVar["Reporter | None"] = contextvars.ContextVar("tour_progress", default=None)


def bar(frac: float, cells: int = BAR) -> str:
    """A fixed-width bar, whole cells filled and the rest shaded: [██████░░░░]."""
    frac = min(1.0, max(0.0, frac))
    full = int(round(frac * cells))
    return "[" + "█" * full + "░" * (cells - full) + "]"


def hbar(frac: float, cells: int = BAR) -> str:
    """A bar for the summary chart, to an eighth of a cell, padded to `cells` wide."""
    eighths = int(round(min(1.0, max(0.0, frac)) * cells * 8))
    s = "█" * (eighths // 8) + (_EIGHTHS[eighths % 8] if eighths % 8 else "")
    return s.ljust(cells)


def dur(seconds: float) -> str:
    """12 s, 1m50s, 1h02m."""
    s = int(round(max(0.0, seconds)))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Reporter:
    def __init__(self, job_id: str, job_type: str | None, clock=time.monotonic):
        self.job_id, self.job_type, self.clock = job_id, job_type or "job", clock
        self.t0 = clock()
        self._stage: str | None = None
        self._stage_t0 = self.t0
        self._last_done: str | None = None
        self._prog: dict | None = None

    def _emit(self, line: str, level: int = logging.INFO) -> None:
        try:
            log.log(level, line)
        except Exception:                        # noqa: BLE001 - reporting never breaks a job
            pass

    # ---- job ------------------------------------------------------------------------------------

    def begin(self) -> None:
        self._emit(f"┌─ job {self.job_id} · {self.job_type}")

    def note(self, text: str) -> None:
        self._emit(f"│   {text}")

    def end(self, result) -> None:
        """The job's last lines: ✓ / ✗ and, with timings, a chart of where the time went."""
        total = self.clock() - self.t0
        err = result.get("error") if isinstance(result, dict) else None
        if err:
            code = err.get("code", "unknown") if isinstance(err, dict) else "unknown"
            msg = (err.get("message", "") if isinstance(err, dict) else str(err)).splitlines()
            from tour_worker.service.redact import redact_text   # error text may hold a signed URL
            msg = [redact_text(m) for m in msg]
            where = f" in {self._stage}" if self._stage else (f" after {self._last_done}" if self._last_done else "")
            self._emit(f"└─ ✗ FAILED{where} after {total:.1f} s: {code}"
                       + (f": {msg[0][:200]}" if msg else ""), logging.WARNING)
        else:
            self._emit(f"└─ ✓ COMPLETED in {total:.1f} s")
        timings = result.get("timings") if isinstance(result, dict) else None
        stages = (timings or {}).get("stages") or {}
        if stages and not err:
            top, whole = max(stages.values()) or 1.0, sum(stages.values()) or 1.0
            width = max(len(k) for k in stages)
            for k, v in stages.items():
                self._emit(f"   {k:<{width}}  {hbar(v / top)}  {v:7.1f} s  {100 * v / whole:3.0f}%")

    # ---- stages and progress --------------------------------------------------------------------

    def stage_start(self, name: str, detail: str = "") -> None:
        self._stage, self._stage_t0, self._prog = name, self.clock(), None
        self._emit(f"│ ▶ {name}" + (f"   ({detail})" if detail else ""))

    def stage_end(self, name: str, seconds: float, failed: bool = False) -> None:
        if failed:
            self._emit(f"│ ✗ {name:<20} {seconds:6.1f} s", logging.WARNING)
            return                               # _stage stays: end() says where it failed
        self._emit(f"│ ✓ {name:<20} {seconds:6.1f} s")
        self._stage, self._last_done, self._prog = None, name, None

    def progress(self, done: int, total: int, unit: str = "chunks") -> None:
        """A bar line when done crosses the next STEP_PCT, STEP_S has passed, or it is finished."""
        if total <= 0:
            return
        now = self.clock()
        p = self._prog
        if p is None or p["total"] != total or done < p["done"]:
            p = self._prog = {"t0": now if done == 0 else self._stage_t0, "total": total, "done": -1,
                              "pct": -STEP_PCT, "t": -1e9}
        pct = 100.0 * done / total
        if not (done == total or pct >= p["pct"] + STEP_PCT or now - p["t"] >= STEP_S) or done == p["done"]:
            return
        el = now - p["t0"]
        tail = f" · {dur(el)}"
        if 0 < done < total:
            tail += f" · ETA {dur(el / done * (total - done))}"
        w = len(str(total))
        self._emit(f"│   {bar(done / total)} {pct:3.0f}%  {done:>{w}}/{total} {unit}{tail}")
        p.update(done=done, pct=pct - pct % STEP_PCT, t=now)


# ---- module functions: no-ops without a current job ----------------------------------------------

def current() -> Reporter | None:
    return _CURRENT.get()


@contextmanager
def job(job_id: str, job_type: str | None):
    """Report the job run inside this block; yields its Reporter. Nested: the outer one is kept."""
    if _CURRENT.get() is not None:
        yield _CURRENT.get()
        return
    rep = Reporter(str(job_id)[:8], job_type)
    token = _CURRENT.set(rep)
    rep.begin()
    try:
        yield rep
    finally:
        _CURRENT.reset(token)


def run(job_id: str, job_type: str | None, fn):
    """fn() reported as one job - unless a job is already being reported (the HTTP service opens it
    around the router), then just fn()."""
    if _CURRENT.get() is not None:
        return fn()
    with job(job_id, job_type) as rep:
        result = fn()
        rep.end(result)
    return result


def note(text: str) -> None:
    rep = _CURRENT.get()
    if rep is not None:
        rep.note(text)


def progress(done: int, total: int, unit: str = "chunks") -> None:
    rep = _CURRENT.get()
    if rep is not None:
        rep.progress(done, total, unit)


class _QuietPolls(logging.Filter):
    """Drop uvicorn's access lines for successful polls (GET /status, /health, /ping): a client polls
    every few seconds and those lines would bury the job progress. Everything else is kept."""
    QUIET = ("/status/", "/health", "/ping", "/stream/")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            _client, method, path, _ver, code = record.args
            return not (method == "GET" and int(code) < 400 and str(path).startswith(self.QUIET))
        except (TypeError, ValueError):
            return True


def setup_logging() -> None:
    """The service's log format: progress lines as `HH:MM:SS <line>`, everything else as before
    (time, level, logger). Successful status polls are left out of the access log."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    logging.getLogger("uvicorn.access").addFilter(_QuietPolls())
