"""Job progress in the log (tour_worker/progress.py): one readable block per job, never part of a result."""
import logging

from tour_worker import progress
from tour_worker.timing import Stopwatch


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "tour_worker.progress"]


def test_a_job_block_has_its_stages_a_bar_and_a_summary(caplog):
    caplog.set_level(logging.INFO, logger="tour_worker.progress")
    clock = Clock()
    rep = progress.Reporter("3f9a1c", "tour_render", clock=clock)
    rep.begin()
    rep.note("1920×1080 · 12.0 s tour")
    rep.stage_start("draw_encode", "4 workers, 10 chunks")
    for done in range(1, 11):
        clock.t += 5
        rep.progress(done, 10)
    rep.stage_end("draw_encode", 50.0)
    rep.end({"status": "completed", "timings": {"stages": {"scene_fetch": 5.0, "draw_encode": 50.0}}})
    out = _lines(caplog)
    assert out[0] == "┌─ job 3f9a1c · tour_render"
    assert out[1] == "│   1920×1080 · 12.0 s tour"
    assert out[2] == "│ ▶ draw_encode   (4 workers, 10 chunks)"
    bars = [l for l in out if "chunks ·" in l]
    assert len(bars) == 10                                   # every 10 %
    assert "[██████████░░░░░░░░░░]  50%   5/10 chunks · 25 s · ETA 25 s" in bars[4]
    assert bars[-1].endswith("100%  10/10 chunks · 50 s")    # no ETA once finished
    assert any(l.startswith("│ ✓ draw_encode") and l.endswith("50.0 s") for l in out)
    assert out[-3].startswith("└─ ✓ COMPLETED in 50.0 s")
    assert out[-1].split()[0] == "draw_encode" and "████████████████████" in out[-1] and out[-1].endswith("91%")


def test_progress_lines_are_throttled_to_steps_or_time(caplog):
    caplog.set_level(logging.INFO, logger="tour_worker.progress")
    clock = Clock()
    rep = progress.Reporter("j", "tour_render", clock=clock)
    rep.stage_start("draw_encode")
    for done in range(1, 201):                               # 200 quick steps: the first, then one per 10 %
        clock.t += 0.01
        rep.progress(done, 200)
    assert len([l for l in _lines(caplog) if "chunks" in l]) == 11
    caplog.clear()
    rep.stage_start("draw_encode")
    rep.progress(1, 1000)
    clock.t += progress.STEP_S + 1                           # slow: a line every STEP_S anyway
    rep.progress(2, 1000)
    rep.progress(2, 1000)                                    # the same count twice: one line
    assert len([l for l in _lines(caplog) if "chunks" in l]) == 2


def test_a_failure_says_where_and_hides_signed_urls(caplog):
    caplog.set_level(logging.INFO, logger="tour_worker.progress")
    rep = progress.Reporter("j", "tour_render", clock=Clock())
    rep.stage_start("scene_fetch")
    rep.stage_end("scene_fetch", 1.0, failed=True)
    rep.end({"status": "failed", "error": {"code": "scene_fetch_failed",
                                           "message": "GET https://b.s3.amazonaws.com/s.ply?X-Amz-Signature=abc failed"}})
    last = _lines(caplog)[-1]
    assert last.startswith("└─ ✗ FAILED in scene_fetch") and "scene_fetch_failed" in last
    assert "X-Amz-Signature" not in last and "abc" not in last


def test_stopwatch_stages_report_to_the_current_job_and_nothing_without_one(caplog):
    caplog.set_level(logging.INFO, logger="tour_worker.progress")
    sw = Stopwatch()
    with sw.stage("alone"):
        pass
    progress.note("nobody listens")
    progress.progress(1, 2)
    assert _lines(caplog) == []                              # no job: no lines, no errors
    result = progress.run("abcdef123456", "tour_plan", lambda: _laps(sw))
    out = _lines(caplog)
    assert result == {"status": "completed"}                 # the result is untouched
    assert out[0] == "┌─ job abcdef12 · tour_plan"
    assert any(l.startswith("│ ▶ scene_load") for l in out) and any(l.startswith("│ ✓ scene_load") for l in out)
    assert any(l.startswith("│ ✓ scoring") for l in out)
    assert not any("per_photo" in l for l in out)            # report=False laps stay out of the log
    assert out[-1].startswith("└─ ✓ COMPLETED") and progress.current() is None


def _laps(sw):
    with sw.stage("scene_load"):
        pass
    sw.lap("per_photo", report=False)
    sw.lap("scoring")
    return {"status": "completed"}


def test_successful_polls_are_left_out_of_the_access_log():
    f = progress._QuietPolls()

    def rec(method, path, code):
        return logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                                 ("1.2.3.4:5", method, path, "1.1", code), None)
    assert not f.filter(rec("GET", "/status/abc", 200))
    assert not f.filter(rec("GET", "/health", 200))
    assert f.filter(rec("POST", "/run", 200))
    assert f.filter(rec("GET", "/status/abc", 404))
