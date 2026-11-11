"""tools/timing_report.py and tour_worker/timing.py."""
import time

from tools.timing_report import report
from tour_worker.timing import Stopwatch, host_info


def test_a_stopwatch_times_laps_and_stages():
    sw = Stopwatch()
    time.sleep(0.01)
    sw.lap("fetch")
    with sw.stage("draw"):
        time.sleep(0.01)
    with sw.stage("draw"):
        pass
    r = sw.result()
    assert r["stages"]["fetch"] >= 0.009 and r["stages"]["draw"] >= 0.009
    assert r["total_s"] >= r["stages"]["fetch"] and r["host"]["cpus"] >= 1


def test_the_report_lists_every_stage_and_scales_to_a_90_s_tour():
    t = {"segment": {"stages": {"model_load": 20.0, "inference": 30.0}, "total_s": 52.0},
         "plan": {"stages": {"fetch": 2.0, "scoring": 8.0, "planning": 5.0}, "total_s": 15.0},
         "render": {"stages": {"draw_encode": 100.0, "audio": 3.0, "join": 2.0}, "total_s": 110.0,
                    "video_s": 90.0, "width": 3840, "height": 2160, "host": {"cpus": 32, "gpu": "A100"},
                    "render_workers": 16, "chunks": 22, "frames": 2700}}
    text = report(t)
    assert "| render | draw_encode | 100.0 |" in text and "| segment | model_load | 20.0 |" in text
    assert "177.0 s" in text and "within the 3-minute target" in text and "A100" in text
    assert host_info()["cpus"] >= 1
