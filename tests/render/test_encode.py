"""tour_worker/render/encode.py: the delivery file is capped, keyframed every 2 s, faststart, deterministic."""
import os
import subprocess
import tempfile

from tour_worker.render.encode import GOP_S, deliver, playback_stats, vbv_min_fullness


def _master(path, seconds=6):
    # 1080p30 noise: harder to compress than any photo tour, so it pushes against the cap
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=1920x1080:rate=30:duration={seconds},noise=alls=40:allf=t",
                    "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={seconds}", "-shortest",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
                   check=True)


def test_delivery_is_capped_keyframed_faststart_and_deterministic():
    with tempfile.TemporaryDirectory() as td:
        m = os.path.join(td, "master.mp4")
        _master(m)
        a, b = os.path.join(td, "a.mp4"), os.path.join(td, "b.mp4")
        deliver(m, a, 30)
        deliver(m, b, 30)
        st = playback_stats(a)
        assert st["faststart"] is True
        assert st["max_keyframe_gap_s"] <= GOP_S + 0.05
        assert st["vbv_min_fullness"] >= 0     # a 10 Mbps reader with a 20 Mbit buffer never stalls
        assert st["profile"] == "High" and int(st["level"]) == 41
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read()


def test_vbv_simulation_flags_a_stall():
    fps = 30
    steady = [10e6 / fps] * 90                       # exactly the cap: never drains
    assert abs(vbv_min_fullness(steady, fps) - (0.9 - 1 / 60)) < 1e-9   # first frame leaves before the refill
    burst = [10e6 / fps] * 30 + [3e6] * 10 + [10e6 / fps] * 30   # 30 Mbit in a third of a second
    assert vbv_min_fullness(burst, fps) < 0
