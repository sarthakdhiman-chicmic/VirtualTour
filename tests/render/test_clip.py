"""Drone clips as shots (render/clip.py, SOW CM-04 - started): a stretch of a video file as a shot's
frames at the tour's size and rate, deterministic, held when the clip runs short."""
import subprocess

import numpy as np

from tour_worker.render.clip import ClipShot


def _clip(path, seconds=2, size="320x180", rate=25):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}",
                    "-pix_fmt", "yuv420p", path], check=True)


def test_a_clip_gives_the_shots_frames_at_the_tours_size_and_rate(tmp_path):
    p = str(tmp_path / "drone.mp4")
    _clip(p, size="400x300")                                # 4:3 - centre cropped to 16:9
    a = ClipShot(p, 0.5, 30, 320, 180, 30)
    b = ClipShot(p, 0.5, 30, 320, 180, 30)
    assert a.frame(0).shape == (180, 320, 3)
    assert all(np.array_equal(a.frame(f), b.frame(f)) for f in (0, 15, 29))       # deterministic
    assert not np.array_equal(a.frame(0), a.frame(29))                               # it moves
    long = ClipShot(p, 1.5, 60, 320, 180, 30)               # 2 s at 30 fps from a 0.5 s tail: held
    assert np.array_equal(long.frame(40), long.frame(59))
    z = ClipShot(p, 0.5, 30, 320, 180, 30, zoom_end=1.2)
    assert np.array_equal(z.frame(0), a.frame(0)) and not np.array_equal(z.frame(29), a.frame(29))
