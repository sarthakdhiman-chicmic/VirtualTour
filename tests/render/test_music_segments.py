"""Sync v2 music edits (tour_worker/render/direct.py _segment_inputs): segments of the track played
back to back with a crossfade centred on each join - every sound lands at the tour time the planner
computed (sum of the earlier segments + its offset in its own segment)."""
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from tour_worker.render.direct import _audio_args

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
SR = 48000


def _click_track(path, clicks, dur):
    import wave
    y = np.zeros(int(dur * SR), dtype=np.float32)
    for t in clicks:
        i = int(round(t * SR))
        y[i:i + 48] = np.hanning(48) * 0.9
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def _render(music, track, out):
    plan = SimpleNamespace(total_duration_s=music.end_s, music=music)
    ins, args = _audio_args(plan, track, 0)
    graph = args[args.index("-filter_complex") + 1].replace(",aresample=48000[music]", "[music]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *ins, "-filter_complex", graph, "-map", "[music]",
                    "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", out], check=True)
    import wave
    with wave.open(out) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32767


def _clicks(y, thr=0.3):
    idx = np.where(np.abs(y) > thr)[0]
    if not len(idx):
        return []
    groups = np.split(idx, np.where(np.diff(idx) > SR // 100)[0] + 1)
    return [float(g[np.argmax(np.abs(y[g]))]) / SR for g in groups]


def test_every_click_lands_where_the_segments_put_it():
    clicks = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    segs = [(0.0, 2.25), (4.75, 6.25), (1.75, 3.25)]          # cut out 2.25-4.75, then jump back
    music = SimpleNamespace(segments=[SimpleNamespace(from_s=a, to_s=b) for a, b in segs], crossfade_s=0.03,
                            end_s=sum(b - a for a, b in segs), fade_out_s=0.01, loops=1, loop_s=None,
                            track_duration_s=8.0, start_s=0.0)
    expected, base = [], 0.0
    for a, b in segs:
        expected += [base + (t - a) + 0.001 for t in clicks if a + 0.02 < t < b - 0.02]   # 48-tap peak
        base += b - a
    with tempfile.TemporaryDirectory() as d:
        track = os.path.join(d, "clicks.wav")
        _click_track(track, clicks, 8.0)
        got = _clicks(_render(music, track, os.path.join(d, "out.wav")))
    assert len(got) == len(expected)
    assert np.max(np.abs(np.array(got) - np.array(expected))) < 0.002      # within 2 ms
