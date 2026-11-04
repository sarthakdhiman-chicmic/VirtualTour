"""tour_worker.music: degenerate tracks (silence, very short) analyse sanely or fail clearly."""
import numpy as np
import pytest

librosa = pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")


def _wav(tmp_path, y, name="t.wav"):
    from tour_worker.music.find_transitions import SR
    p = str(tmp_path / name)
    sf.write(p, y.astype(np.float32), SR)
    return p


def test_silence_is_not_a_whoosh():
    from tour_worker.music.analysis import find_whooshes
    from tour_worker.music.find_transitions import SR
    assert find_whooshes(np.zeros(10 * SR, np.float32)) == []
    noise = np.random.default_rng(0).normal(0, 0.3, SR).astype(np.float32)        # a real burst
    y = np.concatenate([np.zeros(2 * SR, np.float32), noise, np.zeros(2 * SR, np.float32)])
    w = find_whooshes(y)
    assert len(w) == 1 and 1.9 <= w[0]["start"] and w[0]["end"] <= 3.1


def test_a_very_short_track_fails_with_a_clear_error(tmp_path):
    from tour_worker.music.find_transitions import SR, analyze
    p = _wav(tmp_path, 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR))
    with pytest.raises(ValueError, match="at least"):
        analyze(p)
