"""tour_worker/music/structure.py on synthetic audio with a known shape: heroes land on the exact
beat where the music changes (not the bar line), drops are tier 1, dropouts tier 2, and a hit
that recurs every bar is texture, not a hero."""
import numpy as np
import pytest

pytest.importorskip("librosa")

from tour_worker.music.structure import SR, analyse_structure  # noqa: E402

BPM, BEAT = 120, 0.5


def _kick(n, f0=60.0, amp=1.0):
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * f0 * t) * np.exp(-t * 18)


def _track():
    """64 bars of 4/4 at 120 bpm: bars 0-15 quiet (soft hats), a drop on beat 3 of bar 16 (loud
    kick + sub-bass + chords), loud until bar 40, a dropout at bar 40, quiet to the end."""
    n_beats = 64 * 4
    y = np.zeros(int((n_beats * BEAT + 1) * SR))
    rng = np.random.default_rng(0)
    drop_beat, out_beat = 16 * 4 + 2, 40 * 4
    t_all = np.arange(len(y)) / SR
    for b in range(n_beats):
        i = int(b * BEAT * SR)
        hat = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / 300)
        y[i:i + len(hat)] += 0.02 * hat
        if drop_beat <= b < out_beat:
            k = _kick(int(0.4 * SR))
            y[i:i + len(k)] += 0.8 * k
    loud = (t_all >= drop_beat * BEAT) & (t_all < out_beat * BEAT)
    y += loud * (0.35 * np.sin(2 * np.pi * 45 * t_all) + 0.15 * np.sin(2 * np.pi * 220 * t_all)
                 + 0.1 * np.sin(2 * np.pi * 277 * t_all))
    beats = [{"time": b * BEAT, "bar_start": b % 4 == 0} for b in range(n_beats)]
    return y.astype(np.float32), beats, drop_beat * BEAT, out_beat * BEAT


@pytest.fixture(scope="module")
def structure():
    y, beats, drop_t, out_t = _track()
    return analyse_structure(y, beats, [], 0.0, beats[-1]["time"]), drop_t, out_t


def test_the_drop_is_tier_1_on_its_exact_beat(structure):
    st, drop_t, _ = structure
    tier1 = [h for h in st["heroes"] if h["tier"] == 1]
    assert [h["kind"] for h in tier1] == ["drop"]
    assert tier1[0]["t"] == pytest.approx(drop_t, abs=1e-6)      # beat 3 of the bar, not the bar line


def test_the_dropout_is_a_tier_2_hero(structure):
    st, _, out_t = structure
    outs = [h for h in st["heroes"] if h["kind"] == "dropout"]
    assert outs and outs[0]["t"] == pytest.approx(out_t, abs=BEAT + 1e-6) and outs[0]["tier"] == 2


def test_steady_beats_are_not_heroes(structure):
    st, drop_t, out_t = structure
    # nothing but the two changes (and their beat neighbours) is a hero: the kick on every beat of
    # the loud part and the hats are the track's texture
    assert all(min(abs(h["t"] - drop_t), abs(h["t"] - out_t)) <= 1.0 for h in st["heroes"]), st["heroes"]


def test_sections_and_phrases_follow_the_shape(structure):
    st, drop_t, out_t = structure
    starts = [s["start"] for s in st["sections"]]
    assert any(abs(s - drop_t) <= 2 * BEAT for s in starts) and any(abs(s - out_t) <= 2 * BEAT for s in starts)
    assert st["phrases"] == sorted(st["phrases"]) and len(st["phrases"]) >= 64 // 4 - 2
