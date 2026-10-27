"""Musical structure for beat sync v2 (analysis v4): where the music's hero moments are.

v3 knew beats, bar starts, local "changes" and drum hits, but not the song's shape: a drop, a new
section, a build. Its cuts landed on ordinary beats used as transitions, and runs of big moments
collapsed into one (review 2026-10-23: "we are just using a beat to transfer but not the hero
beats"). This module reads the shape from the beat grid v3 already found:

  per beat   loudness (RMS dB), sub-bass dB (< 120 Hz), drum onset, bass onset
  sections   Foote novelty over beat-synchronous chroma + MFCC + loudness + sub-bass, peaks
             snapped to bar starts; each section has a loudness level (low / mid / high)
  heroes     on the exact beat where the music changes, found as loudness / sub-bass STEPS (a
             bar's mean after vs before the beat) - not on the bar line: solarflex's drop is two
             beats after its bar start (12.40 s bar, 13.20 s drop: -25 -> -13 dB, sub +31 dB)
             tier 1: a step up into the track's loud level (a drop, a re-drop after a break)
             tier 2: a smaller step up (an entry), a step down (a dropout / breakdown), a timbre
                     section change with no step, a whoosh, a very big drum or bass hit
  phrases    every PHRASE_BARS bars from each section start: the musical grid cuts fall on when
             no hero is near
  bar_prints per bar: a unit fingerprint (chroma, MFCC, loudness) - two bars with near-equal
             prints can be joined without the ear noticing, so the planner may trim the music
             (sync v2 music edits: remove repeated bars so a drop lands on a cut)

Everything is on the beat grid (a hero between beats takes the nearest beat or half beat).
"""
from __future__ import annotations

import librosa
import numpy as np

from tour_worker.music.kinds import KIND_RANK

SR, HOP = 22050, 256
FR = SR / HOP

KERNEL_BEATS = 16          # Foote kernel half-width: 4 bars of 4/4 either side
MIN_SECTION_BEATS = 16     # sections are at least 4 bars
NOVELTY_PEAK_Z = 0.6       # a boundary's novelty above mean + z x std of the track's novelty
SNAP_BEATS = 2             # a boundary moves to a bar start at most this many beats away
LEVEL_HIGH_DB = 2.5        # within this of the loudest section: "high"
LEVEL_LOW_DB = 5.0         # this far below the median section: "low"
STEP_BEATS = 4             # a step compares the bar after a beat with the bar before it
STEP_LOUD_DB = 6.0         # step score 1.0 = this loudness jump ...
STEP_SUB_DB = 12.0         # ... or this sub-bass jump (the larger counts)
STEP_MIN = 1.0             # a step at least this big is a hero
STEP_TIER1 = 2.0           # a step up this big is tier 1 wherever it lands
PHRASE_BARS = 4
BIG_DRUM = (2.2, 1.6)      # a drum hit x the track's median beat, x the median of 8 bars around
BIG_BASS = (3.5, 2.5)      # the same for a bass onset (sub-bass pulses on every bar are not heroes)
WHOOSH_MIN = 0.5           # whoosh strength for a tier-2 hero
WHOOSH_SNAP_S = 0.12       # a whoosh's start moves onto a beat this close

TEXTURE_BARS = 8           # a hit / whoosh kind more frequent than once per this many bars is the
TEXTURE_KEEP = 0.8         # track's texture (prettyjohn1: a shaker "whoosh" every bar): only its
#                            strongest (this quantile up) stand out, at most one per TEXTURE_BARS


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-6))


def beat_features(y: np.ndarray, beat_times: np.ndarray) -> dict:
    """Per beat (the span from each beat to the next): loudness, sub-bass, drum and bass onset
    peaks at the beat, and chroma / MFCC for the novelty."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    sub = np.sqrt((S[freqs < 120] ** 2).mean(axis=0))
    _, perc = librosa.effects.hpss(y)
    drum_env = librosa.onset.onset_strength(y=perc, sr=SR, hop_length=HOP)
    bass_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[freqs < 200]), sr=SR)
    chroma = librosa.feature.chroma_stft(S=S ** 2, sr=SR)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=SR)), n_mfcc=13)
    n = S.shape[1]
    frames = np.clip(np.round(beat_times * FR).astype(int), 0, n - 1)
    bounds = np.r_[frames, n]
    w = max(1, int(round(0.05 * FR)))                  # onset peak within +-50 ms of the beat

    def span_mean(x):
        return np.array([x[..., a:max(a + 1, b)].mean(axis=-1) for a, b in zip(bounds[:-1], bounds[1:])])

    def peak_at(env):
        return np.array([env[max(0, f - w):f + w + 1].max() for f in frames])

    drum, bass = peak_at(drum_env), peak_at(bass_env)
    return {
        "loud_db": _db(span_mean(rms)),
        "sub_db": _db(span_mean(sub)),
        "drum": drum / (np.median(drum) or 1.0),
        "bass": bass / (np.median(bass) or 1.0),
        "chroma": span_mean(chroma),
        "mfcc": span_mean(mfcc),
    }


def novelty(feats: dict, k: int = KERNEL_BEATS) -> np.ndarray:
    """Foote novelty on the beat-level self-similarity of standardised features."""
    X = np.hstack([feats["chroma"], feats["mfcc"], feats["loud_db"][:, None], feats["sub_db"][:, None]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    ssm = X @ X.T
    n = len(X)
    g = np.exp(-0.5 * ((np.arange(-k, k) + 0.5) / (0.5 * k)) ** 2)
    kern = np.outer(g, g) * np.outer(np.sign(np.arange(-k, k) + 0.5), np.sign(np.arange(-k, k) + 0.5))
    pad = np.pad(ssm, k, mode="edge")
    nov = np.array([(pad[i:i + 2 * k, i:i + 2 * k] * kern).sum() for i in range(n)])
    nov = np.maximum(nov, 0)
    return nov / (nov.max() or 1.0)


def section_bounds(nov: np.ndarray, bar_start: np.ndarray, music_beats: tuple[int, int]) -> list[int]:
    """Beat indices of section starts: novelty peaks above the track's level, at least
    MIN_SECTION_BEATS apart (the stronger wins), each moved to the nearest bar start."""
    lo, hi = music_beats
    thr = nov[lo:hi].mean() + NOVELTY_PEAK_Z * nov[lo:hi].std()
    peaks = [i for i in range(max(1, lo), min(len(nov) - 1, hi))
             if nov[i] >= thr and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    chosen: list[int] = []
    for i in sorted(peaks, key=lambda i: -nov[i]):
        if all(abs(i - j) >= MIN_SECTION_BEATS for j in chosen):
            chosen.append(i)
    out = []
    bars = np.nonzero(bar_start)[0]
    for i in sorted(chosen):
        if len(bars):
            b = bars[np.argmin(np.abs(bars - i))]
            i = int(b) if abs(b - i) <= SNAP_BEATS else i
        if i not in out and i > lo:
            out.append(i)
    return sorted(out)


def _drop_texture(heroes: list[dict], bt, bar, lo: int, hi: int) -> list[dict]:
    """Hits and whooshes that recur every bar or two are texture, not heroes: keep only their
    standouts. Step / section heroes always stay."""
    n_bars = max(1, int(bar[lo:hi].sum()))
    period = TEXTURE_BARS * 4 * float(np.median(np.diff(bt)))        # s per TEXTURE_BARS bars
    out = [h for h in heroes if h["kind"] not in ("whoosh", "bass_hit", "big_hit")]
    for kind in ("whoosh", "bass_hit", "big_hit"):
        hs = [h for h in heroes if h["kind"] == kind]
        if len(hs) <= n_bars / TEXTURE_BARS:
            out += hs
            continue
        floor = float(np.quantile([h["strength"] for h in hs], TEXTURE_KEEP))
        kept: list[dict] = []
        for h in sorted((h for h in hs if h["strength"] >= floor), key=lambda h: (-h["strength"], h["t"])):
            if all(abs(h["t"] - k["t"]) >= period for k in kept):
                kept.append(h)
        out += kept
    return out


def analyse_structure(y: np.ndarray, beats: list[dict], whooshes: list[dict] | None = None,
                      music_start: float = 0.0, music_end: float | None = None) -> dict:
    """The v4 fields: {beat_energy, sections, phrases, heroes}. beats: v3's [{time, bar_start}]."""
    bt = np.array([b["time"] for b in beats], dtype=float)
    bar = np.array([bool(b["bar_start"]) for b in beats])
    if len(bt) < 2 * KERNEL_BEATS:
        return {"structure_version": 1, "beat_energy": [], "sections": [], "phrases": [], "heroes": []}
    music_end = float(music_end if music_end is not None else bt[-1])
    lo = int(np.searchsorted(bt, music_start))
    hi = int(np.searchsorted(bt, music_end))
    f = beat_features(y, bt)
    nov = novelty(f)
    starts = [lo] + section_bounds(nov, bar, (lo, hi))
    ends = starts[1:] + [hi]
    sec_db = [float(np.mean(f["loud_db"][a:b])) for a, b in zip(starts, ends)]
    top, med = max(sec_db), float(np.median(sec_db))
    level = ["high" if d >= top - LEVEL_HIGH_DB else "low" if d <= med - LEVEL_LOW_DB else "mid" for d in sec_db]
    whooshes = whooshes or []

    def near_beat(t: float) -> int:
        return int(np.argmin(np.abs(bt - t)))

    sections, heroes = [], []
    for k, (a, b) in enumerate(zip(starts, ends)):
        sections.append({"start": round(float(bt[a]), 4), "end": round(float(bt[min(b, len(bt) - 1)]), 4),
                         "level": level[k], "loud_db": round(sec_db[k], 2), "novelty": round(float(nov[a]), 3)})

    # steps: where the music gets suddenly louder / fuller (drops) or empties out (breakdowns)
    loud, sub, w = f["loud_db"], f["sub_db"], STEP_BEATS
    score = np.zeros(len(bt))
    for i in range(max(lo, w), min(hi, len(bt) - w)):
        dl = loud[i:i + w].mean() - loud[i - w:i].mean()
        ds = sub[i:i + w].mean() - sub[i - w:i].mean()
        s_ = max(abs(dl) / STEP_LOUD_DB, abs(ds) / STEP_SUB_DB)
        score[i] = s_ * (1 if (dl if abs(dl) / STEP_LOUD_DB >= abs(ds) / STEP_SUB_DB else ds) > 0 else -1)
    peaks = [i for i in range(1, len(bt) - 1)
             if abs(score[i]) >= STEP_MIN and abs(score[i]) >= abs(score[i - 1]) and abs(score[i]) >= abs(score[i + 1])]
    steps: list[int] = []
    for i in sorted(peaks, key=lambda i: -abs(score[i])):
        if all(abs(i - j) >= w for j in steps):
            steps.append(i)
    loud_level = top - LEVEL_HIGH_DB                      # the track's loud sections
    for i in sorted(steps):
        up, mag = score[i] > 0, abs(score[i])
        lands_loud = loud[i:i + 2 * w].mean() >= loud_level
        if up and (lands_loud or mag >= STEP_TIER1):
            heroes.append({"t": round(float(bt[i]), 4), "tier": 1, "kind": "drop",
                           "strength": round(min(1.0, 0.5 + 0.2 * mag), 3)})
        else:
            heroes.append({"t": round(float(bt[i]), 4), "tier": 2, "kind": "entry" if up else "dropout",
                           "strength": round(min(1.0, 0.3 + 0.2 * mag), 3)})
    # a section change with no step nearby (new melody / instruments at the same loudness)
    for a in starts[1:]:
        if all(abs(a - i) > w for i in steps):
            heroes.append({"t": round(float(bt[a]), 4), "tier": 2, "kind": "section",
                           "strength": round(min(1.0, 0.3 + 0.5 * float(nov[a])), 3)})

    # phrases: every PHRASE_BARS bars from each section start (bar starts counted inside it)
    phrases = []
    for a, b in zip(starts, ends):
        bars_in = [i for i in range(a, b) if bar[i] or i == a]
        phrases += [round(float(bt[i]), 4) for i in bars_in[::PHRASE_BARS]]

    # big hits: drum or bass onset well above the track AND its own surroundings
    for i in range(lo, hi):
        for name, kind, (big, big_local) in (("drum", "big_hit", BIG_DRUM), ("bass", "bass_hit", BIG_BASS)):
            v = float(f[name][i])
            local = float(np.median(f[name][max(lo, i - 16):min(hi, i + 17)])) or 1.0
            if v >= big and v / local >= big_local:
                heroes.append({"t": round(float(bt[i]), 4), "tier": 2, "kind": kind,
                               "strength": round(min(1.0, 0.3 + 0.25 * (v - 1)), 3)})
    for w in whooshes:
        if w["strength"] >= WHOOSH_MIN and music_start <= w["peak"] <= music_end:
            # the ear hears a whoosh where it STARTS (review 2026-10-23: "starting frames a bit
            # delayed after the ting" - cuts on the whoosh's loudest point came 0.25-0.45 s after it
            # began); solarflex's whooshes start on a beat and swell to the peak
            i = near_beat(w["start"])
            t = float(bt[i]) if abs(bt[i] - w["start"]) <= WHOOSH_SNAP_S else float(w["start"])
            heroes.append({"t": round(t, 4), "tier": 2, "kind": "whoosh",
                           "strength": round(float(w["strength"]), 3)})

    heroes = _drop_texture(heroes, bt, bar, lo, hi)

    def rank(h):
        return h["tier"], -KIND_RANK.get(h["kind"], 0), -h["strength"]

    # one hero per beat: the higher tier, then what it is, then the stronger, wins
    by_beat: dict[int, dict] = {}
    for h in heroes:
        i = near_beat(h["t"])
        if h["kind"] == "whoosh" and abs(bt[i] - h["t"]) > WHOOSH_SNAP_S:
            i = -1 - len(by_beat)                     # a whoosh between beats keeps its own time
        cur = by_beat.get(i)
        if cur is None or rank(h) < rank(cur):
            by_beat[i] = h
    heroes = sorted(by_beat.values(), key=lambda h: h["t"])

    # bar fingerprints: harmony + timbre + loudness of each bar, unit length (dot = similarity)
    bars_i = [i for i in range(len(bt)) if bar[i]] + [len(bt)]
    prints = []
    for a, b in zip(bars_i, bars_i[1:]):
        if b <= a:
            continue
        v = np.r_[f["chroma"][a:b].mean(0), f["mfcc"][a:b].mean(0) / 50.0, [f["loud_db"][a:b].mean() / 20.0]]
        v = v / (np.linalg.norm(v) or 1.0)
        prints.append({"t": round(float(bt[a]), 4), "v": [round(float(x), 4) for x in v]})

    return {
        "structure_version": 2,
        "bar_prints": prints,
        "beat_energy": [{"t": round(float(t), 4), "loud_db": round(float(d), 2), "sub_db": round(float(s), 2),
                         "step": round(float(score[i]), 3)}
                        for i, (t, d, s) in enumerate(zip(bt, f["loud_db"], f["sub_db"]))],
        "sections": sections,
        "phrases": phrases,
        "heroes": heroes,
    }
