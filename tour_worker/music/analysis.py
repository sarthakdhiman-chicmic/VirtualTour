"""Music analysis v3 for beat-synced tours: find_transitions.py's analysis (v1, unchanged) plus
  - whooshes: noise-like, bright bursts (swooshes / transition effects). Each becomes a cut
    candidate of kind "whoosh" at its loudest moment, valued above any ordinary beat, so the
    planner cuts ON the whoosh. Review 2026-10-16: with beats only, the cut landed on the bar
    after the whoosh, "about a second or so later".
  - rhythm_end: where the hits stop (reverb / fade tails and held notes excluded). The
    planner loops a queued track at the bar nearest it; v1's music_end includes the tail
    (sample4.mp3: music stops at 27.6 s, music_end said 29.7 s -> a 1.4 s gap at each join).
  - hit (v3, per candidate): how hard the drums hit there - the percussive part's onset strength
    (HPSS) at the candidate, / the track's median beat. Review 2026-10-19: v1 places bar starts on
    beat 1, but sample4's loudest hit in every bar is beat 3; cuts on the "bar" came half a bar
    (0.69 s) after the hit the ear follows ("a beat way before the scene ends").

  - v4 (2026-10-23, beat sync v2): the music's structure (tour_worker/music/structure.py) -
    beat_energy, sections, phrases and heroes (tiered hero moments on their exact beats). All
    v3 fields are unchanged, so a v3 planner reads a v4 analysis as before.

  python -m tour_worker.music.analysis <audio> <out.json>        (needs librosa; pin 0.10.2.post1)

Analysed once per track after upload; the JSON is stored with the track (product decision).
"""
import json
import sys

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from tour_worker.music import find_transitions as ft
from tour_worker.music.structure import analyse_structure

ANALYSIS_VERSION = "4"
SR, HOP = ft.SR, ft.HOP
FR = SR / HOP
WHOOSH_VALUE = 1.0          # x strength; ordinary candidates run ~ -0.4..0.7 (median 0)


def find_whooshes(y, min_len=0.12, merge_gap=0.08, flat_thr=0.03, hi_db_thr=-30.0):
    """Bursts that are noise-like (spectral flatness) and bright (energy above 3 kHz) for at
    least min_len s. Single drum hits are too short; tonal music is not flat."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    flat = librosa.feature.spectral_flatness(S=S)[0]
    hi = S[f > 3000].sum(0)
    hi_db = librosa.amplitude_to_db(hi, ref=np.max(hi) + 1e-12)
    on = (flat > flat_thr) & (hi_db > hi_db_thr)
    gap = max(1, int(merge_gap * FR))
    runs, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and (on[j] or on[j:j + gap].any()):
            j += 1
        runs.append((i, j))
        i = j
    out = []
    for a, b in runs:
        if (b - a) / FR < min_len:
            continue
        peak = a + int(np.argmax(hi[a:b]))
        strength = float(np.clip((hi_db[peak] + 30) / 30, 0, 1) * np.clip(flat[a:b].max() / 0.08, 0, 1))
        out.append({"start": round(a / FR, 4), "end": round(b / FR, 4), "peak": round(peak / FR, 4),
                    "strength": round(strength, 3)})
    return out


def rhythm_end(candidates: list[dict]) -> float:
    """Where the beat stops: the last beat whose hit is at least typical for the track (v1's
    candidate value is centred on the median beat, and already weighs hit strength and
    audibility). Loudness or raw onsets are not enough: a held note can ring on after the drums
    stop, and faint echo hits still register as onsets (sample4.mp3)."""
    beats = [c["time"] for c in candidates if c["kind"] != "offbeat" and c["value"] >= 0.0]
    return round(float(beats[-1]), 4) if beats else 0.0


HIT_WINDOW_S = 0.035        # the onset peak may sit a few ms off the beat time


def hit_strengths(y, times: list[float]) -> list[float]:
    """Percussive onset strength at each time (max within +-HIT_WINDOW_S), / the median over the
    times (1.0 = a typical beat of this track; the loudest drum hits run ~2)."""
    if not times:
        return []
    _, perc = librosa.effects.hpss(y)
    env = librosa.onset.onset_strength(y=perc, sr=SR, hop_length=HOP)
    w = max(1, int(round(HIT_WINDOW_S * FR)))
    raw = []
    for t in times:
        i = int(round(t * FR))
        seg = env[max(0, i - w):i + w + 1]
        raw.append(float(seg.max()) if len(seg) else 0.0)
    med = float(np.median(raw)) or 1.0
    return [round(r / med, 3) for r in raw]


TING_HZ = 2500.0


def ting_strengths(y, times: list[float]) -> list[float]:
    """v4: how bright the transient at each time is - onset strength above TING_HZ (hi-hats,
    chimes, the "ting" the ear locks onto), max within +-HIT_WINDOW_S, / the median over the times.
    Review 2026-10-23: cuts a beat after a ting read "delayed after the ting sound"; drum `hit`
    alone does not see these."""
    if not times:
        return []
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=64))
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[f > TING_HZ]), sr=SR, hop_length=64)
    fr = SR / 64
    w = max(1, int(round(HIT_WINDOW_S * fr)))
    raw = [float(env[max(0, int(round(t * fr)) - w):int(round(t * fr)) + w + 1].max(initial=0.0)) for t in times]
    med = float(np.median(raw)) or 1.0
    return [round(r / med, 3) for r in raw]


BANDS_HZ = ((0.0, 150.0), (150.0, 2500.0), (2500.0, 10000.0))
# The ONE definition of "how hard the sound hits" - used by the planner (via the analysis), the
# regression check and the audit of a rendered video (tools/quality/sync_audio_audit.py), so they
# can never disagree again (review 2026-10-23: a per-beat scale made hi-hat "tings" look small next
# to the kick; the planner fixed one late cut and made another - "we fix some, we lose some").
#   accent(t)  the strongest of low (<150 Hz), mid and high (>2.5 kHz) onset strength at t, each
#              band / its median over ALL frames OF THE TRACK (accent_ref, stored in the analysis:
#              the audit of a render divides by the same numbers - a render's own medians differ)
#   peaks      the accent curve's local maxima (accent_peaks, stored): the planner reads strike /
#              pre off them on the tour as it plays, music edits included
#   strike     the accent peak within STRIKE_WIN of a time (the hit a cut there lands on)
#   pre        the accent peak within PRE_WIN before it (up to about a beat earlier)
# A cut reads late when pre > LATE_RATIO x strike.
# Robust to encoding (2026-10-23, Turkey Trot renders vs the track, 37 cuts): single 3 ms flux
# spikes differed up to 41% after AAC and flipped 4 late verdicts. Now: no content above 10 kHz
# (AAC mangles it), a 60 dB floor, and the flux integrated over 40 ms (an attack as the ear hears
# it): 1.3% median / 7.6% worst difference, no verdict flips.
STRIKE_WIN = (-0.10, 0.12)
PRE_WIN = (-0.48, -0.10)
LATE_RATIO = 1.15
ACCENT_HOP = 64
ACCENT_TOP_DB = 60.0
ACCENT_SMOOTH_S = 0.04


def accent_curve(y: np.ndarray, sr: int = SR, ref: list[float] | None = None) -> tuple[np.ndarray, float, list[float]]:
    """(accent per frame, frames per second, per-band reference) - see above. ref: the track's
    own reference (from its analysis) when scoring other audio of it (a render)."""
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=ACCENT_HOP))
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    bands, out_ref = [], []
    for k, (lo, hi) in enumerate(BANDS_HZ):
        m = (f >= lo) & (f < hi) if hi else (f >= lo)
        env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[m], top_db=ACCENT_TOP_DB), sr=sr,
                                           hop_length=ACCENT_HOP)
        env = uniform_filter1d(env, max(1, int(round(ACCENT_SMOOTH_S * sr / ACCENT_HOP))))
        r = float(ref[k]) if ref else (float(np.median(env)) or 1.0)
        out_ref.append(r)
        bands.append(env / r)
    return np.max(bands, axis=0), sr / ACCENT_HOP, out_ref


PEAK_HALF = 3               # a peak is the max of the frames within +-PEAK_HALF of it
PEAK_MIN = 1.0              # ... and at least the track's typical moment


def accent_peaks(acc: np.ndarray, fr: float) -> list[list[float]]:
    """[[t, value]] of the accent curve's local maxima at or above PEAK_MIN."""
    from scipy.ndimage import maximum_filter1d
    mx = maximum_filter1d(acc, size=2 * PEAK_HALF + 1, mode="nearest")
    idx = np.where((acc >= mx - 1e-12) & (acc >= PEAK_MIN))[0]
    keep, last = [], -10 * PEAK_HALF
    for i in idx:                                   # flat tops: the first frame only
        if i - last > PEAK_HALF:
            keep.append(i)
            last = i
    return [[round(float(i / fr), 4), round(float(acc[i]), 2)] for i in keep]


def strike_and_pre(y, times: list[float]) -> tuple[list[float], list[float], list[float], list[list[float]]]:
    """v4: per time (strike, pre) on the shared accent scale, the track's reference and its
    accent peaks (the planner recomputes strike / pre from the peaks on the played tour)."""
    from tour_worker.planner.music_sync_v2 import peaks_window_max
    acc, fr, ref = accent_curve(y)
    peaks = accent_peaks(acc, fr)
    pt = np.array([p[0] for p in peaks]); pv = np.array([p[1] for p in peaks])
    return ([round(peaks_window_max(pt, pv, t, STRIKE_WIN), 3) for t in times],
            [round(peaks_window_max(pt, pv, t, PRE_WIN), 3) for t in times], ref, peaks)


def make_analysis_v2(path: str) -> dict:
    an = ft.make_analysis(path)
    y, _ = librosa.load(path, sr=SR, mono=True)
    an["analysis_version"] = ANALYSIS_VERSION
    an["rhythm_end"] = min(rhythm_end(an["candidates"]), an["music_end"])
    an["whooshes"] = find_whooshes(y)
    for w in an["whooshes"]:
        an["candidates"].append({"time": w["peak"], "bar_start": False, "kind": "whoosh",
                                 "value": round(WHOOSH_VALUE * w["strength"], 5), "change": 1.0,
                                 "what_happens": "whoosh (transition effect)"})
    an["candidates"].sort(key=lambda c: c["time"])
    times = [c["time"] for c in an["candidates"]]
    strike, pre, an["accent_ref"], an["accent_peaks"] = strike_and_pre(y, times)
    for c, h, t, s, p in zip(an["candidates"], hit_strengths(y, times), ting_strengths(y, times), strike, pre):
        c["hit"], c["ting"], c["strike"], c["pre"] = h, t, s, p
    an.update(analyse_structure(y, an["beats"], an["whooshes"], an["music_start"], an["music_end"]))
    return an


if __name__ == "__main__":
    out = make_analysis_v2(sys.argv[1])
    json.dump(out, open(sys.argv[2], "w"), indent=2)
    print(f"{out['tempo_bpm']:.0f} BPM | {len(out['candidates'])} candidates | "
          f"{len(out['whooshes'])} whooshes | music_end {out['music_end']} rhythm_end {out['rhythm_end']} | "
          f"{len(out['sections'])} sections | heroes: {sum(h['tier'] == 1 for h in out['heroes'])} tier 1, "
          f"{sum(h['tier'] == 2 for h in out['heroes'])} tier 2")
