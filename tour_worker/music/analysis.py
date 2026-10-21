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

  python -m tour_worker.music.analysis <audio> <out.json>        (needs librosa; pin 0.10.2.post1)

Analysed once per track after upload; the JSON is stored with the track (product decision).
"""
import json
import sys

import librosa
import numpy as np

from tour_worker.music import find_transitions as ft

ANALYSIS_VERSION = "3"
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
    for c, h in zip(an["candidates"], hit_strengths(y, [c["time"] for c in an["candidates"]])):
        c["hit"] = h
    return an


if __name__ == "__main__":
    out = make_analysis_v2(sys.argv[1])
    json.dump(out, open(sys.argv[2], "w"), indent=2)
    print(f"{out['tempo_bpm']:.0f} BPM | {len(out['candidates'])} candidates | "
          f"{len(out['whooshes'])} whooshes | music_end {out['music_end']} rhythm_end {out['rhythm_end']}")
