"""
find_transitions.py - beat-synced transition points for a video.

Three commands:

  cuts     (default, original behaviour) pick transition points from the music
           alone: every transition ON A BEAT, every shot --min-gap..--max-gap s.
  analyze  analyse a music track once and store the result as JSON (beat grid,
           bar starts, scored cut candidates). Keyed by the file's sha256, so it
           can be cached per track in the music library.
  sync     snap a tour's planned shot lengths onto the music: exactly one cut
           per planned transition, each shot kept close to its planned length,
           every cut on a beat (bar starts and big musical moments preferred),
           the tour ending on a bar line, every cut on an exact video frame.

Install:
    pip install librosa numpy scipy soundfile
    (optional, for --plot)  pip install matplotlib
    Pin the librosa version in production: results can shift between versions.

Usage:
    python find_transitions.py song.mp3 --clicks                     (= cuts)
    python find_transitions.py cuts song.mp3 --min-gap 3 --max-gap 6 --clicks --plot
    python find_transitions.py analyze song.mp3
    python find_transitions.py sync song.mp3 --plan shot_plan.json --audio-out tour_music.wav
    python find_transitions.py sync song.analysis.json --targets 6,7.5,5,4,3.5 --clicks

Outputs (named after the input file):
    cuts:    song.transitions.json / .csv, song.clicks.wav (--clicks), song.analysis.png (--plot)
    analyze: song.analysis.json
    sync:    song.sync.json (cut times, synced durations, per-shot report)
             <plan>.synced.json (--plan: the plan with duration_s replaced,
               total_duration_s updated, a "music" block added, plan_hash recomputed)
             --audio-out FILE: the music trimmed to the tour, faded out at the end
             song.sync_clicks.wav (--clicks): music with a tick on every synced cut
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import warnings

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", message="Trying to estimate tuning")  # silent passages

ANALYSIS_VERSION = "1"
SR = 22050
HOP = 256                      # ~11.6 ms per frame -> precise timing
FR = SR / HOP                  # frames per second
BANDS = {                      # what a listener hears as separate layers
    "sub-bass": (20, 120),
    "bass": (120, 400),
    "mid-range": (400, 2500),
    "treble": (2500, SR / 2),
}


# ---------------------------------------------------------------- helpers
def level_change(power, w, ref):
    """Loudness (dB) of the w frames before / after each frame, averaging ENERGY
    (not dB) so short hits like hi-hats count the way ears hear them."""
    c = np.concatenate([[0.0], np.cumsum(power)])
    n = len(power)
    t = np.arange(n)
    a, d = np.clip(t - w, 0, n), np.clip(t + w, 0, n)
    before = 10 * np.log10((c[t] - c[a]) / np.maximum(t - a, 1) / ref + 1e-12)
    after = 10 * np.log10((c[d] - c[t]) / np.maximum(d - t, 1) / ref + 1e-12)
    valid = ((t - a) >= w // 2) & ((d - t) >= w // 2)
    return np.where(valid, after - before, 0.0), before, after


def norm(x, pct=99.5):
    x = np.clip(np.asarray(x, float), 0, None)
    return np.clip(x / (np.percentile(x, pct) + 1e-9), 0, 1)


def local_norm(x, seconds=6.0):
    """Scale a hit-strength curve against the strongest hits of the surrounding
    few seconds, so quiet songs/sections and any dominant instrument are treated fairly."""
    x = np.clip(np.asarray(x, float), 0, None)
    env = gaussian_filter1d(maximum_filter1d(x, size=max(3, int(seconds * FR))), FR)
    floor = 0.05 * np.percentile(x, 99) + 1e-9
    return np.clip(x / (env + floor), 0, 1)


def foote_novelty(feats, half):
    """High where the sound before a point differs from the sound after it."""
    X = librosa.util.normalize(feats, axis=0)
    S = X.T @ X
    g = np.exp(-0.5 * np.linspace(-2, 2, 2 * half) ** 2)
    K = np.outer(g, g)
    K[:half, half:] *= -1
    K[half:, :half] *= -1
    P = np.pad(S, half, mode="edge")
    nov = np.array([np.sum(K * P[i:i + 2 * half, i:i + 2 * half]) for i in range(S.shape[0])])
    return np.clip(nov, 0, None)


def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:06.3f}"


# ---------------------------------------------------------------- analysis
def analyze(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    duration = len(y) / sr

    # --- per-band loudness (dB relative to the loudest moment of the song)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=HOP,
                                         n_mels=96, fmin=20, fmax=sr / 2, norm=None)  # true energy per band
    freqs = librosa.mel_frequencies(n_mels=96, fmin=20, fmax=sr / 2)
    total_p = mel.sum(axis=0)
    n = len(total_p)
    W_LONG, W_SHORT = int(1.0 * FR), int(0.25 * FR)
    ref = np.convolve(total_p, np.ones(W_LONG) / W_LONG, mode="same").max() + 1e-12

    _, tot_before, tot_after = level_change(total_p, W_LONG, ref)
    band_delta, band_sig, band_share = {}, {}, {}
    for name, (lo, hi) in BANDS.items():
        bp = mel[(freqs >= lo) & (freqs < hi)].sum(axis=0)
        d, before, after = level_change(bp, W_LONG, ref)
        own_max = np.max(np.where(d != 0, np.maximum(before, after), -120))
        rb, ra = np.maximum(before - own_max, -50), np.maximum(after - own_max, -50)
        d = ra - rb                                       # change relative to the layer's own level
        # a layer only matters if it is a noticeable share of the mix (on either side of the change)
        share = np.maximum(before - tot_before, after - tot_after)
        audible = (np.clip((np.maximum(rb, ra) + 50) / 30, 0, 1)
                   * np.clip((share + 40) / 20, 0, 1))
        band_delta[name] = d
        band_sig[name] = np.abs(d) * audible
        band_share[name] = share
    # soft saturation (not a hard clip) so the biggest change keeps a sharp peak
    band_score = 1 - np.exp(-np.max(np.vstack(list(band_sig.values())), axis=0) / 10.0)

    # --- sudden stops / drops after a break (short window on total loudness)
    short_d, short_before, _ = level_change(total_p, W_SHORT, ref)
    short_score = 1 - np.exp(-np.clip(np.abs(short_d) - 4.0, 0, None) / 10.0)  # ignore small wobbles between hits
    long_d = level_change(total_p, W_LONG, ref)[0]

    # --- big hits: clearly stronger than the hits of the last ~3 seconds
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)[:n]
    onset = np.pad(onset, (0, n - len(onset)))
    look = int(3 * FR)
    prev_max = np.array([onset[max(0, i - look):max(1, i - 5)].max() for i in range(n)])
    ratio = np.clip(onset / (prev_max + 1e-6) - 1.0, 0, 3) / 3.0
    impact_score = ratio * norm(onset)

    # --- section changes: timbre (MFCC) and harmony (chords), at ~10 fps
    step = 8
    def coarse(F):
        F = F[:, :n]
        F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-9)
        k = F.shape[1] // step
        return F[:, :k * step].reshape(F.shape[0], k, step).mean(axis=2), k

    def novelty(F):
        Fc, k = coarse(F)
        nv = foote_novelty(Fc, half=max(4, int(2.5 * FR / step)))
        return norm(np.interp(np.arange(n), np.arange(k) * step + step / 2, nv))

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    struct_score = novelty(mfcc)
    y_harm = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP)
    harmony_score = novelty(chroma)

    # a moment counts if ANY layer changes strongly (max), boosted when several agree (mean)
    stack = np.vstack([band_score, struct_score, 0.7 * harmony_score, short_score, 0.8 * impact_score])
    weighted = (0.35 * band_score + 0.20 * struct_score + 0.10 * harmony_score
                + 0.20 * short_score + 0.15 * impact_score)
    combined = 0.6 * stack.max(axis=0) + 0.4 * weighted
    combined = gaussian_filter1d(combined, 3)

    # --- rhythm layers from the percussive / harmonic split (HPSS).
    # Frequency ranges only *suggest* instruments: they are not instrument detection.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    S = np.pad(S[:, :n], ((0, 0), (0, max(0, n - S.shape[1]))))
    H, P = librosa.decompose.hpss(S, margin=1.5)
    ff = librosa.fft_frequencies(sr=sr, n_fft=2048)
    sref = S.max() + 1e-12

    band_level = {}

    def band_onset(M, lo, hi, key):
        # floor 50 dB below the song's peak: rises out of inaudible noise are not hits
        db = librosa.amplitude_to_db(M[(ff >= lo) & (ff < hi)], ref=sref, top_db=50)
        band_level[key] = db.max(axis=0)[:n]
        o = librosa.onset.onset_strength(S=db, sr=sr, hop_length=HOP)[:n]
        return np.pad(o, (0, n - len(o)))

    raw_layers = dict(
        low_hit=band_onset(P, 30, 150, "low_hit"),        # low-frequency hits (kick-like)
        mid_hit=band_onset(P, 1000, 6000, "mid_hit"),     # sharp mid/high hits (snare/clap-like)
        high_hit=band_onset(P, 7000, sr / 2, "high_hit"), # very high ticks (hi-hat/shaker-like)
        bass_note=band_onset(H, 40, 250, "bass_note"),    # new sustained low notes (bassline)
    )
    layers = {k: local_norm(v) for k, v in raw_layers.items()}

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=HOP, trim=False)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    beat_len = 60.0 / max(float(np.atleast_1d(tempo)[0]), 30.0)

    # --- accents: a hit clearly louder than the same layer's hits of the last
    # ~2 bars (at least 3 s), so slow songs compare against their own phrases
    look, gap = int(max(3.0, 8 * beat_len) * FR), max(1, int(0.1 * FR))
    accent = np.zeros(n)
    for k in ("low_hit", "mid_hit", "bass_note"):
        x = raw_layers[k]
        trailing = maximum_filter1d(x, size=look, origin=(look - 1) // 2)   # max of [i-look+1, i]
        prev = np.concatenate([np.zeros(gap), trailing[:-gap]])             # ... ending 0.1 s before i
        floor = 0.1 * np.percentile(x, 99) + 1e-9
        accent = np.maximum(accent, np.clip((x / (prev + floor) - 1.15) / 0.6, 0, 1))

    # --- noise gate: ignore "hits" in near-silence
    w = max(1, int(0.1 * FR))
    loud_db = 10 * np.log10(np.convolve(total_p, np.ones(w) / w, mode="same") / ref + 1e-12)
    gate = np.clip((loud_db + 45) / 15, 0, 1)

    # --- energy 0..1: loudness + rhythmic density (hits per second across layers)
    w2 = int(2 * FR)
    lvl = 10 * np.log10(np.convolve(total_p, np.ones(w2) / w2, mode="same") / ref + 1e-12)
    density = np.zeros(n)
    for k in ("low_hit", "mid_hit", "high_hit"):
        o = raw_layers[k]
        pk, _ = find_peaks(o, height=max(0.3 * np.percentile(o, 99), 1.5),
                           distance=max(1, int(0.06 * FR)))
        # count a hit only where it is audible in the mix
        pk = pk[(gate[pk] > 0.5) & (band_level[k][np.minimum(pk + 2, n - 1)] > -40)]
        imp = np.zeros(n)
        imp[pk] = 1.0
        density += np.convolve(imp, np.ones(w2), mode="same") / 2.0      # hits per second
    # density dominates: a ballad mastered loud is still calm; a busy groove is energetic
    energy = 0.3 * np.clip((lvl + 30) / 30, 0, 1) + 0.7 * np.clip(density / 8.0, 0, 1)
    energy = gaussian_filter1d(energy, FR)


    return dict(y=y, sr=sr, duration=duration, combined=combined, onset=onset,
                band_delta=band_delta, band_sig=band_sig, band_share=band_share, short_d=short_d,
                short_before=short_before, long_d=long_d,
                parts=dict(band=band_score, struct=struct_score, harmony=harmony_score,
                           short=short_score, impact=impact_score),
                tempo=float(np.atleast_1d(tempo)[0]), beat_times=beat_times,
                layers=layers, raw_layers=raw_layers, band_level=band_level, gate=gate, energy=energy,
                accent=accent * gate)


def snap_to_hit(p, onset, onset_frames, radius=0.35):
    """Move a change point onto the actual attack (drum hit / note start) nearby.
    Looks a bit further back than forward: change curves peak just after their cause."""
    r = int(radius * FR)
    near = onset_frames[(onset_frames - p >= -int(1.5 * r)) & (onset_frames - p <= r)]
    if len(near) == 0:
        return p
    weight = onset[near] * np.exp(-0.5 * ((near - p) / (0.25 * FR)) ** 2)
    return int(near[np.argmax(weight)])


def find_events(A):
    """Exact moments where the music changes: peaks of the change curve,
    snapped onto the hit that starts them. Returns [(frame, peak_frame, height)]."""
    C = A["combined"]
    spread = np.percentile(C, 99) - np.median(C)
    peaks, _ = find_peaks(C, prominence=spread * 0.08, distance=int(0.75 * FR))
    # how much each peak stands out from the OTHER peaks of the surrounding ~12 s:
    # a pattern that repeats every bar is not a "change"; a peak towering over its
    # neighbours (drop, break, new layer) is
    win = int(6 * FR)
    height_of = {}
    for p in peaks:
        neigh = peaks[(np.abs(peaks - p) <= win) & (peaks != p)]
        ref_h = np.median(C[neigh]) if len(neigh) else np.median(C)
        stand_out = np.clip((C[p] - ref_h) / (0.4 * spread + 1e-9), 0, 1)
        height_of[p] = 0.3 * C[p] + 0.7 * stand_out
    onset_frames = librosa.onset.onset_detect(onset_envelope=A["onset"], sr=SR,
                                              hop_length=HOP, backtrack=False)
    events = []
    for p in peaks:
        if p < int(0.5 * FR) or p > len(C) - int(0.5 * FR):
            continue
        f = p if A["long_d"][p] < -3 else snap_to_hit(p, A["onset"], onset_frames)
        events.append((f, p, float(height_of[p])))
    return events


def beat_grid(A, events=()):
    """Beat times from the tracker, each nudged (max 50 ms) onto the real hit."""
    onset = A["onset"]
    frames = np.round(A["beat_times"] * FR).astype(int)
    frames = frames[(frames >= 0) & (frames < len(onset))]
    r = int(0.05 * FR)
    strong = np.percentile(onset, 60)
    out = []
    for f in frames:
        a, b = max(0, f - r), min(len(onset), f + r + 1)
        g = a + int(np.argmax(onset[a:b]))
        out.append(g if onset[g] > strong else f)
    out = np.unique(out)

    # intros / breaks without drums have no detected beats: continue the grid at the tempo
    if len(out) >= 2:
        period = float(np.median(np.diff(out)))
        filled = list(np.arange(out[0] - period, 0, -period)[::-1])
        for a, b in zip(out[:-1], out[1:]):
            filled.append(a)
            if b - a > 1.5 * period:
                k = int(round((b - a) / period))
                filled.extend(a + (b - a) * np.arange(1, k) / k)
        filled.append(out[-1])
        filled.extend(np.arange(out[-1] + period, len(onset) - 1, period))
        out = np.unique(np.round(filled).astype(int))

    # downbeats: choose which beat of 4 is "beat 1". Big musical changes almost
    # always start a bar, so use them first; hit strength breaks ties.
    ev_score = np.zeros(len(out))
    strong = [e for e in events if e[2] >= 0.6]          # only clear changes vote for "beat 1"
    for f, _, h in strong:
        i = int(np.argmin(np.abs(out - f)))
        if abs(out[i] - f) <= 0.12 * FR:
            ev_score[i] += h
    low = A["layers"]["low_hit"] if "layers" in A else onset
    tie = lambda k: (onset[out[k::4]].mean() / (onset.mean() + 1e-9)
                     + low[out[k::4]].mean() / (low.mean() + 1e-9)) if len(out[k::4]) else 0
    phase_score = [ev_score[k::4].sum() * 10 + tie(k) for k in range(4)]
    phase = int(np.argmax(phase_score))
    downbeat = np.array([(i - phase) % 4 == 0 for i in range(len(out))])
    return out, downbeat


def snap_stops_to_grid(A, events, frames):
    """A stop / break starts on the first beat that is MISSING its hit, not on
    the last hit before it (the change curve and snap_to_hit put it there).
    For every event where the rhythm drops away (overall level falls, or the
    hits of the next second are far weaker than those of the last second),
    move the cut onto the first grid beat after the last real hit. The grid
    continues through breaks, so this is where the break actually begins."""
    if len(frames) < 2:
        return events
    onset = A["onset"]
    n = len(onset)
    period = float(np.median(np.diff(frames)))
    w = int(1.0 * FR)
    out = []
    for f, p, h in events:
        before = onset[max(0, p - w):p]
        after = onset[p:min(n, p + w)]
        drops = (A["long_d"][p] < -3) or (
            len(before) and len(after) and before.mean() > np.percentile(onset, 50)
            and after.mean() < 0.35 * before.mean())
        if drops:
            lo = max(0, p - w)
            seg = onset[lo:min(n, p + int(0.2 * FR))]
            strong = np.where(seg >= 0.5 * seg.max())[0] if len(seg) and seg.max() > 0 else []
            last_hit = lo + int(strong[-1]) if len(strong) else f
            nxt = frames[frames > last_hit + int(0.05 * FR)]
            if len(nxt) and nxt[0] - last_hit <= 1.25 * period:
                f = int(nxt[0])
        out.append((f, p, h))
    return out


def explain(p, A):
    """Human-readable reason + suggested transition for a change at frame p."""
    sd, ld = A["short_d"][p], A["long_d"][p]
    e = A["energy"]
    rose = e[max(0, p - 2)] - e[max(0, p - int(6 * FR))]      # energy climb over the last ~6 s
    if sd > 6 and ld > 6:
        if rose > 0.15:
            return "drop after a build-up", "hard cut on the hit, flash or zoom-in"
        return "drop / big entrance", "hard cut on the hit, flash or zoom-in"
    if sd < -6 and ld < -6:
        return "sudden stop / break", "freeze frame, cut to black or slow-mo"

    parts = {k: v[p] for k, v in A["parts"].items()}
    weights = dict(band=0.35, struct=0.20, harmony=0.10, short=0.20, impact=0.15)
    main = max(parts, key=lambda k: parts[k] * weights[k])
    if main == "band":
        b = max(A["band_sig"], key=lambda name: A["band_sig"][name][p])
        if A["band_share"][b][p] > -25:          # only name a layer that is a real part of the mix
            up = A["band_delta"][b][p] > 0
            reason = f"{b} {'comes in / gets louder' if up else 'drops out / gets quieter'}"
            return reason, ("hard cut / zoom-in / speed ramp" if up else "crossfade / slow dissolve")
        main = "harmony" if parts["harmony"] >= 0.5 else "struct"
    if main == "struct":
        return "new section (the sound changes)", "cut, whip pan or slide"
    if main == "harmony":
        return "chord / key change", "smooth slide, dissolve or light leak"
    if main == "impact":
        L = A["layers"]
        r = slice(max(0, p - 3), p + 4)
        low, mid = L["low_hit"][r].max(), L["mid_hit"][r].max()
        if low >= mid:
            return "big low-frequency hit (kick/bass-like)", "hard cut, shake or flash"
        return "sharp hit (snare/clap-like)", "hard cut, shake or flash"
    return "energy change", "cut"


def score_candidates(A, frames, downbeat, events, bar_weight):
    """Candidate cut points and their musical value.

    Candidates: every beat, plus off-beat (half-beat) positions where a real
    syncopated hit clearly stands out from the beats around it.
    Value combines: musical change (structure, layers, drops), hit strength on
    that position (low / mid hits and bass notes, each judged against its own
    recent level, so no instrument is assumed to dominate), position in the bar,
    accent size, and confidence that a real sound is there (not silence/noise)."""
    C, L, gate = A["combined"], A["layers"], A["gate"]
    n = len(C)
    hit = np.maximum.reduce([L["low_hit"], L["mid_hit"], 0.6 * L["bass_note"]])
    near = lambda arr, f, r=2: float(arr[max(0, f - r):min(n, f + r + 1)].max())

    first = int(np.argmax(downbeat)) if downbeat.any() else 0
    cands = []                       # (frame, is_bar_start, metric weight, kind)
    for i, f in enumerate(frames):
        pos = (i - first) % 4
        cands.append((int(f), bool(downbeat[i]), {0: 1.0, 2: 0.6}.get(pos, 0.35), "beat"))
    # raw (un-normalised) hit strength, so a loud off-beat hit is compared fairly
    R = A["raw_layers"]
    raw_hit = np.maximum.reduce([R["low_hit"], R["mid_hit"], 0.6 * R["bass_note"]])
    r = int(0.04 * FR)
    for a, b in zip(frames[:-1], frames[1:]):
        m = (int(a) + int(b)) // 2
        lo = max(0, m - r)
        g = lo + int(np.argmax(raw_hit[lo:m + r + 1]))
        stands_out = raw_hit[g] > 1.3 * max(near(raw_hit, int(a)), near(raw_hit, int(b)))
        if hit[g] > 0.7 and stands_out and gate[g] > 0.5:
            cands.append((int(g), False, 0.15, "offbeat"))
    cands.sort()

    cf = np.array([c[0] for c in cands])
    bar = np.array([c[1] for c in cands])
    metric = np.array([c[2] for c in cands])
    kind = [c[3] for c in cands]

    w = int(0.12 * FR)
    change = 0.5 * C[np.clip(cf, 0, n - 1)]
    event_peak = np.full(len(cf), -1)
    for f, p, h in events:
        j = int(np.argmin(np.abs(cf - f)))
        if abs(cf[j] - f) <= w and h > change[j]:
            change[j], event_peak[j] = h, p

    hitv = np.array([near(hit, f) for f in cf])
    accent = np.array([max(near(A["accent"], f, 3), near(A["parts"]["impact"], f, 4)) for f in cf])
    # a clear accent is a musical moment in itself (stab, crash, big hit), even with no section change
    from_accent = 0.9 * accent > change
    change = np.maximum(change, 0.9 * accent)
    event_peak[from_accent] = -1
    g = gate[np.clip(cf, 0, n - 1)]
    confidence = np.clip(hitv / 0.35, 0, 1) * g
    raw = (0.45 * change + 0.20 * hitv + 0.15 * metric + 0.10 * accent + 0.10 * confidence)
    raw = raw + 0.6 * np.clip(change - 0.6, 0, None)   # clearly significant moments dominate
    raw = raw * (0.3 + 0.7 * g)                         # weak triggers in near-silence count little
    value = raw - np.median(raw) + bar_weight * bar
    kind = [("accent" if (fa and k == "beat") else ("offbeat" if k == "offbeat" else k))
            for fa, k in zip(from_accent, kind)]
    return cf, bar, change, value, event_peak, kind


def plan(times, value, duration, gmin, gmax, target=None, spacing_weight=0.0):
    """Pick beats so every shot (start -> 1st cut, cut -> cut, last cut -> end)
    is gmin..gmax seconds long, maximising the total musical value.
    Optional target(t) gives the preferred shot length around time t (shorter in
    energetic music, longer in calm music); shots far from it cost spacing_weight.
    Dynamic programming, so the result is the best possible under the rules."""
    n = len(times)
    span = max(gmax - gmin, 1e-6)

    def cost(a, b):
        if target is None or spacing_weight <= 0:
            return 0.0
        return spacing_weight * (((b - a) - target((a + b) / 2)) / span) ** 2

    best = np.full(n, -np.inf)
    prev = np.full(n, -2)
    for j in range(n):
        if gmin <= times[j] <= gmax:                 # first shot starts at 0
            best[j], prev[j] = value[j] - cost(0.0, times[j]), -1
        for i in range(j - 1, -1, -1):
            g = times[j] - times[i]
            if g > gmax:
                break
            if g >= gmin and best[i] > -np.inf:
                cand = best[i] + value[j] - cost(times[i], times[j])
                if cand > best[j]:
                    best[j], prev[j] = cand, i

    tail = duration - times
    ends = [j for j in range(n) if gmin <= tail[j] <= gmax and best[j] > -np.inf]
    strict = bool(ends)
    if not ends:   # song end doesn't fit the rule exactly: allow a shorter/longer last shot
        ends = [j for j in range(n) if best[j] > -np.inf and tail[j] <= gmax + gmin]
    if not ends:
        return [], False
    j = max(ends, key=lambda k: best[k])
    path = []
    while j >= 0:
        path.append(j)
        j = prev[j]
    return path[::-1], strict


# ---------------------------------------------------------------- repeating sequences
SEQ_KEYS = ("low_hit", "mid_hit", "high_hit", "bass_note")
SEQ_NAMES = {"low_hit": "low-frequency hit (kick or low note)",
             "mid_hit": "sharp hit (snare/clap-like)",
             "high_hit": "high-frequency fill (hi-hat/percussion-like)",
             "bass_note": "bass note",
             "all": "accent"}


def music_bounds(A):
    """First/last frame where music is clearly playing (skips silence and fade-out tails)."""
    on = np.where(A["gate"] > 0.9)[0]
    if len(on) == 0:
        return 0, 0
    return int(on[0]), int(on[-1])


def seq_features(A):
    """One row per sound layer: how strongly that layer hits at each moment (0..~1)."""
    g = A["gate"]
    rows = []
    for k in SEQ_KEYS + ("all",):
        x = (A["onset"] if k == "all" else A["raw_layers"][k]) * g
        pos = x[x > 0]
        x = np.clip(x / (np.percentile(pos, 99) if len(pos) else 1.0), 0, 1.5)
        rows.append(gaussian_filter1d(x, 2))          # tolerate +-20 ms timing jitter
    return np.vstack(rows)


def find_period(F, a, b, beat, min_p=1.0, max_p=12.0):
    """Length of the repeating sequence (loop) between frames a..b.
    Only musical lengths are considered (2, 3, 4, 6, 8, 12, 16 beats, whole bars
    preferred); each is checked by autocorrelation of every sound layer.
    Returns (period_frames, confidence) or (None, 0)."""
    X = F[:, a:b] - F[:, a:b].mean(axis=1, keepdims=True)
    n = X.shape[1]
    ac = np.zeros(n)
    used = 0
    for r in X:
        if r.std() < 1e-6:
            continue
        f = np.fft.rfft(r, 2 * n)
        c = np.fft.irfft(np.abs(f) ** 2)[:n]
        ac += c / (n - np.arange(n)) / (c[0] / n + 1e-12)    # unbiased, normalised
        used += 1
    if not used or beat <= 0:
        return None, 0.0
    ac /= used
    cands = []
    for k, prior in ((4, 1.0), (8, 1.0), (2, 0.9), (16, 0.95), (3, 0.8), (6, 0.85), (12, 0.85)):
        lag = k * beat
        if lag < min_p * FR or lag > max_p * FR or lag >= n / 2 - 3:
            continue
        r = max(2, int(0.03 * lag))                 # allow 3 % tempo-estimate error
        i0, i1 = int(lag) - r, int(lag) + r + 1
        i = i0 + int(np.argmax(ac[i0:i1]))
        y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
        den = y0 - 2 * y1 + y2
        d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den != 0 else 0.0
        cands.append((i + d, float(y1), float(y1) * prior))
    if not cands:
        return None, 0.0
    top = max(c[2] for c in cands)
    if top <= 0:
        return None, 0.0
    # the loop itself, not a multiple of it: the shortest one that repeats almost as well
    lag, conf, _ = min((c for c in cands if c[2] >= 0.9 * top), key=lambda c: c[0])
    return lag, conf


def find_sequence(A, F, a, b, beat, bar_frames=(), min_p=1.0, max_p=12.0):
    """Detect the repeating sequence in frames a..b and its most distinctive moment.
    Returns a dict, or None if the music there does not clearly repeat."""
    P, conf = find_period(F, a, b, beat, min_p, max_p)
    if P is None or conf < 0.35 or (b - a) < 2.5 * P:
        return None
    L = int(round(P))
    cycles = int((b - a) // P)
    blocks = np.stack([F[:, int(round(a + c * P)):int(round(a + c * P)) + L] for c in range(cycles)])
    profile = np.median(blocks, axis=0)          # what happens in EVERY repetition

    # which moment of the loop stands out: a layer whose activity is concentrated in
    # one short part of the loop (a fill, a stab, a snare) says more than a layer
    # that plays on every beat. Hits closer than ~0.3 s count as one group (a fill).
    win = max(3, int(0.6 * FR))
    kern = np.ones(int(0.3 * FR))
    score = np.zeros(L)
    wts = np.zeros(5)
    gpeak = np.zeros(5, dtype=int)
    for i, k in enumerate(SEQ_KEYS + ("all",)):
        prof = profile[i]
        if prof.max() < 0.05:
            continue
        circ = np.r_[prof, prof]
        conc = np.convolve(circ, np.ones(win), mode="valid")[:L].max() / (prof.sum() + 1e-9)
        conc = float(np.clip(conc, 0.0, 1.0))
        group = np.convolve(circ, kern, mode="same")[:L]   # activity of hit groups
        group = group / (group.max() + 1e-9)
        if k == "all":
            presence = 0.5
        else:   # level the layer reaches when it plays (not the silence between hits)
            lvl = np.percentile(A["band_level"][k][a:b], 95)
            presence = float(np.clip((lvl + 60) / 30, 0.3, 1.0))
        wts[i] = presence * conc ** 2
        gpeak[i] = int(np.argmax(group))
        score += wts[i] * group
    peak = int(np.argmax(score))
    uniform = False
    # start of that group: its first clear hit (a fill is cut where it begins)
    # (counting only the layers whose own activity peaks in this group, not e.g. a kick just before it)
    circ_d = lambda x, y: min((x - y) % L, (y - x) % L)
    defining = [i for i in range(5) if wts[i] > 0 and circ_d(gpeak[i], peak) <= int(0.3 * FR)]
    if not defining:
        defining = [int(np.argmax(wts))]
    hits = sum(wts[i] * profile[i] / (profile[i].max() + 1e-9) for i in defining)
    phi = peak
    back = int(0.45 * FR)
    cand = [(peak - d) % L for d in range(back, -int(0.15 * FR), -1)]
    strong = [q for q in cand if hits[q] >= 0.5 * max(hits[c] for c in cand)
              and hits[q] >= hits[(q - 1) % L] and hits[q] >= hits[(q + 1) % L]]
    if strong:
        phi = strong[0]
    # nothing in the loop stands out (e.g. a kick on every beat): use the bar start
    if score.max() < 1.6 * (score.mean() + 1e-9) and len(bar_frames):
        bf = np.asarray(bar_frames)
        bf = bf[(bf >= a) & (bf < b)]
        if len(bf):
            phi = int(round((bf[0] - a) % P)) % L
            defining = [int(np.argmax(wts))]
            uniform = True

    # name the sequence point after the layer that defines it most
    layer = (SEQ_KEYS + ("all",))[max(defining, key=lambda i: wts[i])]

    # does each repetition really contain that moment? (intro / fills / breaks may not)
    half = int(0.4 * FR)
    idx = (phi + np.arange(-half, half + 1)) % L
    tmpl = profile[:, idx].ravel()
    w = (score / (score.max() + 1e-9))[idx]
    matches = []
    for c in range(int(np.ceil((b - a - phi) / P)) + 1):
        f0 = int(round(a + c * P + phi))
        if f0 >= b:
            break
        lo_, hi_ = f0 - half, f0 + half + 1
        if lo_ < 0 or hi_ > F.shape[1]:
            continue
        win = F[:, lo_:hi_].ravel()
        sim = float(np.dot(win, tmpl) / (np.linalg.norm(win) * np.linalg.norm(tmpl) + 1e-9))
        matches.append((f0, sim))
    return dict(period=P, confidence=conf, phase=phi, layer=layer,
                matches=matches, weights=w, uniform=uniform)


def sequence_cuts(A, F, seq, min_gap, max_gap, match_thr=0.6):
    """Cut on the sequence point of every repetition that really matches,
    keeping cuts min_gap apart (every 2nd/3rd loop if the loop is short)."""
    P = seq["period"]
    # a loop that is (within 3 %) as long as min-gap still gets a cut every repetition
    k = max(1, int(np.ceil(0.97 * min_gap * FR / P)))
    good = [(f, s) for f, s in seq["matches"] if s >= match_thr]
    if not good:
        return []
    # choose which repetitions to use when skipping (k > 1): the best-matching set
    best_off = max(range(k), key=lambda o: sum(s for i, (f, s) in enumerate(seq["matches"])
                                               if i % k == o and s >= match_thr))
    picked = [(f, s) for i, (f, s) in enumerate(seq["matches"]) if i % k == best_off and s >= match_thr]
    # land exactly on the hit of the layer that defines the sequence point
    row = (SEQ_KEYS + ("all",)).index(seq["layer"])
    snap = int(0.08 * FR)
    cuts = []
    for f, s in picked:
        lo = max(0, f - snap)
        seg = F[row, lo:f + snap + 1] + 0.5 * F[4, lo:f + snap + 1]
        cuts.append((lo + int(np.argmax(seg)), s))
    return cuts


# ---------------------------------------------------------------- shared pipeline
def build(path, bar_weight):
    """Full analysis of one track: features, musical events, beat grid and
    scored cut candidates. Returns (A, events, frames, downbeat, scored) where
    scored is the tuple from score_candidates, or None if no beat was found."""
    A = analyze(path)
    events = find_events(A)
    frames, downbeat = beat_grid(A, events)
    if len(frames) < 2:
        return A, events, frames, downbeat, None
    events = snap_stops_to_grid(A, events, frames)
    scored = score_candidates(A, frames, downbeat, events, bar_weight)
    return A, events, frames, downbeat, scored


def describe(j, A, scored):
    """Reason + suggestion for candidate j (beat / accent / event)."""
    cand, bar, change, value, event_peak, kind = scored
    big = np.percentile(change, 85)
    f = int(cand[j])
    if change[j] >= big and event_peak[j] >= 0:
        return explain(int(event_peak[j]), A)
    if kind[j] == "offbeat":
        return "syncopated accent (off-beat hit)", "hard cut on the hit"
    if kind[j] == "accent" and change[j] >= big:
        r = slice(max(0, f - 3), f + 4)
        if A["parts"]["harmony"][max(0, f - 10):f + 11].max() > 0.5:
            return "new chord struck (harmony change)", "smooth slide, dissolve or light leak"
        if A["layers"]["mid_hit"][r].max() > A["layers"]["low_hit"][r].max():
            return "sharp accent (snare/clap-like hit)", "hard cut, shake or flash"
        return "big low-frequency accent (kick/bass-like hit)", "hard cut, shake or flash"
    return ("bar start (keeps the rhythm)" if bar[j] else "on the beat (keeps the rhythm)",
            "hard cut on the beat")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- command: cuts (original behaviour)
def run_cuts(args):
    A, events, frames, downbeat, scored = build(args.audio, args.bar_weight)
    C = A["combined"]
    if scored is None:
        print("Could not find a beat in this audio.")
        return
    times = frames / FR
    cand, bar, change, value, event_peak, kind = scored
    ctimes = cand / FR

    def label(j):
        return describe(j, A, scored)

    def on_bar(f):
        j = int(np.argmin(np.abs(frames - f)))
        return bool(downbeat[j] and abs(frames[j] - f) <= 0.07 * FR)

    # ---- 1) repeating sequences (primary): split the music at clear section changes,
    #         find each section's loop and cut on its sequence point every repetition
    F = seq_features(A)
    m0, m1 = music_bounds(A)
    gmin, gmax = args.min_gap * FR, args.max_gap * FR
    section_events = sorted((f, p) for f, p, h in events if h >= 0.75 and m0 + gmin <= f <= m1 - gmin)
    bounds = [f for f, _ in section_events]
    edges = [m0] + bounds + [m1]
    cuts = []          # (frame, strength 0-100, reason, suggestion, priority)
    seq_info = []
    for si, (s0, s1) in enumerate(zip(edges[:-1], edges[1:])):
        seq = (find_sequence(A, F, s0, s1, beat=float(np.median(np.diff(frames))),
                             bar_frames=frames[downbeat]) if s1 - s0 > 2 * FR else None)
        if seq is None:
            continue
        P = seq["period"]
        seq_info.append((s0, s1, P, seq["confidence"]))
        point = "bar start" if seq["uniform"] else SEQ_NAMES[seq["layer"]]
        what = f"repeating sequence: {point} (repeats every {P / FR:.2f} s)"
        for f, sim in sequence_cuts(A, F, seq, args.min_gap, args.max_gap):
            if si == 0 and f < m0 + P * 0.98:
                continue   # let the first slide play one full repetition of the sequence
            cuts.append((f, int(round(100 * sim)), what, "hard cut on the sequence point", 1))
    for f, pk in section_events:   # where one section's sequence ends and the next begins
        reason, sugg = explain(int(pk), A)
        cuts.append((int(f), 100, reason, sugg, 2))

    if seq_info:
        # keep cuts at least min-gap apart (section changes win over sequence points)
        cuts.sort()
        kept = []
        for c in cuts:
            if kept and c[0] - kept[-1][0] < 0.97 * gmin:
                if c[4] > kept[-1][4]:
                    kept[-1] = c
                continue
            if c[0] < gmin:
                continue
            kept.append(c)
        cuts = kept

        # ---- 2) where no sequence point is available for longer than max-gap
        #         (intro, transitions, very long loops), use the best musical beat
        while True:
            pts = [0] + [c[0] for c in cuts] + [m1]
            added = False
            for a_, b_ in zip(pts[:-1], pts[1:]):
                if b_ - a_ <= gmax:
                    continue
                ok = [j for j in range(len(cand)) if a_ + gmin <= cand[j] <= b_ - gmin]
                if not ok:
                    continue
                # prefer a beat that leaves no remaining gap over max-gap, then musical value
                j = max(ok, key=lambda j: (cand[j] - a_ <= gmax) + (b_ - cand[j] <= gmax) + value[j])
                reason, sugg = label(j)
                cuts.append((int(cand[j]), int(round(100 * change[j] / (change.max() + 1e-9))),
                             reason, sugg, 0))
                cuts.sort()
                added = True
                break
            if not added:
                break
        # the last slide should not be shorter than min-gap either
        while len(cuts) > 1 and A["duration"] * FR - cuts[-1][0] < gmin \
                and A["duration"] * FR - cuts[-2][0] <= max(gmax, m1 - cuts[-2][0]):
            cuts.pop()
        chosen = [(c[0] / FR, c[1], on_bar(c[0]), c[2], c[3]) for c in cuts]
        tail = A["duration"] - (chosen[-1][0] if chosen else 0)
        strict = args.min_gap <= tail <= args.max_gap
    else:
        # ---- fallback: no repeating sequence found -> best musical beats (previous method)
        energy, n_frames = A["energy"], len(A["energy"])
        target = lambda t: args.max_gap - (args.max_gap - args.min_gap) * float(
            energy[min(n_frames - 1, max(0, int(t * FR)))])
        spacing_weight = 1.2 * (np.percentile(value, 90) - np.percentile(value, 10))
        path, strict = plan(ctimes, value, A["duration"], args.min_gap, args.max_gap,
                            target=target, spacing_weight=spacing_weight)
        if not path:
            print("Song is too short for these gap settings.")
            return
        chosen = []
        for j in path:
            reason, sugg = label(j)
            chosen.append((float(ctimes[j]), int(round(100 * change[j] / (change.max() + 1e-9))),
                           bool(bar[j]), reason, sugg))
    if not chosen:
        print("Song is too short for these gap settings.")
        return

    transitions, last = [], 0.0
    for t, strength, is_bar, reason, suggestion in chosen:
        transitions.append({
            "time": round(t, 3),
            "timecode": fmt(t),
            "shot_length": round(t - last, 2),
            "strength": strength,
            "on_bar_start": is_bar,
            "what_happens": reason,
            "suggested_transition": suggestion,
        })
        last = t
    final_shot = A["duration"] - last

    base = os.path.splitext(args.audio)[0]
    with open(base + ".transitions.json", "w") as fh:
        json.dump({"audio": args.audio, "duration": round(A["duration"], 3),
                   "tempo_bpm": round(A["tempo"], 1), "min_gap": args.min_gap,
                   "max_gap": args.max_gap, "final_shot_length": round(final_shot, 2),
                   "transitions": transitions}, fh, indent=2)
    with open(base + ".transitions.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=transitions[0].keys())
        wr.writeheader()
        wr.writerows(transitions)

    print(f"Duration {fmt(A['duration'])} | ~{A['tempo']:.0f} BPM | "
          f"{len(transitions)} transitions | shots {args.min_gap:g}-{args.max_gap:g} s")
    for s0, s1, P, conf in seq_info:
        print(f"Repeating sequence {fmt(s0 / FR)}-{fmt(s1 / FR)}: every {P / FR:.2f} s "
              f"(~{P / FR / (60.0 / max(A['tempo'], 1)):.0f} beats, confidence {conf:.2f})")
    if not seq_info:
        print("No clearly repeating sequence found - using the strongest musical beats.")
    print(f"\n{'#':>3}  {'time':>9}  {'shot':>5}  {'str':>3}  {'bar':>3}  what happens")
    for i, t in enumerate(transitions, 1):
        print(f"{i:>3}  {t['timecode']:>9}  {t['shot_length']:>5.2f}  {t['strength']:>3}  "
              f"{'yes' if t['on_bar_start'] else '':>3}  {t['what_happens']}")
    print(f"\nLast shot (to end of song): {final_shot:.2f} s")
    if not strict:
        music_end = music_bounds(A)[1] / FR
        if final_shot > args.max_gap and A["duration"] - music_end > 0.5:
            print(f"Note: the music ends at {fmt(music_end)} and then fades/silence follows, "
                  "so no cut is placed in the tail; the last slide simply runs to the end.")
        else:
            print("Note: the song's ending didn't allow a last shot inside the range; "
                  "it's the closest possible.")

    if args.clicks:
        clicks = librosa.clicks(times=[t["time"] for t in transitions], sr=SR,
                                length=len(A["y"]), click_freq=2000, click_duration=0.08)
        sf.write(base + ".clicks.wav", np.clip(0.7 * A["y"] + 0.7 * clicks, -1, 1), SR)
        print(f"Listen to check: {base}.clicks.wav")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from librosa import display as lr_display
        except ImportError:
            print("Install matplotlib for --plot")
            return
        tt = np.arange(len(C)) / FR
        fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        lr_display.waveshow(A["y"], sr=SR, ax=ax[0], alpha=0.6)
        ax[0].set_title("waveform + transitions (red) on the beat grid (grey)")
        ax[1].plot(tt, C, label="music change score")
        for bt in times:
            ax[1].axvline(bt, c="gray", alpha=0.15, lw=0.8)
        for t in transitions:
            for a in ax:
                a.axvline(t["time"], c="red", alpha=0.8)
        ax[1].legend()
        ax[1].set_xlabel("seconds")
        fig.tight_layout()
        fig.savefig(base + ".analysis.png", dpi=110)
        print(f"Graph: {base}.analysis.png")


# ---------------------------------------------------------------- command: analyze
def make_analysis(path):
    """Analyse a track once; everything sync needs, as plain JSON-ready data.
    Candidate values are stored WITHOUT the bar-start bonus (sync adds its own)."""
    A, events, frames, downbeat, scored = build(path, bar_weight=0.0)
    m0, m1 = music_bounds(A)
    out = {
        "analysis_version": ANALYSIS_VERSION,
        "librosa_version": librosa.__version__,
        "audio": os.path.abspath(path),
        "audio_sha256": file_sha256(path),
        "duration": round(float(A["duration"]), 4),
        "music_start": round(m0 / FR, 4),
        "music_end": round(m1 / FR, 4),
        "tempo_bpm": round(float(A["tempo"]), 2),
        "has_beat": scored is not None,
        "beats": [],
        "candidates": [],
    }
    if scored is None:
        return out
    out["beat_period_s"] = round(float(np.median(np.diff(frames))) / FR, 4)
    out["beats"] = [{"time": round(float(f) / FR, 4), "bar_start": bool(b)}
                    for f, b in zip(frames, downbeat)]
    cand, bar, change, value, event_peak, kind = scored
    for j in range(len(cand)):
        what, _ = describe(j, A, scored)
        out["candidates"].append({
            "time": round(float(cand[j]) / FR, 4),
            "bar_start": bool(bar[j]),
            "kind": kind[j],
            "value": round(float(value[j]), 5),
            "change": round(float(change[j]), 5),
            "what_happens": what,
        })
    return out


def run_analyze(args):
    an = make_analysis(args.audio)
    out = os.path.splitext(args.audio)[0] + ".analysis.json"
    with open(out, "w") as fh:
        json.dump(an, fh, indent=2)
    print(f"Duration {fmt(an['duration'])} | ~{an['tempo_bpm']:.0f} BPM | "
          f"{len(an['beats'])} beats | {len(an['candidates'])} cut candidates")
    print(f"Analysis: {out}")


# ---------------------------------------------------------------- command: sync
def snap_durations(times, values, bars, targets, gmin, gmax, final_min,
                   length_weight, drift, end_on_bar):
    """Choose exactly one candidate per planned boundary (N shots -> N-1 cuts plus
    the tour end) maximising musical value minus a cost for straying from the
    planned shot lengths. Hard rules: every shot gmin..gmax s, last shot >=
    final_min s, every boundary within `drift` s of its planned position (so the
    total stays close to the plan), the tour end on a bar start if end_on_bar.
    Dynamic programming over (boundary k, candidate j). Returns indices or None."""
    N, n = len(targets), len(times)
    D = np.cumsum(targets)
    best = np.full((N, n), -np.inf)
    prev = np.full((N, n), -2, dtype=int)
    for k in range(N):
        lo = max(gmin, final_min) if k == N - 1 else gmin
        js = np.where(np.abs(times - D[k]) <= drift)[0]
        if k == N - 1 and end_on_bar:
            js = js[bars[js]]
        for j in js:
            if k == 0:
                L = times[j]
                if lo - 1e-6 <= L <= gmax + 1e-6:
                    best[0, j] = values[j] - length_weight * ((L - targets[0]) / targets[0]) ** 2
                    prev[0, j] = -1
                continue
            iis = np.where(np.isfinite(best[k - 1]))[0]
            for i in iis:
                L = times[j] - times[i]
                if not (lo - 1e-6 <= L <= gmax + 1e-6):
                    continue
                cand = best[k - 1, i] + values[j] - length_weight * ((L - targets[k]) / targets[k]) ** 2
                if cand > best[k, j]:
                    best[k, j], prev[k, j] = cand, i
    if not np.isfinite(best[N - 1]).any():
        return None
    j = int(np.argmax(best[N - 1]))
    path = []
    for k in range(N - 1, -1, -1):
        path.append(j)
        j = int(prev[k, j])
    return path[::-1]


def frame_align(boundaries_s, fps):
    """Boundary times -> exact video frames -> cumulative integer centiseconds
    (the Shot Plan rule: shot k spans round(S_k*fps)..round(S_{k+1}*fps)).
    Returns (durations_s, boundary_frames) and asserts the round trip lands
    every cut on the intended frame."""
    frames = np.round(np.asarray(boundaries_s) * fps).astype(int)
    for k in range(1, len(frames)):                 # keep frames strictly increasing
        frames[k] = max(frames[k], frames[k - 1] + 1)
    cum_cs = np.round(frames * 100.0 / fps).astype(int)
    back = np.round(cum_cs / 100.0 * fps).astype(int)
    assert np.array_equal(back, frames), "frame alignment round trip failed"
    durations = np.diff(np.concatenate([[0], cum_cs])) / 100.0
    return [round(float(d), 2) for d in durations], frames.tolist()


def load_targets(args):
    """Planned shot lengths from --targets or a Shot Plan (--plan)."""
    if args.plan:
        with open(args.plan) as fh:
            plan_doc = json.load(fh)
        shots = sorted(plan_doc["shots"], key=lambda s: s["order"])
        return [float(s["duration_s"]) for s in shots], plan_doc, shots
    return [float(x) for x in args.targets.split(",") if x.strip()], None, None


def canonical_hash(plan_doc):
    """Shot Plan v1 hash: sha256 of canonical JSON (sorted keys, compact
    separators, UTF-8) of the plan without provenance.plan_hash."""
    doc = json.loads(json.dumps(plan_doc))
    doc.get("provenance", {}).pop("plan_hash", None)
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_tour_audio(src, end_s, fade_s, out_path):
    """The music from 0 to end_s at the source quality, with an equal-power
    fade-out over the last fade_s seconds (the tour ends on a bar line)."""
    y, sr = librosa.load(src, sr=None, mono=False)
    y = np.atleast_2d(y)
    n = int(round(end_s * sr))
    if y.shape[1] < n:
        y = np.pad(y, ((0, 0), (0, n - y.shape[1])))
    y = y[:, :n].copy()
    m = min(n, int(round(fade_s * sr)))
    if m > 0:
        y[:, n - m:] *= np.cos(np.linspace(0, np.pi / 2, m)) ** 2
    sf.write(out_path, y.T, sr)


def run_sync(args):
    src = args.audio
    if src.lower().endswith(".json"):
        with open(src) as fh:
            an = json.load(fh)
        audio_path = args.audio_file or an["audio"]
    else:
        audio_path = src
        cache = os.path.splitext(src)[0] + ".analysis.json"
        an = None
        if os.path.exists(cache):
            with open(cache) as fh:
                an = json.load(fh)
            if (an.get("analysis_version") != ANALYSIS_VERSION
                    or an.get("audio_sha256") != file_sha256(src)):
                an = None                                   # stale cache
        if an is None:
            an = make_analysis(src)
            with open(cache, "w") as fh:
                json.dump(an, fh, indent=2)

    targets, plan_doc, shots = load_targets(args)
    if len(targets) < 1:
        sys.exit("No planned shot lengths given (use --targets or --plan).")
    N = len(targets)
    if any(t <= 0 for t in targets):
        sys.exit("Planned shot lengths must be positive.")
    planned_total = float(sum(targets))

    path, reason, relaxed = None, None, False
    if not an["has_beat"] or len(an["candidates"]) < N:
        reason = "no_beat"
    elif an["music_end"] < planned_total - args.max_drift:
        reason = "music_too_short"
    else:
        cands = an["candidates"]
        times = np.array([c["time"] for c in cands])
        bars = np.array([c["bar_start"] for c in cands])
        values = np.array([c["value"] + args.bar_weight * c["bar_start"]
                           - args.offbeat_penalty * (c["kind"] == "offbeat") for c in cands])
        usable = times <= an["music_end"] + 1e-6           # never cut in a fade-out tail
        times, bars, values = times[usable], bars[usable], values[usable]
        idx_map = np.where(usable)[0]
        # relax step by step only if the strict rules leave no solution
        attempts = [(args.max_drift, True), (2 * args.max_drift, True),
                    (2 * args.max_drift, False), (4 * args.max_drift, False)]
        for drift, end_bar in attempts:
            if end_bar and not args.end_on_bar:
                continue
            path = snap_durations(times, values, bars, targets, args.min_gap, args.max_gap,
                                  args.final_min, args.length_weight, drift, end_bar)
            if path is not None:
                relaxed = (drift, end_bar) != (args.max_drift, args.end_on_bar)
                path = [int(idx_map[j]) for j in path]
                break
        if path is None:
            reason = "no_solution_within_rules"

    if path is None:
        # fall back to the planned lengths, frame-aligned, unsynced
        boundaries = np.cumsum(targets)
        durations, bframes = frame_align(boundaries, args.fps)
        synced, cut_info = False, [None] * N
    else:
        boundaries = [an["candidates"][j]["time"] for j in path]
        durations, bframes = frame_align(boundaries, args.fps)
        synced, cut_info = True, [an["candidates"][j] for j in path]
    end_s = round(sum(durations), 2)

    report = []
    for k in range(N):
        c = cut_info[k]
        report.append({
            "shot": k + 1,
            "shot_id": shots[k].get("shot_id") if shots else None,
            "role": shots[k].get("role") if shots else None,
            "planned_s": round(targets[k], 2),
            "synced_s": durations[k],
            "delta_s": round(durations[k] - targets[k], 2),
            "ends_at_s": round(bframes[k] / args.fps, 3),
            "ends_at_frame": bframes[k],
            "on_bar_start": bool(c["bar_start"]) if c else None,
            "what_happens": (c["what_happens"] if k < N - 1 else "tour end") if c else None,
        })
    music = {
        "track": os.path.basename(audio_path),
        "track_sha256": an["audio_sha256"],
        "analysis_version": an["analysis_version"],
        "start_s": 0.0,
        "end_s": end_s,
        "fade_out_s": args.fade_out,
        "tempo_bpm": an["tempo_bpm"],
        "synced": synced,
    }
    if not synced:
        music["fallback_reason"] = reason
    result = {"music": music, "planned_total_s": round(planned_total, 2), "total_s": end_s,
              "fps": args.fps, "durations_s": durations, "shots": report}
    if synced and relaxed:
        result["note"] = "strict rules had no solution; drift and/or end-on-bar were relaxed"

    base = os.path.splitext(src)[0]
    if base.endswith(".analysis"):
        base = base[:-len(".analysis")]
    with open(base + ".sync.json", "w") as fh:
        json.dump(result, fh, indent=2)

    if plan_doc is not None:
        by_id = {s["shot_id"]: d for s, d in zip(shots, durations)}
        for s in plan_doc["shots"]:
            s["duration_s"] = by_id[s["shot_id"]]
        plan_doc["total_duration_s"] = end_s
        plan_doc["music"] = music
        if not synced:
            plan_doc.setdefault("warnings", []).append(f"music_not_synced:{reason}")
        if "provenance" in plan_doc:
            plan_doc["provenance"]["plan_hash"] = canonical_hash(plan_doc)
        out_plan = os.path.splitext(args.plan)[0] + ".synced.json"
        with open(out_plan, "w") as fh:
            json.dump(plan_doc, fh, indent=2, ensure_ascii=False)
        print(f"Synced plan: {out_plan}")

    print(f"{'Synced' if synced else 'NOT synced (' + str(reason) + ')'} | {N} shots | "
          f"planned {planned_total:.2f} s -> {end_s:.2f} s | ~{an['tempo_bpm']:.0f} BPM | {args.fps:g} fps")
    print(f"\n{'#':>3}  {'plan':>5}  {'sync':>5}  {'delta':>6}  {'cut at':>9}  {'bar':>3}  what happens")
    for r in report:
        print(f"{r['shot']:>3}  {r['planned_s']:>5.2f}  {r['synced_s']:>5.2f}  {r['delta_s']:>+6.2f}  "
              f"{fmt(r['ends_at_s']):>9}  {('yes' if r['on_bar_start'] else '') if synced else '':>3}  "
              f"{r['what_happens'] or ''}")
    print(f"\nSync report: {base}.sync.json")

    if args.audio_out:
        write_tour_audio(audio_path, end_s, args.fade_out, args.audio_out)
        print(f"Tour music ({end_s:.2f} s, {args.fade_out:g} s fade-out): {args.audio_out}")
    if args.clicks:
        y, _ = librosa.load(audio_path, sr=SR, mono=True)
        y = np.pad(y, (0, max(0, int(end_s * SR) - len(y))))[:int(end_s * SR)]
        cut_times = [b / args.fps for b in bframes[:-1]]
        clicks = librosa.clicks(times=cut_times, sr=SR, length=len(y), click_freq=2000, click_duration=0.08)
        sf.write(base + ".sync_clicks.wav", np.clip(0.7 * y + 0.7 * clicks, -1, 1), SR)
        print(f"Listen to check: {base}.sync_clicks.wav")


# ---------------------------------------------------------------- main
COMMANDS = ("cuts", "analyze", "sync")


def main():
    argv = sys.argv[1:]
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["cuts"] + argv          # original usage: find_transitions.py song.mp3 ...
    ap = argparse.ArgumentParser(description="Beat-synced transition points for a video.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("cuts", help="pick transition points from the music alone (original behaviour)")
    pc.add_argument("audio")
    pc.add_argument("--min-gap", type=float, default=2.0, help="shortest allowed shot (s)")
    pc.add_argument("--max-gap", type=float, default=6.0, help="longest allowed shot (s)")
    pc.add_argument("--bar-weight", type=float, default=0.15,
                    help="extra preference for the first beat of a bar (0 = none)")
    pc.add_argument("--clicks", action="store_true", help="write song.clicks.wav to listen and check")
    pc.add_argument("--plot", action="store_true", help="write song.analysis.png (needs matplotlib)")

    pa = sub.add_parser("analyze", help="analyse a track once and store song.analysis.json")
    pa.add_argument("audio")

    ps = sub.add_parser("sync", help="snap a tour's planned shot lengths onto the music")
    ps.add_argument("audio", help="music file, or a song.analysis.json from 'analyze'")
    grp = ps.add_mutually_exclusive_group(required=True)
    grp.add_argument("--plan", help="Shot Plan JSON; its shots' duration_s are the planned lengths")
    grp.add_argument("--targets", help="planned shot lengths in seconds, comma-separated, in order")
    ps.add_argument("--audio-file", help="music file, if the input is an analysis JSON and the file moved")
    ps.add_argument("--min-gap", type=float, default=3.0, help="shortest allowed shot (s)")
    ps.add_argument("--max-gap", type=float, default=8.0, help="longest allowed shot (s)")
    ps.add_argument("--final-min", type=float, default=5.0, help="shortest allowed final shot (s)")
    ps.add_argument("--fps", type=float, default=30.0, help="video frame rate; every cut lands on a frame")
    ps.add_argument("--max-drift", type=float, default=1.5,
                    help="how far (s) any cut may move from its planned position")
    ps.add_argument("--length-weight", type=float, default=4.0,
                    help="cost of changing a shot's planned length (higher = keep lengths closer)")
    ps.add_argument("--bar-weight", type=float, default=0.3, help="preference for cuts on a bar start")
    ps.add_argument("--offbeat-penalty", type=float, default=0.1, help="penalty for off-beat cuts")
    ps.add_argument("--no-end-on-bar", dest="end_on_bar", action="store_false",
                    help="do not require the tour to end on a bar start")
    ps.add_argument("--fade-out", type=float, default=1.5, help="music fade-out at the tour end (s)")
    ps.add_argument("--audio-out", help="write the music trimmed to the tour with the fade-out")
    ps.add_argument("--clicks", action="store_true", help="write song.sync_clicks.wav to listen and check")

    args = ap.parse_args(argv)
    if args.cmd in ("cuts", "sync") and args.max_gap < args.min_gap:
        ap.error("--max-gap must be >= --min-gap")
    {"cuts": run_cuts, "analyze": run_analyze, "sync": run_sync}[args.cmd](args)


if __name__ == "__main__":
    main()
