"""Beat sync (plan 1.5): snap the planned shot lengths onto a music track's cut candidates.

The track is analysed ONCE, when the super admin uploads it (worker `music_analyze` job, the
analysis of find_transitions.py), and the analysis is stored with the track. The planner only
needs the stored JSON: this module is numpy only (no librosa).

Called by build_shot_plan after the shot lengths are allocated and BEFORE the cameras are planned,
so each camera move is fitted to its synced length (moves have minimum lengths).

Rules (from find_transitions.py `sync`, unchanged): exactly one cut per planned boundary; every
cut on a beat (bar starts and big musical moments preferred, off-beats penalised); each shot
within the planner's shot_min_s..shot_max_s and the last >= final_shot_min_s; every boundary
within music_max_drift s of its planned position; the tour ends on a bar start; every cut on an
exact video frame (the worker's timeline: boundary k at round(S_k * fps)). If the strict rules
have no solution they are relaxed step by step; if nothing fits, the planned lengths are kept
(warning music_not_synced:<reason>).

A track shorter than the tour is queued again after itself (product decision 2026-10-16). Each copy
is cut at the track's LOOP POINT - the last bar start before its music ends - and the next copy
starts there: tracks often end in seconds of fade / silence (sample.mp3: music ends at 21.3 s of
27 s), and queuing whole files put ~6 s of silence into the tour, with cuts landing in it. Joining
on a bar line also keeps the beat going across the join. The worker cuts the audio the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import PlannerConfig
from .schemas import MusicHook, MusicTrack, Punch

# v2 (worker modules/music/analysis.py) = v1 + whoosh candidates + rhythm_end; v3 = v2 + per
# candidate "hit" (drum hit strength, 1.0 = a typical beat of the track)
SUPPORTED_ANALYSIS_VERSIONS = ("1", "2", "3")


@dataclass
class SyncResult:
    durations: list[float]          # per shot, seconds (2 decimals, frame exact)
    music: MusicTrack
    warning: str | None
    hooks_hit: int = 0              # hooks (drops, whooshes, entries, stops) with a cut on them
    hooks_reachable: int = 0        # hooks inside the tour
    hook_cuts: list[int] | None = None          # boundaries k (end of shot k) that land on a hook
    accents: list[tuple[int, float]] | None = None  # (shot k, s into the shot): hooks with no cut


def snap_durations(
    times, values, bars, targets, gmin, gmax, final_min, length_weight, drift, end_on_bar,
    required=None, end_margin=0.5,
):
    """Choose exactly one candidate per planned boundary (N shots -> N-1 cuts plus the tour end)
    maximising musical value minus a cost for straying from the planned shot lengths. Hard rules:
    shot k between gmin[k]..gmax[k] s (scalars: every shot), the last shot >= final_min s, every
    boundary within `drift` s of its planned position, the tour end on a bar start if end_on_bar.
    required: hook clusters (arrays of candidate indices) that must each get a cut on one of
    their members - no two consecutive boundaries may enclose a whole cluster, and no cut may
    land inside a cluster except on a member (clusters ending within end_margin of the tour end
    are exempt). Dynamic programming over (boundary k, candidate j). Returns indices, or None."""
    N, n = len(targets), len(times)
    D = np.cumsum(targets)
    gmins = np.broadcast_to(np.asarray(gmin, dtype=float), (N,))
    gmaxs = np.broadcast_to(np.asarray(gmax, dtype=float), (N,))
    allowed = np.ones(n, dtype=bool)
    if required:
        first = np.array([times[c].min() for c in required])
        last = np.array([times[c].max() for c in required])
        for c, a, b in zip(required, first, last, strict=True):
            inside = (times > a + 1e-6) & (times < b - 1e-6)
            inside[c] = False
            allowed &= ~inside

    def skips(t0, t1, end):
        if not required:
            return False
        # the last shot: hooks up to the PLANNED end must be inside the tour - ending the tour
        # early is not a way around a hook
        limit = max(t1, float(D[-1])) - end_margin if end else t1
        return bool(np.any((first > t0 + 1e-6) & (last < limit - 1e-6)))

    best = np.full((N, n), -np.inf)
    prev = np.full((N, n), -2, dtype=int)
    for k in range(N):
        lo = max(gmins[k], final_min) if k == N - 1 else gmins[k]
        gmax = gmaxs[k]
        js = np.where((np.abs(times - D[k]) <= drift) & allowed)[0]
        if k == N - 1 and end_on_bar:
            js = js[bars[js]]
        end = k == N - 1
        for j in js:
            if k == 0:
                L = times[j]
                if lo - 1e-6 <= L <= gmax + 1e-6 and not skips(0.0, times[j], end):
                    best[0, j] = values[j] - length_weight * ((L - targets[0]) / targets[0]) ** 2
                    prev[0, j] = -1
                continue
            for i in np.where(np.isfinite(best[k - 1]))[0]:
                L = times[j] - times[i]
                if not (lo - 1e-6 <= L <= gmax + 1e-6) or skips(times[i], times[j], end):
                    continue
                cost = length_weight * ((L - targets[k]) / targets[k]) ** 2
                cand = best[k - 1, i] + values[j] - cost
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


def frame_align(boundaries_s, fps, lead_frames=0):
    """Boundary times -> exact video frames -> per-shot durations (centiseconds) whose cumulative
    sums round back to those frames (the worker: shot k spans round(S_k*fps)..round(S_k+1*fps)).
    Cuts (not the tour end) land lead_frames early: a cut right on a beat's onset reads as late."""
    frames = np.round(np.asarray(boundaries_s, dtype=float) * fps).astype(int)
    frames[:-1] -= lead_frames
    for k in range(1, len(frames)):                 # strictly increasing
        frames[k] = max(frames[k], frames[k - 1] + 1)
    cum_cs = np.round(frames * 100.0 / fps).astype(int)
    back = np.round(cum_cs / 100.0 * fps).astype(int)
    assert np.array_equal(back, frames), "frame alignment round trip failed"
    durations = np.diff(np.concatenate([[0], cum_cs])) / 100.0
    return [round(float(d), 2) for d in durations]


def music_stop(analysis: dict) -> float:
    """Where the music really stops: v2's rhythm_end (the last typical beat; tails and held notes
    excluded), else v1's music_end."""
    end = float(analysis["music_end"])
    return min(end, float(analysis.get("rhythm_end") or end))


def loop_point(analysis: dict) -> float:
    """Where a queued copy of the track ends and the next begins: the bar start nearest the point
    the music stops (keeps the beat across the join); that point itself when there are no bars."""
    stop = music_stop(analysis)
    end = float(analysis["music_end"])
    bars = [float(b["time"]) for b in analysis.get("beats") or [] if b.get("bar_start")]
    bars = [b for b in bars if 0.5 * stop <= b <= end + 1e-6]
    return round(min(bars, key=lambda b: abs(b - stop)), 4) if bars else stop


def queued_candidates(analysis: dict, needed_s: float):
    """Cut candidates of the track queued after itself (each copy cut at loop_point) until it
    covers needed_s. Returns (times, values, bars, kinds, usable_end_s, copies, loop_s, changes,
    local_times, hits) - local_times: the time within its copy of the track; hits: drum hit
    strength (v3; 1.0 = a typical beat, also when the analysis has none)."""
    end = music_stop(analysis)
    loop = loop_point(analysis)
    copies = 1
    if needed_s > end and loop > 0:          # the last copy plays to the music end
        copies = 1 + math.ceil((needed_s - end) / loop)
    cands = analysis.get("candidates") or []
    t_all, v_all, b_all, k_all, c_all, l_all, h_all = [], [], [], [], [], [], []
    for k in range(copies):
        last = k == copies - 1
        for c in cands:
            t = float(c["time"])
            if t < loop - 1e-6 or (last and t <= end + 1e-6):    # a copy plays [0, loop)
                t_all.append(t + k * loop)
                v_all.append(c["value"])
                b_all.append(bool(c["bar_start"]))
                k_all.append(c["kind"])
                c_all.append(float(c.get("change") or 0.0))
                l_all.append(t)
                h_all.append(float(c.get("hit") or 1.0))
    usable_end = (copies - 1) * loop + end                     # no cut in the last fade tail
    arr = np.array
    return (arr(t_all, dtype=float), arr(v_all, dtype=float), arr(b_all, dtype=bool), k_all,
            usable_end, copies, loop, arr(c_all, dtype=float), arr(l_all, dtype=float),
            arr(h_all, dtype=float))


def hook_mask(kinds, changes, local_times, cfg: PlannerConfig) -> np.ndarray:
    """The track's hooks - the moments a cut belongs on: whooshes, and big musical changes (a drop,
    a layer coming in or out, a new section, a stop). Not the first hit of the track (every song
    "starts"), which would sit on the tour's first frame anyway."""
    whoosh = np.array([k == "whoosh" for k in kinds], dtype=bool)
    big = whoosh | (changes >= cfg.music_hook_change)
    return big & (local_times >= cfg.music_hook_skip_start_s)


def hook_clusters(times, hooks, cfg: PlannerConfig) -> list[np.ndarray]:
    """Hooks closer than music_hook_merge_s are one musical moment (a riser and its drop 2.4 s
    later): one cut serves it, the other hooks get an accent. Returns candidate-index arrays."""
    idx = np.where(hooks)[0]
    idx = idx[np.argsort(times[idx], kind="stable")]
    out: list[list[int]] = []
    for j in idx:
        if out and times[j] - times[out[-1][-1]] < cfg.music_hook_merge_s:
            out[-1].append(int(j))
        else:
            out.append([int(j)])
    return [np.array(c) for c in out]


def length_bounds(targets: list[float], roles: list[str] | None, cfg: PlannerConfig, wide=False):
    """Per shot (min, max) synced length: hero shots keep near their planned length (they are the
    long establishing shots), detail shots flex more; within the planner's absolute limits.
    wide: every shot gets music_wide_bounds (the fallback that lets every hook get its cut)."""
    lo, hi = [], []
    for k, t in enumerate(targets):
        role = (roles[k] if roles else None) or "detail"
        f_lo, f_hi = (cfg.music_wide_bounds if wide
                      else cfg.music_role_bounds.get(role, cfg.music_role_bounds["detail"]))
        lo.append(max(cfg.music_shot_abs_min_s, f_lo * t))
        hi.append(min(cfg.music_shot_abs_max_s, f_hi * t))
    return np.array(lo), np.array(hi)


def ramp_warp(cfg: PlannerConfig) -> list[list[float]]:
    """A shot that cuts on a hook speeds INTO the cut ("velocity" edit): the move plays at a steady
    pace, then over the last music_ramp_share of the shot its pace rises linearly to
    music_ramp_peak x that pace at the cut. Normalised, so it holds for any synced length (the
    whole move still spans the shot). Points [output t, move t]."""
    R, p = cfg.music_ramp_share, cfg.music_ramp_peak
    r = 1.0 - R
    v0 = 1.0 / (r + R * (1.0 + p) / 2.0)

    def m(u):
        if u <= r:
            return v0 * u
        x = u - r
        return v0 * (r + x + (p - 1.0) * x * x / (2.0 * R))

    us = (0.0, r / 2, r, r + R / 2, 1.0)
    return [[round(u, 6), round(m(u), 6)] for u in us[:-1]] + [[1.0, 1.0]]


def time_warp(planned: float, synced: float, cfg: PlannerConfig,
              ramp: bool = False) -> list[list[float]] | None:
    """Shot timing for a synced length: the first part plays at normal pace (the first
    music_warp_hold of its real time), the rear speeds up (shorter shot) or slows down (longer)
    so the planned move ends exactly on the cut. Points [output t, move t], monotone; the worker
    interpolates them smoothly. None when the lengths match to within a frame. ramp: the shot
    cuts on a hook - ramp_warp instead (it retimes to any length by itself)."""
    if ramp:
        return ramp_warp(cfg)
    if abs(planned - synced) < 1.0 / cfg.fps:
        return None
    h = cfg.music_warp_hold * min(planned, synced)
    a_out, a_src = h / synced, h / planned
    return [[0.0, 0.0], [round(a_out / 2, 6), round(a_src / 2, 6)],
            [round(a_out, 6), round(a_src, 6)], [1.0, 1.0]]


def sync_to_music(
    targets: list[float], track: dict, cfg: PlannerConfig, roles: list[str] | None = None
) -> SyncResult:
    """track: {track_id, s3_uri, sha256, analysis} from the music library. roles: per shot
    ("hero" / "detail"), for the length bounds.

    Hooks first (user 2026-10-19, after "synced to some random beat"): every hook moment (hooks
    closer than music_hook_merge_s are one moment) MUST get a cut, within the role bounds, then
    within the wider music_wide_bounds; only when even that fails are hooks merely preferred
    (the bonus). A hook without a cut becomes an accent (a zoom punch in the shot it falls in)."""
    an = track["analysis"]
    planned = float(sum(targets))
    fps = cfg.fps
    lead = cfg.music_cut_lead_frames
    needed = planned + 4 * cfg.music_max_drift + cfg.music_fade_out_s
    times, vals, bars, kinds, usable_end, copies, loop, changes, local_t, hits = queued_candidates(
        an, needed
    )
    hooks = hook_mask(kinds, changes, local_t, cfg) if len(times) else np.zeros(0, bool)

    path, reason, required_ok, clusters = None, None, False, []
    if an.get("analysis_version") not in SUPPORTED_ANALYSIS_VERSIONS:
        reason = "analysis_version"
    elif not an.get("has_beat") or len(times) < len(targets):
        reason = "no_beat"
    else:
        # the beat the ear follows: the hardest drum hits (v3), not the bar line the beat tracker
        # guessed (sample4: the bar's loudest hit is beat 3, the "bar start" half a bar off)
        # (with hit strengths, the bar line earns nothing: it is only the tracker's guess)
        strong = np.clip(hits - 1.0, 0.0, cfg.music_hit_cap)
        bar_w = cfg.music_bar_weight_with_hits if np.any(hits != 1.0) else cfg.music_bar_weight
        values = (vals + bar_w * bars + cfg.music_hook_bonus * hooks
                  + cfg.music_hit_weight * strong
                  - cfg.music_offbeat_penalty * np.array([k == "offbeat" for k in kinds]))
        usable = times <= usable_end + 1e-6
        idx = np.where(usable)[0]
        t_u, v_u, b_u = times[usable], values[usable], bars[usable]
        # clusters in usable-index space; hooks before skip-in (the tour's first frame) excluded
        clusters = [c for c in hook_clusters(t_u, hooks[usable], cfg)
                    if t_u[c].min() > cfg.music_shot_abs_min_s - 1e-6]
        role_b = length_bounds(targets, roles, cfg)
        wide_b = length_bounds(targets, roles, cfg, wide=True)
        d = cfg.music_max_drift
        attempts = (
            (True, role_b, d, True), (True, role_b, 2 * d, True), (True, wide_b, 2 * d, True),
            (True, wide_b, 4 * d, True), (True, wide_b, 4 * d, False),
            (False, role_b, d, True), (False, role_b, 2 * d, True), (False, role_b, 2 * d, False),
            (False, role_b, 4 * d, False),
        )
        for req, (lo, hi), dr, end_bar in attempts:
            p = snap_durations(t_u, v_u, b_u, targets, lo, hi, cfg.final_shot_min_s,
                               cfg.music_length_weight, dr, end_bar,
                               required=clusters if req else None)
            if p is not None:
                path, required_ok = [int(idx[j]) for j in p], req
                break
        if path is None:
            reason = "no_solution_within_rules"

    if path is None:
        durations = frame_align(np.cumsum(targets), fps)
    else:
        durations = frame_align([times[j] for j in path], fps, lead)
    tour_end = float(np.sum(durations))
    cuts = np.cumsum(durations)[:-1]
    tol = (lead + 1) / fps

    # every hook inside the tour: a cut on it, or an accent in the shot it falls in
    report, hook_cuts, accents = [], [], []
    starts = np.concatenate([[0.0], cuts])
    hook_idx = np.where(hooks)[0] if len(times) else np.array([], dtype=int)
    merged = {int(j) for c in hook_clusters(times, hooks, cfg) if len(c) > 1 for j in c}
    for j in hook_idx[np.argsort(times[hook_idx], kind="stable")]:
        t = float(times[j])
        if not 0.5 < t < tour_end - 0.5:
            continue
        near = int(np.argmin(np.abs(cuts - t))) if len(cuts) else -1
        if near >= 0 and abs(cuts[near] - t) <= tol:
            hook_cuts.append(near)
            report.append(MusicHook(t=round(t, 3), kind="cut"))
            continue
        k = int(np.searchsorted(starts, t, side="right") - 1)
        at = round(t - float(starts[k]) - lead / fps, 3)
        why = ("merged" if int(j) in merged
               else "too_close_to_start" if t < cfg.music_shot_abs_min_s else "no_fit")
        # accents stay music_accent_min_gap_s from cuts and from each other: a run of fills ~1 s
        # apart (solarflex: 40-45 s) is one moment with one cut, not five punches
        last_acc = max((float(starts[a]) + b for a, b in accents), default=-1e9)
        near_cut = float(np.abs(cuts - t).min()) if len(cuts) else 1e9
        spaced = min(near_cut, t - last_acc) >= cfg.music_accent_min_gap_s
        if (spaced and at >= cfg.music_accent_edge_s
                and durations[k] - at >= cfg.music_accent_edge_s):
            accents.append((k, at))
            report.append(MusicHook(t=round(t, 3), kind="accent", reason=why))
        elif why == "merged":                          # its moment already has a cut / accent
            report.append(MusicHook(t=round(t, 3), kind="covered", reason=why))
        else:
            report.append(MusicHook(t=round(t, 3), kind="missed", reason=why))

    end_s = round(sum(durations), 2)
    music = MusicTrack(
        track_id=str(track["track_id"]), uri=track["s3_uri"], sha256=track["sha256"],
        analysis_version=str(an.get("analysis_version")),
        tempo_bpm=float(an.get("tempo_bpm") or 0.0),
        start_s=0.0, end_s=end_s, track_duration_s=float(an["duration"]), loops=copies, loop_s=loop,
        fade_out_s=cfg.music_fade_out_s, synced=path is not None, fallback_reason=reason,
        cuts_on_bar=(sum(bool(bars[j]) for j in path) if path is not None else None),
        cut_lead_frames=lead if path is not None else 0,
        hooks_required=required_ok, hooks=report or None,
    )
    warning = None if path is not None else f"music_not_synced:{reason}"
    if path is not None and not required_ok and clusters:
        warning = "music_hooks_not_all_cut"
    return SyncResult(durations, music, warning,
                      hooks_hit=sum(h.kind == "cut" for h in report), hooks_reachable=len(report),
                      hook_cuts=sorted(set(hook_cuts)), accents=accents)


def apply_sync(shots, synced: SyncResult, cfg: PlannerConfig) -> None:
    """Retime planned shots (cameras planned at their normal lengths) to the synced lengths: a
    shot cutting on a hook ramps into the cut, the others get the plain time warp; the shot after
    a hook cut opens on a zoom punch and every accent punches mid-shot."""
    hook_cuts = set(synced.hook_cuts or [])
    for k, (shot, d) in enumerate(zip(shots, synced.durations, strict=True)):
        warp = time_warp(shot.duration_s, d, cfg, ramp=k in hook_cuts)
        if warp is not None:
            shot.camera.time_warp = warp
        shot.duration_s = d
    punches: dict[int, list[Punch]] = {}
    for k in hook_cuts:
        if k + 1 < len(shots):
            punches.setdefault(k + 1, []).append(Punch(
                at_s=0.0, amount=cfg.music_punch_cut, attack_s=0.0,
                decay_s=cfg.music_punch_decay_s))
    for k, at in synced.accents or []:
        punches.setdefault(k, []).append(Punch(
            at_s=at, amount=cfg.music_punch_accent, attack_s=cfg.music_punch_attack_s,
            decay_s=cfg.music_punch_decay_s))
    for k, ps in punches.items():
        shots[k].camera.punches = sorted(ps, key=lambda p: p.at_s)
