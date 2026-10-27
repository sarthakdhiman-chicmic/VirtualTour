"""Beat sync v2 (2026-10-23): cut on the music's hero moments, punch the ones a cut cannot take.

v1-v3 ranked every beat by a local "value" and a drum-hit bonus; heroes were only the moments with
change >= 0.6, and any run of them closer than 3 s collapsed into one cut with the rest marked
"covered" and given nothing. Review: "we are just using a beat to transfer but not the hero beats",
"the corresponding beats are still being missed". v2 reads the analysis v4 structure
(tour_worker/music/structure.py: heroes on their exact beat, tier 1 = drops, tier 2 = entries,
dropouts, whooshes, big hits; phrases = the musical grid):

  value     a hero always outranks a beat (tier 1 >> tier 2 >> phrase start > bar > beat); plain
            beats only fill in where no hero is in reach
  cuts      heroes are taken strongest first; one closer than music_v2_hero_gap_s to a taken one
            is left for a punch (user 2026-10-23: "cut the biggest, punch the rest"); taken tier-1
            heroes MUST get their cut, taken tier-2 heroes are strongly preferred
  punches   every hero inside the tour without a cut gets a zoom punch sized by its tier, at least
            a beat apart and music_accent_edge_s from any cut
  start     the music may start later than 0:00 (user 2026-10-23): the track's start, its phrase
            starts and its changes (drops, dropouts, entries, sections) in the first
            music_v2_start_share are tried. Opening ON a change is rewarded; opening a few seconds
            BEFORE one is penalised (review: the tour opened on a loud tail and the music emptied
            out under the first shot)
  edits     the music may be trimmed to fit the tour (user 2026-10-23: "trim the music ... according
            to our music profile"): whole bars out, joined where the two sides sound alike
            (analysis bar_prints; as seamless as two neighbouring bars, music_v2_splice_max), at
            most music_v2_max_edits per tour, never through a drop -
              land   remove 1-4 bars before a drop so it falls on a planned cut
              reach  skip whole phrases of repetition so a drop past the tour's end comes inside it
            The music becomes segments of the track, joined with a centred crossfade.
  quality   every possible cut: how hard the sound hits there (strike: the strongest of low / mid
            / high onset) minus a stronger sound up to a beat BEFORE it (pre) - a cut a beat after a
            harder sound reads late (review: "delayed after the ting")
  end       the tour ends on a phrase start when it can, else a bar, else any beat

Everything else is v3's: snap_durations (the DP), frame-exact cuts with the cut lead, role length
bounds then the wide bounds, the 120 s cap upstream, fixed room order. A track without v4 heroes,
or a v2 sync that finds no solution, falls back to v3 (planner/music_sync.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tour_worker.music.kinds import KIND_RANK

LATE_RATIO = 1.15               # = tour_worker.music.analysis.LATE_RATIO (that module needs librosa)
STRIKE_WIN = (-0.10, 0.12)      # = analysis STRIKE_WIN / PRE_WIN: the shared accent windows
PRE_WIN = (-0.48, -0.10)


def peaks_window_max(pt: np.ndarray, pv: np.ndarray, t: float, win: tuple[float, float]) -> float:
    """The strongest accent peak within win of t (0 when none)."""
    a, b = np.searchsorted(pt, t + win[0], side="left"), np.searchsorted(pt, t + win[1], side="right")
    return float(pv[a:b].max(initial=0.0))


def played_peaks(an: dict, segments) -> tuple[np.ndarray, np.ndarray]:
    """The track's accent peaks (analysis v4) as the tour plays them: tour times, values."""
    t_all, v_all, base = [], [], 0.0
    peaks = an.get("accent_peaks") or []
    for a, b in segments:
        for t, v in peaks:
            if a - 1e-9 <= t < b:
                t_all.append(base + t - a)
                v_all.append(v)
        base += b - a
    o = np.argsort(t_all, kind="stable")
    return np.asarray(t_all, dtype=float)[o], np.asarray(v_all, dtype=float)[o]


CLARITY_SPAN_S = 2.0            # clarity: a cut's strike / the median accent within +-this
CLEAR = 2.2                     # at or above: the cut sits on a beat you can hear (loud sections ~3)
UNCLEAR = 1.5                   # at or below: nothing stands out (solarflex's soft intro: 1.45-1.7)


def clarity(pt: np.ndarray, pv: np.ndarray, t: float, strike: float) -> float:
    """How clearly the hit at t stands out from the music around it (strike / the median accent
    peak within CLARITY_SPAN_S). Review 2026-10-23: the first cuts sat in a soft intro at 1.45-1.7x
    - "you cant even tell what beat they are getting synced to"."""
    a, b = np.searchsorted(pt, t - CLARITY_SPAN_S), np.searchsorted(pt, t + CLARITY_SPAN_S)
    around = pv[a:b]
    return float(strike / (np.median(around) if len(around) else 1.0))


def played_clarity(an: dict, segments, times, strike) -> np.ndarray | None:
    if not an.get("accent_peaks"):
        return None
    pt, pv = played_peaks(an, segments)
    return np.array([clarity(pt, pv, t, s) for t, s in zip(times, strike)])


def played_strike_pre(an: dict, segments, times) -> tuple[np.ndarray, np.ndarray] | None:
    """(strike, pre) at tour times, read off the peaks as the tour plays - a cut just after a
    music edit hears what precedes it in the TOUR, not in the track. None without peaks."""
    if not an.get("accent_peaks"):
        return None
    pt, pv = played_peaks(an, segments)
    return (np.array([peaks_window_max(pt, pv, t, STRIKE_WIN) for t in times]),
            np.array([peaks_window_max(pt, pv, t, PRE_WIN) for t in times]))

from .config import PlannerConfig

MATCH_S = 0.06                  # a hero / phrase time matches a candidate this close
STRUCTURAL = ("drop", "dropout", "entry", "section", "breakdown")   # the music itself changes


def _rank(tier: int, kind: str | None, strength: float) -> tuple:
    """Order heroes are taken in: tier, then what the moment is, then strength."""
    return tier, -KIND_RANK.get(kind or "", 0), -strength


@dataclass
class V2Plan:
    segments: list[tuple[float, float]]   # the track pieces the tour plays (track time)
    edits: list[dict]                     # music edits: {tour_s, from_s, to_s, cost, why}
    path: list[int]             # candidate indices
    times: np.ndarray           # candidate tour times
    bars: np.ndarray
    tier: np.ndarray            # per candidate: 0 / 1 / 2
    strength: np.ndarray
    kinds_hero: list[str | None]
    loop: float
    required_ok: bool
    score: float
    usable_end: float

    @property
    def start_s(self) -> float:
        return self.segments[0][0]


# ---- the music as track pieces ------------------------------------------------------------
def base_segments(an: dict, start: float, needed_s: float) -> tuple[list[tuple[float, float]], float]:
    """The track from `start`, queued again after itself (each copy cut at the loop point, as v3)
    until it covers needed_s; the last copy plays to the file's end (fade tail)."""
    from .music_sync import loop_point, music_stop
    stop, loop, dur = music_stop(an), loop_point(an), float(an["duration"])
    segs, have = [], 0.0
    a = start
    while True:
        if have + (stop - a) >= needed_s or loop <= a:
            segs.append((a, dur))
            break
        segs.append((a, loop))
        have += loop - a
        a = 0.0
    return segs, loop


def seg_times(segments: list[tuple[float, float]], track_t: float) -> list[float]:
    """Every tour time at which track time track_t plays."""
    out, base = [], 0.0
    for a, b in segments:
        if a - 1e-6 <= track_t < b - 1e-9:
            out.append(base + track_t - a)
        base += b - a
    return out


def seg_candidates(an: dict, segments: list[tuple[float, float]]):
    """The track's cut candidates as the tour plays them: (tour times, bar flags, kinds, local
    track times, strike, pre, usable_end) - no cut past the music's stop in the last piece."""
    from .music_sync import music_stop
    stop = music_stop(an)
    cands = sorted(an.get("candidates") or [], key=lambda c: c["time"])
    t_all, b_all, k_all, l_all, s_all, p_all = [], [], [], [], [], []
    base, usable_end = 0.0, 0.0
    for n, (a, b) in enumerate(segments):
        hi = min(b, stop) if n == len(segments) - 1 else b
        for c in cands:
            t = float(c["time"])
            if a - 1e-6 <= t < hi - 1e-9 or (n == len(segments) - 1 and abs(t - hi) < 1e-9):
                t_all.append(base + t - a)
                b_all.append(bool(c["bar_start"]))
                k_all.append(c["kind"])
                l_all.append(t)
                strike = c.get("strike")
                s_all.append(float(strike) if strike is not None
                             else max(float(c.get("hit") or 1.0), float(c.get("ting") or 0.0)))
                p_all.append(float(c.get("pre") or 0.0))
        usable_end = base + (hi - a)
        base += b - a
    arr = np.array
    return (arr(t_all, dtype=float), arr(b_all, dtype=bool), k_all, arr(l_all, dtype=float),
            arr(s_all, dtype=float), arr(p_all, dtype=float), usable_end)


def _bars(an: dict) -> np.ndarray:
    return np.array(sorted(float(b["time"]) for b in an.get("beats") or [] if b.get("bar_start")))


def splice_cost(an: dict, from_t: float, to_t: float, _cache: dict | None = None) -> float:
    """How audible a jump is from the bar starting at from_t to the bar starting at to_t: 1 - the
    mean similarity of the bars just before the join and of the bars just after it (bar_prints).
    0 = identical; two neighbouring bars of one section run ~0.003-0.011, a jump between
    sections ~0.09 (solarflex). inf when the prints are missing."""
    prints = {round(float(p["t"]), 3): np.asarray(p["v"], dtype=float) for p in an.get("bar_prints") or []}
    keys = np.array(sorted(prints))
    if not len(keys):
        return float("inf")

    def at(t, offset):
        i = int(np.argmin(np.abs(keys - t)))
        if abs(keys[i] - t) > 0.05 or not 0 <= i + offset < len(keys):
            return None
        return prints[keys[i + offset]]

    a0, a1, b0, b1 = at(from_t, -1), at(from_t, 0), at(to_t, -1), at(to_t, 0)
    if any(v is None for v in (a0, a1, b0, b1)):
        return float("inf")
    return float(1.0 - 0.5 * (a0 @ b0 + a1 @ b1))


def _usable_len(an: dict, segments) -> float:
    """Tour seconds of music a cut may land in (the last piece up to the music's stop)."""
    from .music_sync import music_stop
    *head, (a, b) = segments
    return sum(y - x for x, y in head) + max(0.0, min(b, music_stop(an)) - a)


def _topped_up(an: dict, segments, needed: float):
    """segments, queued again after themselves (base_segments) when they no longer cover needed."""
    if _usable_len(an, segments) >= needed:
        return segments
    *head, (a, _) = segments
    return head + base_segments(an, a, needed - sum(y - x for x, y in head))[0]


def _jump(segments, from_t: float, to_t: float):
    """The segments with the track from from_t to to_t skipped (inside one piece), else None."""
    out = []
    done = False
    for a, b in segments:
        if not done and a + 1e-6 < from_t and to_t < b - 1e-6 and from_t < to_t:
            out += [(a, from_t), (to_t, b)]
            done = True
        else:
            out.append((a, b))
    return out if done else None


def _mark(local_t: np.ndarray, times: list[float]) -> np.ndarray:
    """Per candidate: does its time in its copy of the track match one of `times`."""
    if not len(times):
        return np.zeros(len(local_t), dtype=bool)
    ts = np.sort(np.asarray(times, dtype=float))
    k = np.clip(np.searchsorted(ts, local_t), 1, len(ts) - 1) if len(ts) > 1 else np.zeros(len(local_t), int)
    near = np.minimum(np.abs(ts[k] - local_t), np.abs(ts[np.maximum(k - 1, 0)] - local_t))
    return near <= MATCH_S


def _heroes_on(local_t: np.ndarray, heroes: list[dict]):
    """Per candidate (tier, strength, kind) of the hero on it (the higher tier, then stronger)."""
    tier = np.zeros(len(local_t), dtype=int)
    strength = np.zeros(len(local_t))
    kind: list[str | None] = [None] * len(local_t)
    for h in heroes:
        d = np.abs(local_t - float(h["t"]))
        near = np.where(d <= MATCH_S)[0]
        # one candidate per hero and queued copy: the nearest (a whoosh's beat and its own
        # whoosh candidate sit within ms of each other)
        for j in near:
            if any(abs(local_t[i] - local_t[j]) < 1e-9 for i in near if i != j) and d[j] > d[near].min():
                continue
            if j > 0 and d[j - 1] <= MATCH_S and d[j - 1] < d[j]:
                continue
            if j + 1 < len(d) and d[j + 1] <= MATCH_S and d[j + 1] <= d[j] and abs(local_t[j + 1] - local_t[j]) < 2 * MATCH_S:
                continue
            if tier[j] == 0 or _rank(h["tier"], h["kind"], h["strength"]) < _rank(tier[j], kind[j], strength[j]):
                tier[j], strength[j], kind[j] = int(h["tier"]), float(h["strength"]), h["kind"]
    return tier, strength, kind


def start_options(an: dict, cfg: PlannerConfig) -> list[float]:
    """0:00, every phrase start and every change of the music in the first music_v2_start_share."""
    from .music_sync import music_stop
    limit = cfg.music_v2_start_share * music_stop(an)
    starts = {0.0} | {round(float(p), 4) for p in an.get("phrases") or [] if 0 < float(p) <= limit}
    starts |= {round(float(h["t"]), 4) for h in an.get("heroes") or []
               if h["kind"] in STRUCTURAL and 0 < float(h["t"]) <= limit}
    return sorted(starts)


def opening_score(an: dict, start: float, cfg: PlannerConfig) -> float:
    """On a change: rewarded. The track's own beginning: fine. A few seconds before a change (the
    first shot plays the end of one part and the music turns under it): penalised."""
    changes = [float(h["t"]) for h in an.get("heroes") or [] if h["kind"] in STRUCTURAL]
    if any(abs(t - start) <= MATCH_S for t in changes):
        return cfg.music_v2_open_on_change
    if start <= float(an.get("music_start") or 0.0) + 0.1:
        return cfg.music_v2_open_natural
    if any(0.25 < t - start <= cfg.music_v2_open_before_s for t in changes):
        return -cfg.music_v2_open_before_change
    return 0.0


def _plan_for(segments, edits: list[dict], loop: float, targets, an: dict, cfg: PlannerConfig,
              roles) -> V2Plan | None:
    """The best cuts for the music as these segments play it, and how good the result is."""
    from .music_sync import length_bounds, snap_durations

    planned = float(sum(targets))
    times, bars, kinds, local_t, strike, pre, usable_end = seg_candidates(an, segments)
    if len(times) < len(targets):
        return None
    sp = played_strike_pre(an, segments, times)
    clear = None
    if sp is not None:
        strike, pre = sp
        clear = played_clarity(an, segments, times, strike)
    tier, strength, kind = _heroes_on(local_t, an.get("heroes") or [])
    phrase = _mark(local_t, an.get("phrases") or [])
    offbeat = np.array([k == "offbeat" for k in kinds])
    t1b, t1s = cfg.music_v2_tier1
    t2b, t2s = cfg.music_v2_tier2
    # cut quality on the shared accent scale (tour_worker/music/analysis.py): + how hard the sound
    # hits at the cut (/ the track's typical candidate), - a harder sound up to a beat before it
    # (pre / strike above 1: the cut reads late; the audit of a render measures the same)
    typical = float(np.median(strike)) or 1.0
    ratio = pre / np.maximum(strike, 1e-6)
    quality = (cfg.music_v2_strike * np.clip(strike / typical - 1.0, 0.0, 1.0)
               - cfg.music_v2_pre * np.clip((ratio - 1.0) / (LATE_RATIO - 1.0), 0.0, 3.0))
    if clear is not None:            # a cut on a moment that does not stand out cannot be seen in time
        quality -= cfg.music_v2_unclear * np.clip((CLEAR - clear) / (CLEAR - UNCLEAR), 0.0, 1.0)
    values = (cfg.music_v2_phrase * phrase + cfg.music_v2_bar * bars + quality
              - cfg.music_offbeat_penalty * offbeat
              + np.where(tier == 1, t1b + t1s * strength, 0.0) + np.where(tier == 2, t2b + t2s * strength, 0.0))

    # heroes a cut may take, strongest first; one too close to a taken hero is left for a punch
    reach = planned + 2 * cfg.music_max_drift
    order = sorted((j for j in np.where(tier > 0)[0]
                    if cfg.music_shot_abs_min_s - 1e-6 <= times[j] <= min(reach, usable_end)),
                   key=lambda j: (*_rank(tier[j], kind[j], strength[j]), times[j]))
    taken: list[int] = []
    for j in order:
        if all(abs(times[j] - times[i]) >= cfg.music_v2_hero_gap_s for i in taken):
            taken.append(int(j))
    for j in taken:
        if tier[j] == 2:
            values[j] += cfg.music_v2_taken_bonus
    required = [np.array([j]) for j in taken if tier[j] == 1]

    usable = times <= usable_end + 1e-6
    idx = np.where(usable)[0]
    remap = {int(j): i for i, j in enumerate(idx)}
    t_u, v_u = times[usable], values[usable]
    req_u = [np.array([remap[int(c[0])]]) for c in required if int(c[0]) in remap]
    end_masks = (phrase[usable], bars[usable], np.ones(len(t_u), dtype=bool))
    role_b = length_bounds(targets, roles, cfg)
    wide_b = length_bounds(targets, roles, cfg, wide=True)
    d = cfg.music_max_drift
    tg = np.asarray(targets, dtype=float)

    def objective(p):                          # snap_durations' own objective, for comparing
        ends = t_u[p]
        lengths = np.diff(np.concatenate([[0.0], ends]))
        return float(v_u[p].sum() - cfg.music_length_weight * (((lengths - tg) / tg) ** 2).sum())

    # each attempt is solved within the role bounds AND the wide bounds, the better kept: a hero
    # reachable only by stretching a shot a little more is worth it (a plain beat is not)
    path, required_ok = None, False
    for req, dr in ((True, d), (True, 2 * d), (True, 4 * d), (False, d), (False, 2 * d), (False, 4 * d)):
        if req and not req_u:
            continue
        for end_mask in end_masks:
            sols = [p for lo, hi in (role_b, wide_b)
                    if (p := snap_durations(t_u, v_u, end_mask, targets, lo, hi, cfg.final_shot_min_s,
                                            cfg.music_length_weight, dr, True,
                                            required=req_u if req else None)) is not None]
            if sols:
                best = max(sols, key=objective)
                path, required_ok = [int(idx[j]) for j in best], req
                break
        if path is not None:
            break
    if path is None:
        return None

    # how good is this music: tier-1 heroes cut, tier-2 cut, the first drop on an early shot, the
    # opening, clean cuts, and as few / as seamless music edits as possible
    cuts = path[:-1]
    in_tour = [j for j in np.where(tier == 1)[0] if 0.5 < times[j] < times[path[-1]] - 0.5]
    score = sum(10.0 for j in cuts if tier[j] == 1) + sum(3.0 * strength[j] for j in cuts if tier[j] == 2)
    score -= 10.0 * sum(1 for j in in_tour if j not in cuts)
    first_drop = next((k for k, j in enumerate(cuts) if tier[j] == 1), None)
    if first_drop is not None and first_drop <= 2:
        score += cfg.music_v2_early_drop_bonus
    start = segments[0][0]
    score += opening_score(an, start, cfg) - cfg.music_v2_start_cost * start
    score += float(quality[cuts].sum())
    score -= sum(cfg.music_v2_edit_cost + 100.0 * e["cost"] for e in edits)
    return V2Plan(list(segments), list(edits), path, times, bars, tier, strength, kind, loop,
                  required_ok or not required, score, usable_end)


def _edit_options(an: dict, segments, targets, cfg: PlannerConfig) -> list[tuple[str, float, float, float]]:
    """Seamless bar jumps worth trying for this music: (why, from_t, to_t, cost).
      land   1-4 bars out before a drop so it falls on a planned cut
      reach  whole phrases out so a drop past the tour's end comes inside it"""
    bars = _bars(an)
    if len(bars) < 3 or not an.get("bar_prints"):
        return []
    bar_s = float(np.median(np.diff(bars)))
    planned = float(sum(targets))
    grid = np.cumsum(targets)[:-1]
    drops = sorted(float(h["t"]) for h in an.get("heroes") or [] if h["tier"] == 1)
    start = segments[0][0]
    out = []

    def jumps(lo_t: float, hi_t: float, k: int, why: str, best_n: int = 2):
        """The best_n most seamless jumps of k bars between lo_t and hi_t (track time)."""
        found = []
        for i in range(len(bars) - k):
            a, b = float(bars[i]), float(bars[i + k])
            if a < lo_t - 1e-6 or b > hi_t + 1e-6 or a <= start + 0.5:
                continue
            if any(a - 1e-6 <= d < b - 1e-6 for d in drops):            # never through a drop
                continue
            c = splice_cost(an, a, b)
            if c <= cfg.music_v2_splice_max:
                found.append((c, a, b))
        return [(why, a, b, c) for c, a, b in sorted(found)[:best_n]]

    prev = start
    for d in drops:
        tt = seg_times(segments, d)
        if not tt or tt[0] <= 0:
            continue
        th = tt[0]
        if th <= planned + cfg.music_max_drift:
            # land: the planned cut just before the drop, k bars earlier than it
            before = [g for g in grid if g < th - 0.2]
            if before:
                k = int(round((th - before[-1]) / bar_s))
                for kk in sorted({k, k - 1, k + 1}):
                    if 1 <= kk <= 4:
                        out += jumps(prev, d, kk, "land")
        elif seg_times(segments, d)[0] > planned - cfg.music_shot_abs_min_s:
            # reach: whole phrases (4 bars) out between the previous drop and this one
            lo_k = int(np.ceil((th - (planned - 3.0)) / (4 * bar_s)))
            hi_k = int(np.floor((th - 0.5 * planned) / (4 * bar_s)))
            for kk in range(max(1, lo_k), min(hi_k, lo_k + 2) + 1):
                out += jumps(prev + 4 * bar_s, d - 4 * bar_s, 4 * kk, "reach")
            break                                   # the first drop past the end is enough
        prev = d
    return out


def plan_v2(targets, an: dict, cfg: PlannerConfig, roles) -> V2Plan | None:
    """Every start, as the track plays and with up to music_v2_max_edits seamless music edits;
    the best scoring plan (earliest start / fewest edits on ties: they are tried first)."""
    planned = float(sum(targets))
    best = None

    def consider(p):
        nonlocal best
        if p is not None and (best is None or p.score > best.score + 1e-9):
            best = p

    for s in start_options(an, cfg):
        needed = planned + 4 * cfg.music_max_drift + cfg.music_fade_out_s
        segs, loop = base_segments(an, s, needed)
        consider(_plan_for(segs, [], loop, targets, an, cfg, roles))
        if cfg.music_v2_max_edits <= 0:
            continue
        singles = []
        for why, a, b, c in _edit_options(an, segs, targets, cfg):
            # the trimmed music must still cover the tour: top it up as the base music does
            j = _jump(segs, a, b)
            if j is None:
                continue
            j = _topped_up(an, j, needed)
            e = {"tour_s": round(seg_times(segs, a)[0], 3), "from_s": round(a, 4), "to_s": round(b, 4),
                 "cost": round(c, 4), "why": why}
            p = _plan_for(j, [e], loop, targets, an, cfg, roles)
            consider(p)
            singles.append((j, e))
        if cfg.music_v2_max_edits >= 2:
            # a "land" edit near the start and a "reach" edit later, applied together
            for j1, e1 in [x for x in singles if x[1]["why"] == "land"]:
                for why, a, b, c in _edit_options(an, j1, targets, cfg):
                    if why != "reach":
                        continue
                    j2 = _jump(j1, a, b)
                    if j2 is None:
                        continue
                    j2 = _topped_up(an, j2, needed)
                    e2 = {"tour_s": round(seg_times(j1, a)[0], 3), "from_s": round(a, 4), "to_s": round(b, 4),
                          "cost": round(c, 4), "why": why}
                    consider(_plan_for(j2, [e1, e2], loop, targets, an, cfg, roles))
    return best


def punches_for(plan: V2Plan, durations: list[float], cfg: PlannerConfig, beat_s: float, lead: int):
    """Heroes inside the tour: (report rows, hero cut indices, accents [(shot k, at_s, amount)])."""
    from .schemas import MusicHook

    fps = cfg.fps
    cuts = np.cumsum(durations)[:-1]
    tour_end = float(np.sum(durations))
    starts = np.concatenate([[0.0], cuts])
    tol = (lead + 1) / fps
    rows, hook_cuts, accents = [], [], []
    heroes = [j for j in np.where(plan.tier > 0)[0] if 0.5 < plan.times[j] < tour_end - 0.5]
    t1a = cfg.music_v2_punch_tier1
    t2b, t2s = cfg.music_v2_punch_tier2
    placed: list[float] = []
    for j in sorted(heroes, key=lambda j: (*_rank(plan.tier[j], plan.kinds_hero[j], plan.strength[j]), plan.times[j])):
        t = float(plan.times[j])
        tier, kind = int(plan.tier[j]), plan.kinds_hero[j]
        near = int(np.argmin(np.abs(cuts - t))) if len(cuts) else -1
        if near >= 0 and abs(cuts[near] - t) <= tol:
            hook_cuts.append(near)
            rows.append(MusicHook(t=round(t, 3), kind="cut", tier=tier, hero=kind))
            continue
        if not cfg.music_punches:                              # no zoom punches: the moment plays uncut
            rows.append(MusicHook(t=round(t, 3), kind="uncut", tier=tier, hero=kind))
            continue
        k = int(np.searchsorted(starts, t, side="right") - 1)
        at = round(t - float(starts[k]) - lead / fps, 3)        # punches lead their beat, as cuts
        spaced = (all(abs(t - p) >= beat_s - 1e-6 for p in placed)
                  and (float(np.abs(cuts - t).min()) if len(cuts) else 1e9) >= cfg.music_accent_edge_s)
        if spaced and at >= cfg.music_accent_edge_s and durations[k] - at >= cfg.music_accent_edge_s:
            amount = t1a if tier == 1 else t2b + t2s * float(plan.strength[j])
            accents.append((k, at, round(min(amount, 0.06), 4)))
            placed.append(t)
            rows.append(MusicHook(t=round(t, 3), kind="accent", tier=tier, hero=kind))
        elif len(cuts) and float(np.abs(cuts - t).min()) < cfg.music_accent_edge_s:
            # a cut this close already marks the moment; a punch would collide with it
            rows.append(MusicHook(t=round(t, 3), kind="covered", tier=tier, hero=kind, reason="next_to_cut"))
        else:
            rows.append(MusicHook(t=round(t, 3), kind="missed", tier=tier, hero=kind,
                                  reason="too_close_to_start" if t < cfg.music_shot_abs_min_s else "no_room"))
    rows.sort(key=lambda r: r.t)
    accents.sort()
    return rows, sorted(set(hook_cuts)), accents
