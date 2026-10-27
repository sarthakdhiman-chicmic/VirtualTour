"""Sync score for a beat-synced tour: the RENDERED cut frames against the track's analysis.

  python -m tools.quality.sync_quality <workdir with plan.json> <track analysis.json>

The plan's music block says what the planner meant (hooks: cut / accent / missed, the cut lead);
this measures what the timeline actually does, from tour_worker.render.timeline (the frames the renderer
draws), on the track queued as the worker plays it (copies joined at loop_s):
  - per cut: signed offset to the nearest musical moment (ms; the planner cuts lead_frames early,
    so about -33 ms is "on"), and whether that moment is a bar start
  - per hook: the cut on it (offset ms), or the zoom punch in the shot it falls in (accent)
  - median |offset - lead| over all cuts, cuts on a bar
  - drum hits (analysis v3 "hit"): cuts on a strong hit, and cuts LATE after one - a strong hit
    0.2-1.0 s before the cut with none on it (review 2026-10-19: "a beat way before the scene ends")
Sync v2 (analysis v4 heroes; 2026-10-23) - measured against the analysis, not the planner's own
report:
  - tier-1 cut rate: drops inside the tour with a cut on them (must be 100% when reachable)
  - hero coverage: tier 1 + 2 heroes with a cut or a punch on them
  - better moment in reach: a cut on an ordinary beat while an uncut tier-1 hero sat within a
    beat of it (must be 0)
The track may start mid-way (music.start_s): everything is measured on the track as played.
Writes <workdir>/sync.json and exits 1 when a limit below is broken.
"""
import argparse
import json
import os
import sys

import numpy as np

from tour_worker.contract.plan import parse_plan
from tour_worker.render.timeline import compute_timeline

LIMITS = {
    "hook_cut_err_frames": 1.0,      # a hook cut within a frame of (hook - lead)
    "median_cut_err_frames": 1.0,    # cuts in general: on a musical moment, give or take a frame
    "missed_hooks": 0,               # when the planner says every hook moment got a cut
    "accent_err_frames": 1.0,        # an accent's punch peaks within a frame of its hook
    "tier1_cut_rate": 1.0,           # sync v2: every reachable drop has its cut
    "tier1_better_in_reach": 0,      # sync v2: no cut on a plain beat a beat away from an uncut drop
}


STRONG_HIT = 1.3                 # x the track's typical beat
from tour_worker.planner.music_sync_v2 import LATE_RATIO, played_strike_pre  # noqa: E402


def played(analysis: dict, music, items: list, key: str = "time") -> list[tuple[float, dict]]:
    """(tour time, row) for every timed row of the track as the worker plays it: the plan's music
    segments (sync v2 edits) when set, else the track from start_s queued at loop_s."""
    segs = getattr(music, "segments", None)
    if segs:
        pieces = [(float(s.from_s), float(s.to_s)) for s in segs]
    else:
        loop = music.loop_s or float(analysis["duration"])
        start = float(getattr(music, "start_s", 0.0) or 0.0)
        n = max(1, int(music.loops or 1))
        pieces = [(start if k == 0 else 0.0, loop if k < n - 1 else float(analysis["duration"]) + 1) for k in range(n)]
    out, base = [], 0.0
    for n, (a, b) in enumerate(pieces):
        last = n == len(pieces) - 1
        for c in items:
            t = float(c[key])
            if a - 1e-6 <= t < b - 1e-9 or (last and abs(t - b) < 1e-9):
                out.append((base + t - a, c))
        base += b - a
    return out


def queued(analysis: dict, music, with_hits: bool = False):
    """Candidate times / bar flags (/ hit strengths) of the track as the worker plays it."""
    rows = played(analysis, music, analysis.get("candidates") or [])
    t = [r[0] for r in rows]
    bar = [bool(r[1].get("bar_start", False)) for r in rows]
    hit = [float(r[1].get("hit") or 1.0) for r in rows]
    o = np.argsort(t, kind="stable")
    out = (np.asarray(t)[o], np.asarray(bar)[o])
    return out + (np.asarray(hit)[o],) if with_hits else out


def measure(plan, analysis: dict) -> dict:
    fps = plan.output.fps
    m = plan.music
    lead = getattr(m, "cut_lead_frames", 0) or 0
    tl = compute_timeline(plan)
    cuts = np.array([t["nominal_end"] for t in tl[:-1]], dtype=float) / fps
    starts = np.array([t["nominal_start"] for t in tl], dtype=float) / fps
    start_s = float(getattr(m, "start_s", 0.0) or 0.0)
    times, bars, hits = queued(analysis, m, with_hits=True)
    rep = {"cut_lead_frames": lead, "cuts": [], "hooks": [], "missed_hooks": 0,
           "hook_cut_err_frames": 0.0, "accent_err_frames": 0.0}
    errs = []
    for c in cuts:
        j = int(np.argmin(np.abs(times - (c + lead / fps))))
        off = (c - times[j]) * 1000
        errs.append(abs(off / 1000 * fps + lead))
        rep["cuts"].append({"t": round(float(c), 3), "offset_ms": round(float(off), 1), "on_bar": bool(bars[j])})
    rep["median_cut_err_frames"] = float(np.median(errs)) if errs else 0.0
    strong = times[hits >= STRONG_HIT]
    rep["has_hits"] = bool(np.any(hits != 1.0))
    rep["cuts_on_strong_hit"], rep["cuts_late_after_hit"] = 0, 0
    for c, entry in zip(cuts, rep["cuts"]):
        beat = c + lead / fps
        if len(strong) and np.abs(strong - beat).min() <= 1.5 / fps:
            rep["cuts_on_strong_hit"] += 1
            entry["hit"] = "on"
        elif len(strong) and np.any((beat - strong >= 0.2) & (beat - strong <= 1.0)):
            rep["cuts_late_after_hit"] += 1
            entry["hit"] = f"late {float((beat - strong[strong <= beat - 0.2].max()) * 1000):.0f} ms"
    rep["cuts_on_bar"] = sum(c["on_bar"] for c in rep["cuts"])
    raw_hooks = getattr(m, "hooks", None) or []
    for h in raw_hooks:
        t, kind = float(h["t"] if isinstance(h, dict) else h.t), (h["kind"] if isinstance(h, dict) else h.kind)
        entry = {"t": t, "planned": kind}
        if kind == "cut":
            j = int(np.argmin(np.abs(cuts - t)))
            err = abs((cuts[j] - t) * fps + lead)
            entry["offset_ms"] = round(float((cuts[j] - t) * 1000), 1)
            rep["hook_cut_err_frames"] = max(rep["hook_cut_err_frames"], float(err))
        elif kind == "accent":
            k = int(np.searchsorted(starts, t - lead / fps, side="right") - 1)
            ps = plan.shots[k].camera.punches or []
            at = [starts[k] + p.at_s for p in ps]
            if at:
                err = min(abs((a - t) * fps + lead) for a in at)
                entry["punch_err_frames"] = round(float(err), 2)
                rep["accent_err_frames"] = max(rep["accent_err_frames"], float(err))
            else:
                entry["punch_err_frames"] = None
                rep["accent_err_frames"] = float("inf")
        elif kind == "missed":
            # sync v2 tiers its heroes: only a missed drop (tier 1) is a failure there - a tier-2
            # moment with no room for a punch is reported, not failed (hero_metrics)
            tier = h.get("tier") if isinstance(h, dict) else getattr(h, "tier", None)
            if tier in (None, 1):
                rep["missed_hooks"] += 1
        rep["hooks"].append(entry)
    rep["hooks_required"] = bool(getattr(m, "hooks_required", False))
    if analysis.get("heroes") is not None:
        rep["heroes"] = hero_metrics(plan, analysis, cuts, starts, start_s, lead)
    return rep


def hero_metrics(plan, analysis: dict, cuts, starts, start_s: float, lead: int) -> dict:
    """Sync v2 metrics from the analysis' own heroes (v4), on the track as played."""
    fps, m = plan.output.fps, plan.music
    rows = played(analysis, m, analysis["heroes"], key="t")
    beat = float(analysis.get("beat_period_s") or 0.5)
    end = float(plan.total_duration_s)
    tol = (lead + 1) / fps
    punch_at = [float(starts[k] + p.at_s) for k, s in enumerate(plan.shots) for p in (s.camera.punches or [])]
    min_shot = 2.5                               # a cut cannot come sooner (planner music_shot_abs_min_s)
    out = {"tier1": 0, "tier1_cut": 0, "tier1_reachable": 0, "tier2": 0, "covered": 0, "list": []}
    for t, h in rows:
        if not 0.5 < t < end - 0.5:
            continue
        cut = len(cuts) and float(np.abs(cuts - t).min()) <= tol
        punched = any(abs(p - t) <= 1.5 / fps for p in punch_at)
        out["tier1" if h["tier"] == 1 else "tier2"] += 1
        if h["tier"] == 1:
            out["tier1_cut"] += int(bool(cut))
            out["tier1_reachable"] += int(t >= min_shot)
        out["covered"] += int(bool(cut) or punched)
        out["list"].append({"t": round(t, 3), "tier": h["tier"], "kind": h["kind"],
                            "got": "cut" if cut else "punch" if punched else "nothing"})
    uncut_t1 = [e["t"] for e in out["list"] if e["tier"] == 1 and e["got"] != "cut"]
    hero_t = [e["t"] for e in out["list"]]
    out["tier1_better_in_reach"] = sum(
        1 for c in cuts
        if not any(abs(c - t) <= tol for t in hero_t) and any(abs(c - t) <= beat + tol for t in uncut_t1))
    out["tier1_cut_rate"] = round(out["tier1_cut"] / out["tier1_reachable"], 3) if out["tier1_reachable"] else None
    n = out["tier1"] + out["tier2"]
    out["coverage"] = round(out["covered"] / n, 3) if n else None
    # cuts with a harder sound up to a beat before them (the shared accent scale; the same rule
    # as tools/quality/sync_audio_audit.py on the rendered audio)
    segs = ([(float(s.from_s), float(s.to_s)) for s in m.segments] if getattr(m, "segments", None)
            else None)
    after = []
    if segs and analysis.get("accent_peaks"):
        sv, pv = played_strike_pre(analysis, segs, cuts + lead / fps)
        after = [round(float(c), 2) for c, s, p in zip(cuts, sv, pv) if p > LATE_RATIO * s]
    else:
        cands = played(analysis, m, analysis.get("candidates") or [])
        for c in cuts:
            near = [r for t, r in cands if abs(t - (c + lead / fps)) <= 1.5 / fps]
            if near and float(near[0].get("pre") or 0.0) > LATE_RATIO * float(near[0].get("strike") or 1.0):
                after.append(round(float(c), 2))
    out["cuts_after_stronger"] = after
    # the opening and the music edits
    changes = [t for t, h in rows if h["kind"] in ("drop", "dropout", "entry", "section", "breakdown")]
    out["opening"] = ("on_change" if any(abs(t) <= 0.06 for t in changes)
                      else "natural" if start_s <= float(analysis.get("music_start") or 0.0) + 0.1
                      else "before_change" if any(0.25 < t <= 3.0 for t in changes) else "plain")
    edits = getattr(m, "edits", None) or []
    out["edits"] = len(edits)
    out["max_splice_cost"] = max((float(e["cost"]) for e in edits), default=0.0)
    return out


def failures(rep: dict) -> list[str]:
    out = []
    for key in ("hook_cut_err_frames", "median_cut_err_frames", "accent_err_frames"):
        if rep[key] > LIMITS[key]:
            out.append(f"{key} {rep[key]:.2f} > {LIMITS[key]}")
    if rep["hooks_required"] and rep["missed_hooks"] > LIMITS["missed_hooks"]:
        out.append(f"missed_hooks {rep['missed_hooks']}")
    h = rep.get("heroes")
    if h:
        if h["tier1_cut_rate"] is not None and h["tier1_cut_rate"] < LIMITS["tier1_cut_rate"]:
            out.append(f"tier1_cut_rate {h['tier1_cut_rate']} < {LIMITS['tier1_cut_rate']}")
        if h["tier1_better_in_reach"] > LIMITS["tier1_better_in_reach"]:
            out.append(f"tier1_better_in_reach {h['tier1_better_in_reach']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("analysis")
    a = ap.parse_args(argv)
    plan = parse_plan(json.load(open(os.path.join(a.workdir, "plan.json"))))
    if plan.music is None:
        print("no music: nothing to measure")
        return 0
    rep = measure(plan, json.load(open(a.analysis)))
    rep["failures"] = failures(rep)
    rep["limits"] = LIMITS
    json.dump(rep, open(os.path.join(a.workdir, "sync.json"), "w"), indent=1)
    hooks = rep["hooks"]
    n_cut = sum(h["planned"] == "cut" for h in hooks)
    n_acc = sum(h["planned"] == "accent" for h in hooks)
    n_cov = sum(h["planned"] == "covered" for h in hooks)
    print(f"sync: hooks {len(hooks)} -> cut {n_cut}, accent {n_acc}, covered {n_cov}, missed {rep['missed_hooks']}"
          f"{' (all required)' if rep['hooks_required'] else ''}; cuts on bar {rep['cuts_on_bar']}/{len(rep['cuts'])}; "
          f"median cut err {rep['median_cut_err_frames']:.2f} fr, worst hook cut {rep['hook_cut_err_frames']:.2f} fr, "
          f"worst accent {rep['accent_err_frames']:.2f} fr (lead {rep['cut_lead_frames']} fr)")
    h = rep.get("heroes")
    if h:
        print(f"heroes: drops cut {h['tier1_cut']}/{h['tier1_reachable']} reachable "
              f"({h['tier1']} in the tour), hero coverage {h['coverage']} ({h['covered']}/{h['tier1'] + h['tier2']} "
              f"cut or punched), cuts beside an uncut drop {h['tier1_better_in_reach']}")
        print(f"cuts after a harder sound: {len(h['cuts_after_stronger'])} {h['cuts_after_stronger']}; "
              f"opening {h['opening']}; music edits {h['edits']} (worst join {h['max_splice_cost']:.4f})")
    if rep["has_hits"]:
        print(f"drum hits: cuts on a strong hit {rep['cuts_on_strong_hit']}/{len(rep['cuts'])}, "
              f"late after a strong hit {rep['cuts_late_after_hit']} "
              + str([f"{c['t']:.2f}s {c['hit']}" for c in rep["cuts"] if c.get("hit", "").startswith("late")]))
    for h in hooks:
        extra = (f"{h['offset_ms']:+.0f} ms" if "offset_ms" in h
                 else f"punch {h['punch_err_frames']} fr" if "punch_err_frames" in h else "")
        print(f"  hook {h['t']:7.2f}s {h['planned']:<7} {extra}")
    print("SYNC PASS" if not rep["failures"] else "SYNC FAIL:\n  " + "\n  ".join(rep["failures"]))
    return 1 if rep["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
