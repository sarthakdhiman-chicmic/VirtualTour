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
}


STRONG_HIT = 1.3                 # x the track's typical beat


def queued(analysis: dict, loop_s: float | None, loops: int, with_hits: bool = False):
    """Candidate times / bar flags (/ hit strengths) of the track as the worker plays it."""
    cands = analysis.get("candidates") or []
    loop = loop_s or float(analysis["duration"])
    t, bar, hit = [], [], []
    for k in range(max(1, loops)):
        last = k == max(1, loops) - 1
        for c in cands:
            if c["time"] < loop - 1e-6 or last:
                t.append(float(c["time"]) + k * loop)
                bar.append(bool(c["bar_start"]))
                hit.append(float(c.get("hit") or 1.0))
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
    times, bars, hits = queued(analysis, m.loop_s, m.loops, with_hits=True)
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
            rep["missed_hooks"] += 1
        rep["hooks"].append(entry)
    rep["hooks_required"] = bool(getattr(m, "hooks_required", False))
    return rep


def failures(rep: dict) -> list[str]:
    out = []
    for key in ("hook_cut_err_frames", "median_cut_err_frames", "accent_err_frames"):
        if rep[key] > LIMITS[key]:
            out.append(f"{key} {rep[key]:.2f} > {LIMITS[key]}")
    if rep["hooks_required"] and rep["missed_hooks"] > LIMITS["missed_hooks"]:
        out.append(f"missed_hooks {rep['missed_hooks']}")
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
