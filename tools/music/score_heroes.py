"""How well the detected hero moments (analysis v4 `heroes`) agree with hero moments tapped by ear
(tools/music/tap_heroes.html -> <track>.heroes.json).

  python -m tools.music.score_heroes <analysis.json> <track.heroes.json> [--tol-beats 1]

Taps land late (reaction time), so the typical lag is measured first (median of the matched
offsets) and removed; a tap matches the nearest unmatched detected hero within tol beats.
Reported per tier: recall (your taps the detector found), precision (detected heroes you also
tapped), and the misses / false alarms with their times.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def match(taps: list[float], found: list[float], tol: float, lag: float = 0.0):
    """Greedy nearest matching: [(tap index, found index, offset s)]."""
    pairs = sorted(((abs(t - lag - f), i, j) for i, t in enumerate(taps) for j, f in enumerate(found)
                    if abs(t - lag - f) <= tol))
    used_t, used_f, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_t and j not in used_f:
            used_t.add(i)
            used_f.add(j)
            out.append((i, j, taps[i] - found[j]))
    return out


def score(analysis: dict, tapped: dict, tol_beats: float = 1.0) -> dict:
    beat = float(analysis.get("beat_period_s") or 0.5)
    tol = tol_beats * beat
    heroes = analysis.get("heroes") or []
    marks = tapped["marks"] if isinstance(tapped, dict) else tapped
    all_taps = [m["t"] for m in marks]
    all_found = [h["t"] for h in heroes]
    first = match(all_taps, all_found, tol + 0.3)          # wide pass, to measure the tap lag
    lag = float(np.median([o for _, _, o in first])) if first else 0.0
    out = {"beat_s": round(beat, 4), "tol_s": round(tol, 4), "tap_lag_s": round(lag, 3), "tiers": {}}
    for name, tap_tiers, found_tiers in (("tier1", {1}, {1}), ("tier1+2", {1, 2}, {1, 2})):
        taps = [m["t"] for m in marks if m.get("tier", 2) in tap_tiers]
        found = [h["t"] for h in heroes if h["tier"] in found_tiers]
        # recall of tier 1 counts a tier-1 tap found at ANY tier (found, just ranked low)
        found_for_recall = all_found if name == "tier1" else found
        m_rec = match(taps, found_for_recall, tol, lag)
        m_pre = match(taps if name != "tier1" else all_taps, found, tol, lag)
        hit_t = {i for i, _, _ in m_rec}
        hit_f = {j for _, j, _ in m_pre}
        out["tiers"][name] = {
            "taps": len(taps), "found": len(found),
            "recall": round(len(hit_t) / len(taps), 3) if taps else None,
            "precision": round(len(hit_f) / len(found), 3) if found else None,
            "missed_taps": [round(t, 2) for i, t in enumerate(taps) if i not in hit_t],
            "false_alarms": [round(f, 2) for j, f in enumerate(found) if j not in hit_f],
            "found_as_lower_tier": ([round(taps[i], 2) for i, j, _ in m_rec if heroes[j]["tier"] != 1]
                                    if name == "tier1" else None),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis")
    ap.add_argument("taps")
    ap.add_argument("--tol-beats", type=float, default=1.0)
    a = ap.parse_args()
    res = score(json.load(open(a.analysis)), json.load(open(a.taps)), a.tol_beats)
    print(f"tap lag {res['tap_lag_s'] * 1000:.0f} ms, tolerance +-{res['tol_s'] * 1000:.0f} ms")
    for name, r in res["tiers"].items():
        print(f"{name}: recall {r['recall']} ({r['taps']} taps), precision {r['precision']} ({r['found']} found)")
        print(f"  missed taps:  {r['missed_taps']}")
        print(f"  false alarms: {r['false_alarms']}")
        if r["found_as_lower_tier"]:
            print(f"  tapped as tier 1, found only as tier 2: {r['found_as_lower_tier']}")


if __name__ == "__main__":
    main()
