"""Which music tracks make good synced tours? Analyse every audio file in a folder (analysis v4,
cached next to it as <name>.analysis.v4.json) and run each through the regression tours
(tools/music/regression_tours.json), then rank them.

  python -m tools.music.rank_tracks <folder> [--json out.json]

Per track: length, tempo, the music's shape (drops, other hero moments, sections), and over the
regression tours: drops cut, hero coverage, cuts after a harder sound (late), openings, music
edits. The verdict says why a track is weak (no drops: cuts can only follow the beat; a slow
tempo: few cut points; many late cuts: a busy, syncopated groove).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from tools.music.sync_regression import TOURS, score_case
from tour_worker.planner.config import PlannerConfig

AUDIO = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def analysis_for(folder: str, name: str) -> dict:
    out = os.path.join(folder, f"{os.path.splitext(name)[0]}.analysis.v4.json")
    if os.path.exists(out):
        return json.load(open(out))
    from tour_worker.music.analysis import make_analysis_v2   # librosa: only when analysing
    an = make_analysis_v2(os.path.join(folder, name))
    an["audio"] = name
    json.dump(an, open(out, "w"), indent=1)
    return an


def rank(folder: str) -> list[dict]:
    cfg = PlannerConfig()
    tours = {k: v for k, v in json.load(open(TOURS)).items() if not k.startswith("_")}
    rows = []
    for name in sorted(f for f in os.listdir(folder) if f.lower().endswith(AUDIO)):
        an = analysis_for(folder, name)
        heroes = an.get("heroes") or []
        t1 = sum(h["tier"] == 1 for h in heroes)
        t2 = sum(h["tier"] == 2 for h in heroes)
        cases = [score_case(t["targets"], t["roles"], an, cfg) for t in tours.values()]
        drops = sum(c["drops"] for c in cases)
        cut = sum(c["drops_cut"] for c in cases)
        late = sum(len(c["after"]) for c in cases) / max(1, sum(c["cuts"] for c in cases))
        cover = sum((c["coverage"] or 0) for c in cases) / len(cases)
        good_open = sum(c["opening"] in ("on_change", "natural") for c in cases)
        edits = sum(c["edits"] for c in cases)
        why = []
        if not an.get("has_beat"):
            why.append("no steady beat: cannot sync")
        if t1 == 0:
            why.append("no drops: cuts can only follow the beat")
        if float(an.get("tempo_bpm") or 0) < 80:
            why.append("slow tempo: few cut points")
        if late > 0.12:
            why.append(f"{late:.0%} of cuts land after a harder sound (busy / syncopated groove)")
        if float(an["duration"]) < 60:
            why.append("short: loops in most tours")
        score = (3.0 * min(t1, 3) + 0.3 * min(t2, 10) + 4.0 * (cut / drops if drops else 0)
                 + 3.0 * cover - 20.0 * late + 0.5 * good_open - (5.0 if not an.get("has_beat") else 0.0))
        rows.append({"track": name, "score": round(score, 2), "length_s": round(float(an["duration"]), 1),
                     "bpm": round(float(an.get("tempo_bpm") or 0), 1), "drops": t1, "other_heroes": t2,
                     "sections": len(an.get("sections") or []), "drops_cut": f"{cut}/{drops}",
                     "coverage": round(cover, 3), "late_share": round(late, 3),
                     "good_openings": f"{good_open}/{len(cases)}", "edits": edits,
                     "verdict": "good" if score >= 12 and not why else "ok" if score >= 6 else "weak",
                     "why": why})
    return sorted(rows, key=lambda r: -r["score"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = rank(a.folder)
    print(f"{'track':<58} {'score':>6} {'len':>6} {'bpm':>6} {'drops':>5} {'heroes':>6} {'cut':>6} {'cover':>6} {'late':>5} {'open':>5}  verdict")
    for r in rows:
        print(f"{r['track'][:58]:<58} {r['score']:>6} {r['length_s']:>6} {r['bpm']:>6} {r['drops']:>5} "
              f"{r['other_heroes']:>6} {r['drops_cut']:>6} {r['coverage']:>6} {r['late_share']:>5.0%} "
              f"{r['good_openings']:>5}  {r['verdict']}" + (f" - {'; '.join(r['why'])}" if r["why"] else ""))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
