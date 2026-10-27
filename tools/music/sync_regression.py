"""Beat-sync regression check: every library track against a fixed set of shot plans, scored on
what the review asked for - so a change that fixes one cut cannot quietly break another
(review 2026-10-23: "we fix some, we lose some").

  python -m tools.music.sync_regression <library folder> [--save base.json] [--compare base.json] [--v3]

Per case (tour x track), from the sync itself (seconds, no render):
  drops       tier-1 heroes inside the tour / with a cut
  coverage    heroes inside the tour with a cut or a punch
  after       cuts with a harder sound up to a beat before them (analysis pre > 1.15 x strike: the
              shared accent scale, = tools/quality/sync_audio_audit.py on a render)
  strike      cuts on a hard sound (strike >= 1.3)
  opening     on_change | plain | before_change (the music turning 0.25-3 s into the first shot)
  edits       music edits, and the worst join (bar-print distance)
--compare fails (exit 1) when any case got worse on any of them; --v3 scores today's v3 sync.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

from tools.quality.sync_quality import played
from tour_worker.planner.music_sync_v2 import LATE_RATIO, played_clarity, played_strike_pre

UNCLEAR_LIMIT = 1.8      # a cut whose hit stands out less than this from the music around it
from tour_worker.planner import music_sync
from tour_worker.planner.config import PlannerConfig

TOURS = os.path.join(os.path.dirname(__file__), "regression_tours.json")
OPEN_RANK = {"on_change": 2, "natural": 2, "plain": 1, "before_change": 0}
STRUCTURAL = ("drop", "dropout", "entry", "section", "breakdown")


def score_case(targets, roles, an: dict, cfg: PlannerConfig, v3: bool = False) -> dict:
    track = {"track_id": "t", "s3_uri": "s3://t", "sha256": "0" * 64, "analysis": an}
    r = (music_sync._sync_v3(targets, track, cfg, roles) if v3
         else music_sync.sync_to_music(targets, track, cfg, roles))
    m = r.music
    fps, lead = cfg.fps, m.cut_lead_frames or 0
    cuts = np.cumsum(r.durations)[:-1]
    end = float(np.sum(r.durations))
    tol = (lead + 1) / fps
    starts = np.concatenate([[0.0], cuts])
    punch_at = []
    if cfg.music_punches:
        punch_at = [float(starts[a[0]] + a[1]) + lead / fps for a in (r.accents or [])]
        punch_at += [float(cuts[k]) + lead / fps for k in (r.hook_cuts or [])]   # the opening punch
    heroes = [(t, h) for t, h in played(an, m, an.get("heroes") or [], key="t") if 0.5 < t < end - 0.5]
    drops = [t for t, h in heroes if h["tier"] == 1 and t >= cfg.music_shot_abs_min_s]
    cut_at = lambda t: bool(len(cuts)) and float(np.abs(cuts + lead / fps - t).min()) <= tol  # noqa: E731
    segs = [(float(s.from_s), float(s.to_s)) for s in m.segments] if getattr(m, "segments", None) else None
    beats_at = cuts + lead / fps
    if segs and an.get("accent_peaks"):
        sv, pv = played_strike_pre(an, segs, beats_at)
    else:
        cands = played(an, m, an.get("candidates") or [])
        rows_at = [min(cands, key=lambda x: abs(x[0] - b))[1] for b in beats_at] if cands else []
        sv = np.array([float(r.get("strike") or 1.0) for r in rows_at])
        pv = np.array([float(r.get("pre") or 0.0) for r in rows_at])
    typical = float(np.median([float(c.get("strike") or 1.0) for c in an.get("candidates") or []] or [1.0]))
    strike = int(np.sum(sv >= 1.3 * typical))
    after = [round(float(c), 2) for c, s, p in zip(cuts, sv, pv) if p > LATE_RATIO * s]
    unclear = []
    if segs and an.get("accent_peaks"):
        cl = played_clarity(an, segs, beats_at, sv)
        unclear = [round(float(c), 2) for c, x in zip(cuts, cl) if x < UNCLEAR_LIMIT]
    changes = [t for t, h in played(an, m, an.get("heroes") or [], key="t") if h["kind"] in STRUCTURAL]
    natural = float(m.start_s or 0.0) <= float(an.get("music_start") or 0.0) + 0.1
    opening = ("on_change" if any(abs(t) <= 0.06 for t in changes)
               else "natural" if natural      # the song's own beginning (a short pickup is fine)
               else "before_change" if any(0.25 < t <= 3.0 for t in changes) else "plain")
    covered = sum(1 for t, _ in heroes if cut_at(t) or any(abs(p - t) <= 1.5 / fps for p in punch_at))
    edits = (getattr(m, "edits", None) or [])
    return {"tour_s": round(end, 2), "music_start_s": round(float(m.start_s or 0.0), 2),
            "drops_cut": sum(cut_at(t) for t in drops), "drops": len(drops),
            "coverage": round(covered / len(heroes), 3) if heroes else None, "heroes": len(heroes),
            "after": after, "unclear": unclear, "strike": strike, "cuts": len(cuts), "opening": opening,
            "edits": len(edits), "worst_join": round(max((e["cost"] for e in edits), default=0.0), 4),
            "warnings": [w for w in [r.warning, *(r.extra_warnings or [])] if w]}


def worse(new: dict, old: dict, cfg: PlannerConfig) -> list[str]:
    out = []
    if new["drops"] and old["drops"] and new["drops_cut"] / new["drops"] < old["drops_cut"] / old["drops"] - 1e-9:
        out.append(f"drops cut {old['drops_cut']}/{old['drops']} -> {new['drops_cut']}/{new['drops']}")
    if new["drops_cut"] < old["drops_cut"]:
        out.append(f"fewer drops cut {old['drops_cut']} -> {new['drops_cut']}")
    if (new["coverage"] or 0) < (old["coverage"] or 0) - 0.02:
        out.append(f"coverage {old['coverage']} -> {new['coverage']}")
    if len(new["after"]) > len(old["after"]):
        out.append(f"cuts after a harder sound {len(old['after'])} -> {len(new['after'])} {new['after']}")
    if len(new.get("unclear", [])) > len(old.get("unclear", [])):
        out.append(f"unclear cuts {len(old.get('unclear', []))} -> {len(new['unclear'])} {new['unclear']}")
    if OPEN_RANK[new["opening"]] < OPEN_RANK[old["opening"]]:
        out.append(f"opening {old['opening']} -> {new['opening']}")
    if new["worst_join"] > cfg.music_v2_splice_max + 1e-9:
        out.append(f"join {new['worst_join']} over the seamless limit")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--v3", action="store_true")
    a = ap.parse_args(argv)
    cfg = PlannerConfig()
    tours = {k: v for k, v in json.load(open(TOURS)).items() if not k.startswith("_")}
    tracks = sorted(f for f in os.listdir(a.library) if f.endswith(".analysis.v4.json"))
    res = {}
    for tr in tracks:
        an = json.load(open(os.path.join(a.library, tr)))
        name = tr.split(".analysis")[0]
        for tour, t in tours.items():
            res[f"{tour} x {name}"] = score_case(t["targets"], t["roles"], an, cfg, v3=a.v3)
    old = json.load(open(a.compare)) if a.compare else {}
    bad = 0
    print(f"{'case':<32} {'drops':>6} {'cover':>6} {'after':>6} {'unclr':>6} {'strike':>7} {'opening':>14} {'edits':>6} {'join':>7} {'start':>7}")
    for k, r in res.items():
        d = f"{r['drops_cut']}/{r['drops']}"
        print(f"{k:<32} {d:>6} {str(r['coverage']):>6} {len(r['after']):>6} {len(r.get('unclear', [])):>6} {r['strike']:>3}/{r['cuts']:<3} "
              f"{r['opening']:>14} {r['edits']:>6} {r['worst_join']:>7} {r['music_start_s']:>7}")
        if k in old:
            w = worse(r, old[k], cfg)
            if w:
                bad += 1
                print("    WORSE: " + "; ".join(w))
    tot = lambda key: sum(r[key] for r in res.values())  # noqa: E731
    print(f"\ntotal: drops cut {tot('drops_cut')}/{tot('drops')}, cuts after a harder sound "
          f"{sum(len(r['after']) for r in res.values())}, unclear cuts "
          f"{sum(len(r.get('unclear', [])) for r in res.values())}, openings before a change "
          f"{sum(r['opening'] == 'before_change' for r in res.values())}, edits {tot('edits')}")
    if a.save:
        json.dump(res, open(a.save, "w"), indent=1)
    if a.compare:
        print("REGRESSION CHECK PASS" if not bad else f"REGRESSION CHECK FAIL: {bad} case(s) worse")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
