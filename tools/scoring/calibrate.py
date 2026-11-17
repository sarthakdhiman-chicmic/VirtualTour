"""Calibrate the photo-scoring weights against human hero picks (SOW PS-07: "thresholds calibrated on
20-30 real listings against human hero picks").

  python -m tools.scoring.calibrate <manifest.json> [--step 0.05] [--out report.json]

manifest.json: [{"listing": "<name>", "photos": [{"photo_id", "path", "room_type"}, ...],
                 "heroes": {"<room_key>": "<photo_id the person picked>", ...}}, ...]
room_key as the scorer forms it (scoring/rooms.room_key_for: the canonical room type, e.g.
"living_room"; "bedroom_2" when two rooms of one type carry labels).

Every photo is scored once (scoring/scorer.score_photo: its four criteria); then each weighting on a
grid (the four weights in steps of --step, summing to 1) picks each room's hero the scorer's way
(highest weighted total of the criteria it has, renormalised), and is scored by agreement: the share
of rooms whose hero is the person's pick. Prints the current weights' agreement, the best grid
weights, and the rooms they still disagree on. Deterministic."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

CRITERIA = ("framing", "lighting", "coverage", "depth_of_field")


def _scores(manifest: list[dict]) -> list[dict]:
    from tour_worker.scoring.config import ScoringConfig
    from tour_worker.scoring.imaging import photo_from_bytes
    from tour_worker.scoring.scorer import score_photo
    cfg = ScoringConfig()
    rows = []
    for item in manifest:
        for p in item["photos"]:
            with open(p["path"], "rb") as f:
                ph = photo_from_bytes(str(p["photo_id"]), f.read(), room_type=p.get("room_type"))
            sc = score_photo(ph, cfg)
            rows.append({"listing": item["listing"], "photo_id": sc.photo_id, "room_key": sc.room_key,
                         "criteria": {k: (sc.criteria[k].score if k in sc.criteria else None) for k in CRITERIA}})
    return rows


def _hero(rows: list[dict], w: dict[str, float]) -> str:
    def total(r):
        have = {k: v for k, v in r["criteria"].items() if v is not None}
        ws = sum(w[k] for k in have) or 1.0
        return sum(w[k] * v for k, v in have.items()) / ws
    return max(sorted(rows, key=lambda r: r["photo_id"]), key=total)["photo_id"]


def agreement(rows: list[dict], manifest: list[dict], w: dict[str, float]) -> tuple[float, list[tuple]]:
    picks = [(m["listing"], rk, pid) for m in manifest for rk, pid in (m.get("heroes") or {}).items()]
    ok, misses = 0, []
    for listing, rk, pid in picks:
        room = [r for r in rows if r["listing"] == listing and r["room_key"] == rk]
        if not room:
            misses.append((listing, rk, pid, None))
            continue
        got = _hero(room, w)
        if got == str(pid):
            ok += 1
        else:
            misses.append((listing, rk, pid, got))
    return (ok / len(picks) if picks else 0.0), misses


def grid(step: float):
    n = int(round(1 / step))
    for a, b, c in itertools.product(range(n + 1), repeat=3):
        d = n - a - b - c
        if d >= 0:
            yield dict(zip(CRITERIA, (a * step, b * step, c * step, d * step)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    manifest = json.load(open(a.manifest))
    rows = _scores(manifest)
    from tour_worker.scoring.config import Weights
    cur = Weights().model_dump()
    cur_ag, _ = agreement(rows, manifest, cur)
    best = max(grid(a.step), key=lambda w: (round(agreement(rows, manifest, w)[0], 6),
                                             -sum(abs(w[k] - cur[k]) for k in CRITERIA)))
    best_ag, misses = agreement(rows, manifest, best)
    rep = {"listings": len(manifest), "rooms_picked": sum(len(m.get("heroes") or {}) for m in manifest),
           "current": {"weights": cur, "agreement": round(cur_ag, 3)},
           "best": {"weights": {k: round(v, 3) for k, v in best.items()}, "agreement": round(best_ag, 3)},
           "still_disagree": [{"listing": l, "room": r, "picked": p, "scorer": g} for l, r, p, g in misses]}
    print(json.dumps(rep, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    if len(manifest) < 20:
        print(f"note: {len(manifest)} listings - the SOW asks for 20-30", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
