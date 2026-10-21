"""Pick the photos the student pass featurises and the adapter trains on, and the held-out split.

  python training/fireplace_adapter/select_photos.py [--max-total 450]

- every teacher photo with a fireplace, lit or not (lit only stratifies the split; it is recomputed by
  relabel_lit.py with the current fire.py when its _lit.json exists);
- negatives (no fireplace): as many as positives, at least 150 and enough for --min-total photos;
  half of them "hard" - the ADE scene
  has lamps, chandeliers, sconces, lights, a stove / oven or a kitchen island (warm lights and
  stone are what fooled the old colour search) - the rest random. Seeded.
- at most SELECT_CAP photos per folder (the shared web-image / upload folders are big) and DUP_CAP per
  near-duplicate group - for negatives only; every fireplace photo is kept.
- near-duplicates (pHash): a group never straddles the split; near-duplicates of the known evaluation listings are dropped.
- split: by listing folder (never by photo), ~20% of the folders held out, stratified so the held-out
  part has ~20% of the fireplace photos and ~20% of the lit ones.
Writes out/fireplace_adapter/selected.json {ids, split: {id: train|holdout}, counts}.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

HARD = ("lamp", "chandelier", "sconce", "light", "stove", "oven", "kitchen island", "candle")
HOLDOUT = 0.2
DUP_HAMMING = 10                             # pHash bits: closer = the same photo re-encoded / re-cropped
DUP_CAP = 2                                  # near-duplicates kept per group


def phash(bgr) -> int:
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def dup_groups(hashes: dict[str, int]) -> dict[str, str]:
    """Union-find over photos whose pHashes are within DUP_HAMMING bits -> id: group root."""
    parent = {k: k for k in hashes}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    ks = sorted(hashes)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if bin(hashes[a] ^ hashes[b]).count("1") <= DUP_HAMMING:
                parent[root(a)] = root(b)
    return {k: root(k) for k in ks}
SELECT_CAP = 12                              # photos per folder at most


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-total", type=int, default=450)
    ap.add_argument("--min-total", type=int, default=350)   # fill up with negatives when positives are few
    a = ap.parse_args()
    rows = {r["hash"]: r for r in C.manifest()}
    lab = {}
    for f in os.listdir(C.TEACHER):
        if f.endswith(".json"):
            j = json.load(open(os.path.join(C.TEACHER, f)))
            if not j.get("skipped") and f[:-5] in rows:
                lp = os.path.join(C.TEACHER, f[:-5] + "_lit.json")      # relabel_lit.py (current fire.py)
                if os.path.exists(lp):
                    j["lit"] = json.load(open(lp))["lit"]
                lab[f[:-5]] = j
    # near-duplicates (the same room uploaded again under another ETag): pHash groups. Photos that
    # look like a known evaluation-listing photo are dropped; a group never straddles the split.
    hp = os.path.join(C.OUT, "phash.json")
    ph = json.load(open(hp)) if os.path.exists(hp) else {}
    for h in lab:
        if h not in ph:
            ph[h] = phash(C.read_bgr(C.photo_path(rows[h])))
    for name, d in C.KNOWN_LISTINGS.items():
        for f in sorted(os.listdir(d)):
            if f.endswith(".jpg") and f"known:{name}/{f}" not in ph:
                ph[f"known:{name}/{f}"] = phash(cv2.imread(os.path.join(d, f)))
    json.dump(ph, open(hp, "w"))
    known_ph = [v for k, v in ph.items() if k.startswith("known:")]
    near_known = {h for h in lab if any(bin(ph[h] ^ k).count("1") <= DUP_HAMMING for k in known_ph)}
    for h in near_known:
        lab.pop(h)
    group = dup_groups({h: ph[h] for h in lab})
    rng = random.Random(C.SEED)
    used = defaultdict(int)                      # photos taken per folder (cap SELECT_CAP)
    used_g = defaultdict(int)                    # photos taken per near-duplicate group (cap DUP_CAP)

    def ok(h):
        return used[rows[h]["folder"]] < SELECT_CAP and used_g[group[h]] < DUP_CAP

    def take(h):
        used[rows[h]["folder"]] += 1
        used_g[group[h]] += 1

    # every fireplace photo is kept (user 2026-10-21: the caps must not cost fireplace photos); they
    # still count towards the caps the negatives see
    pos = sorted(h for h, j in lab.items() if j["has_fireplace"])
    for h in pos:
        take(h)
    neg = sorted(h for h, j in lab.items() if not j["has_fireplace"])
    n_neg = min(len(neg), max(150, len(pos), a.min_total - len(pos)), max(0, a.max_total - len(pos)))
    hard = [h for h in neg if any(any(k in c for k in HARD) for c in lab[h]["scene"])]
    easy = [h for h in neg if h not in set(hard)]
    rng.shuffle(hard)
    rng.shuffle(easy)
    take_hard = hard[:n_neg // 2]
    rest = [h for h in easy + hard[n_neg // 2:]]
    rng.shuffle(rest)
    negs = []
    for h in take_hard + rest:                   # hard ones first, then the rest, within the folder cap
        if len(negs) < n_neg and ok(h):
            take(h)
            negs.append(h)
    take_hard = [h for h in take_hard if h in set(negs)]
    ids = sorted(pos + negs)

    # split by unit = listing folder, merged with every folder a near-duplicate of its photos lives in
    unit = {f: f for f in {rows[h]["folder"] for h in ids}}

    def uroot(f):
        while unit[f] != f:
            f = unit[f]
        return f
    first = {}
    for h in ids:
        g = group[h]
        if g in first:
            unit[uroot(rows[h]["folder"])] = uroot(first[g])
        else:
            first[g] = rows[h]["folder"]
    by = defaultdict(list)
    for h in ids:
        by[uroot(rows[h]["folder"])].append(h)

    def stratum(folder):
        js = [lab[h] for h in by[folder]]
        return 2 if any(j["lit"] for j in js) else 1 if any(j["has_fireplace"] for j in js) else 0

    split = {}
    for s in (2, 1, 0):
        folders = sorted(f for f in by if stratum(f) == s)
        rng.shuffle(folders)
        n_photos = sum(len(by[f]) for f in folders)
        held = 0
        for f in folders:
            hold = held < HOLDOUT * n_photos
            held += len(by[f]) if hold else 0
            for h in by[f]:
                split[h] = "holdout" if hold else "train"
    counts = {}
    for part in ("train", "holdout"):
        hs = [h for h in ids if split[h] == part]
        counts[part] = {"photos": len(hs), "fireplace": sum(lab[h]["has_fireplace"] for h in hs),
                        "lit": sum(lab[h]["lit"] for h in hs), "folders": len({rows[h]["folder"] for h in hs})}
    counts["scanned"] = len(lab) + len(near_known)
    counts["dropped_near_known_listing"] = len(near_known)
    counts["near_duplicate_groups_in_selection"] = len({group[h] for h in ids})
    counts["split_units"] = len(by)
    counts["teacher_fireplace_scanned"] = sum(j["has_fireplace"] for j in lab.values())
    counts["fireplace_selected"] = len(pos)
    counts["fireplace_near_duplicate_groups"] = len({group[h] for h in pos})
    counts["teacher_lit"] = sum(j["lit"] for j in lab.values())
    counts["hard_negatives"] = len(take_hard)
    json.dump({"ids": ids, "split": split, "counts": counts}, open(os.path.join(C.OUT, "selected.json"), "w"),
              indent=1)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
