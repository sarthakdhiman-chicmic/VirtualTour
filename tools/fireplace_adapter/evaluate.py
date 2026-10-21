"""Evaluate the fireplace adapter (only the tiny head runs: COCO features are stored).

  .venv/bin/python tools/fireplace_adapter/relabel_lit.py && .venv/bin/python tools/fireplace_adapter/relabel_lit.py --known
  python tools/safe_run.py --cpus 8,10,12,14 --memory 7G -- .venv/bin/python tools/fireplace_adapter/evaluate.py

Lit fires, end to end, always with the CURRENT modules/perceptual/fire.py:
  teacher      ADE20K fireplace mask (cached) + fire.py           (relabel_lit.py -> <id>_lit.json)
  adapter      adapter fireplace mask (COCO features) + fire.py   (computed here)
  coco_colour  old COCO colour search fire.find_fires on the cached COCO panoptic map (rerun here when
               <id>_pan.png exists, else the result stored by the feature pass)
Held-out split: fireplace IoU vs the teacher, photo-level fireplace P/R, lit P/R of adapter and
coco_colour vs the teacher. Known listings (turkey, downing): the same, plus all three vs a human answer
(HUMAN_LIT), the pendant / sconce traps and Downing's glass-fronted gas fires.
Writes eval/metrics.json, eval/per_photo_*.jsonl and overlay sheets eval/*.jpg.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import adapter as A  # noqa: E402
from relabel_lit import lit_in  # noqa: E402
from modules.perceptual import fire as F  # noqa: E402

MAX_SIDE = 2048                      # modules.segment.MAX_SIDE (not imported: it would pick a model)
MIN_PART_AREA = 0.003                # modules.segment.MIN_PART_AREA
PX_THR = 0.5

# human answer for the known listings (contact sheets 2026-10-21). Turkey indices are those of
# out/perceptual_turkey/images: img_03 / img_04 are the lit living rooms (= turkey_mix img_05 / img_06);
# turkey img_05 / img_06 are the kitchens with the warm pendants.
HUMAN_LIT = {"turkey__img_03", "turkey__img_04", "downing__img_03", "downing__img_06", "downing__img_07",
             "downing__img_11", "downing__img_17"}
HUMAN_FIREPLACE = HUMAN_LIT | {"turkey__img_10", "turkey__img_12", "downing__img_15", "downing__img_18"}
TRAPS = {"turkey__img_05": "kitchen: warm pendants", "turkey__img_06": "kitchen: pendants, stone",
         "downing__img_05": "sconces", "downing__img_10": "kitchen + sconces", "downing__img_08":
         "petrified-wood table", "downing__img_18": "dining: pendants, unlit fireplace",
         "turkey__img_12": "bar: unlit, lit brick + firewood", "turkey__img_10": "outdoor fireplace: unlit, stone"}
GLASS_GAS = {"downing__img_03", "downing__img_06", "downing__img_11", "downing__img_17"}
FIRE_VALIDATE_11 = {  # out/fire_validate.py cases (2026-10-19): id -> lit?
    "turkey__img_03": True, "turkey__img_04": True, "downing__img_03": True, "downing__img_06": True,
    "downing__img_17": True, "turkey__img_12": False, "downing__img_08": False, "turkey__img_05": False,
    "turkey__img_06": False, "turkey__img_10": False, "downing__img_07": True}
METHODS = ("teacher_lit", "adapter_lit", "coco_colour")


def load_model() -> A.FireplaceAdapter:
    cfg = json.load(open(os.path.join(C.MODEL, "config.json")))
    m = A.FireplaceAdapter(width=cfg["width"], drop=cfg["drop"])
    m.load_state_dict(torch.load(os.path.join(C.MODEL, "adapter.pt"), map_location="cpu"))
    return m.eval()


def predict(model, feats: dict, space_hw: tuple[int, int]) -> np.ndarray:
    """Fireplace probability at the photo's MAX_SIDE size."""
    with torch.no_grad():
        p = torch.sigmoid(model(**A.build_inputs(feats)))[0, 0].numpy()
    return cv2.resize(p, (space_hw[1], space_hw[0]), interpolation=cv2.INTER_LINEAR)


def fireplace_mask(prob: np.ndarray) -> np.ndarray:
    """Adapter probability -> production-style fireplace mask: parts >= MIN_PART_AREA, 0/255."""
    m = (prob > PX_THR).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_PART_AREA * m.size
    return (keep[lab] * 255).astype(np.uint8)


_COCO_ID2LABEL: dict | None = None


def coco_colour(bgr: np.ndarray, feat_dir: str, pid: str, sj: dict) -> bool:
    """Old COCO colour search with the current fire.py (cached panoptic map; no model)."""
    global _COCO_ID2LABEL
    pp = os.path.join(feat_dir, pid + "_pan.png")
    if not os.path.exists(pp) or "segments" not in sj:
        return bool(sj["coco_fire"])                   # stored at feature time (older fire.py maybe)
    if _COCO_ID2LABEL is None:
        cfg = json.load(open(os.path.join(C.ROOT, "models", "oneformer_coco_swin_large", "config.json")))
        _COCO_ID2LABEL = {int(k): v for k, v in cfg["id2label"].items()}
    pan = cv2.imread(pp, cv2.IMREAD_UNCHANGED).astype(np.int32)
    return bool(F.find_fires(bgr, pan, sj["segments"], _COCO_ID2LABEL))


def run_set(model, ids, feat_dir, teacher_dir, photo_of) -> list[dict]:
    rows = []
    for pid in ids:
        bgr = C.at_max_side(C.read_bgr(photo_of(pid)), MAX_SIDE)
        feats = dict(np.load(os.path.join(feat_dir, pid + ".npz")))
        t = time.time()
        prob = predict(model, feats, bgr.shape[:2])
        pm = fireplace_mask(prob)
        sec_head = time.time() - t
        t = time.time()
        fires = lit_in(bgr, pm)[0] if pm.any() else []
        sec_rules = time.time() - t
        tj = json.load(open(os.path.join(teacher_dir, pid + ".json")))
        lp = os.path.join(teacher_dir, pid + "_lit.json")
        t_lit = json.load(open(lp))["lit"] if os.path.exists(lp) else (tj["lit"] and tj["has_fireplace"])
        sj = json.load(open(os.path.join(feat_dir, pid + ".json")))
        tm = cv2.imread(os.path.join(teacher_dir, pid + "_fireplace.png"), 0) > 0
        pb = pm > 0
        rows.append({"id": pid, "teacher_fp": tj["has_fireplace"], "teacher_lit": bool(t_lit),
                     "adapter_fp": bool(pb.any()), "adapter_lit": bool(fires),
                     "coco_colour": coco_colour(bgr, feat_dir, pid, sj), "fp_score": float(prob.max()),
                     "inter": int((tm & pb).sum()), "union": int((tm | pb).sum()),
                     "sec_head": round(sec_head, 3), "sec_rules": round(sec_rules, 3),
                     "student_sec": sj["seconds"]})
        k = 480 / bgr.shape[1]                         # small copies only, for the overlays (RAM)
        sz = (480, round(bgr.shape[0] * k))
        rs = lambda a: cv2.resize(a, sz, interpolation=cv2.INTER_AREA)   # noqa: E731
        rows[-1]["_vis"] = (rs(bgr), rs(prob), rs(tm.astype(np.uint8)) > 0,
                            [[round(v * k) for v in r["bbox"]] for r in fires])
    return rows


def pr(rows, pred_key, true_key) -> dict:
    tp = sum(r[pred_key] and r[true_key] for r in rows)
    fp = sum(r[pred_key] and not r[true_key] for r in rows)
    fn = sum(not r[pred_key] and r[true_key] for r in rows)
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": None if p is None else round(p, 3),
            "recall": None if rc is None else round(rc, 3)}


def iou_stats(rows) -> dict:
    pos = [r for r in rows if r["teacher_fp"]]
    per = [r["inter"] / r["union"] for r in pos if r["union"]]
    return {"mean_iou_on_teacher_fireplaces": round(float(np.mean(per)), 3) if per else None,
            "median_iou": round(float(np.median(per)), 3) if per else None,
            "global_iou_all_photos": round(sum(r["inter"] for r in rows) / max(1, sum(r["union"] for r in rows)), 3),
            "n": len(pos)}


def overlay(row, title: str) -> np.ndarray:
    vis, prob, tm, fires = row["_vis"]
    vis = vis.copy()
    g = prob > PX_THR
    vis[g] = (0.5 * vis[g] + [0, 110, 0]).astype(np.uint8)                  # adapter fireplace: green
    cs, _ = cv2.findContours(tm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs, -1, (255, 0, 0), 2)                            # teacher fireplace: blue line
    for x0, y0, x1, y1 in fires:
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)            # adapter + fire.py fire: yellow
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(vis, title[:64], (4, 16), 0, 0.45, (255, 255, 255), 1)
    return vis


def sheet(tiles, path, cols=4) -> None:
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def tag(r) -> str:
    f = lambda k: "Y" if r[k] else "n"   # noqa: E731
    return f"T:{f('teacher_fp')}{f('teacher_lit')} A:{f('adapter_fp')}{f('adapter_lit')} coco:{f('coco_colour')}"


def main() -> None:
    torch.set_num_threads(4)
    os.makedirs(C.EVAL, exist_ok=True)
    model = load_model()
    sel = json.load(open(os.path.join(C.OUT, "selected.json")))
    man = {r["hash"]: r for r in C.manifest()}
    hold = [h for h in sel["ids"] if sel["split"][h] == "holdout"]
    ho = run_set(model, hold, C.FEATURES, C.TEACHER, lambda h: C.photo_path(man[h]))
    kids = sorted(f[:-5] for f in os.listdir(os.path.join(C.KNOWN, "features")) if f.endswith(".json"))
    kn = run_set(model, kids, os.path.join(C.KNOWN, "features"), os.path.join(C.KNOWN, "teacher"),
                 lambda k: os.path.join(C.KNOWN_LISTINGS[k.split("__")[0]], k.split("__")[1] + ".jpg"))
    for r in kn:
        r["human_lit"] = r["id"] in HUMAN_LIT
        r["human_fp"] = r["id"] in HUMAN_FIREPLACE
    by = {r["id"]: r for r in kn}
    m = {"fire_py_sha256": hashlib.sha256(open(F.__file__, "rb").read()).hexdigest(),
         "holdout": {"n": len(ho), "teacher_fireplace": sum(r["teacher_fp"] for r in ho),
                     "teacher_lit": sum(r["teacher_lit"] for r in ho),
                     "fireplace_vs_teacher": pr(ho, "adapter_fp", "teacher_fp"), "iou": iou_stats(ho),
                     **{f"lit_{k}_vs_teacher": pr(ho, k, "teacher_lit") for k in METHODS[1:]}},
         "known": {"n": len(kn), "fireplace_vs_teacher": pr(kn, "adapter_fp", "teacher_fp"),
                   "fireplace_vs_human": pr(kn, "adapter_fp", "human_fp"),
                   "teacher_fireplace_vs_human": pr(kn, "teacher_fp", "human_fp"), "iou": iou_stats(kn),
                   **{f"lit_{k}_vs_teacher": pr(kn, k, "teacher_lit") for k in METHODS[1:]},
                   **{f"lit_{k}_vs_human": pr(kn, k, "human_lit") for k in METHODS}},
         "known_traps": {k: {"what": v, **{mm: by[k][mm] for mm in METHODS}} for k, v in TRAPS.items()},
         "known_glass_gas": {k: {mm: by[k][mm] for mm in METHODS + ("teacher_fp", "adapter_fp")}
                             for k in sorted(GLASS_GAS)},
         "fire_validate_11_correct": {mm: sum(by[k][mm] == v for k, v in FIRE_VALIDATE_11.items()) for mm in METHODS},
         "runtime_s_per_photo": {"adapter_head_median": round(float(np.median([r["sec_head"] for r in ho + kn])), 3),
                                 "fire_rules_median": round(float(np.median([r["sec_rules"] for r in ho + kn])), 3),
                                 "coco_pass_median": round(float(np.median([r["student_sec"] for r in ho + kn])), 2)}}
    cfp = [r for r in ho if r["coco_colour"] and not r["teacher_lit"]]
    m["holdout"]["coco_colour_false_fires"] = {"n": len(cfp), "adapter_also": sum(r["adapter_lit"] for r in cfp)}
    json.dump(m, open(os.path.join(C.EVAL, "metrics.json"), "w"), indent=1)
    for name, rows in (("holdout", ho), ("known", kn)):
        with open(os.path.join(C.EVAL, f"per_photo_{name}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")
    sheet([overlay(r, f"{r['id']} {tag(r)}") for r in kn if r["id"] in HUMAN_FIREPLACE | set(TRAPS)],
          os.path.join(C.EVAL, "known_fireplaces_and_traps.jpg"))
    lit_ho = [r for r in ho if r["teacher_lit"] or r["adapter_lit"]]
    sheet([overlay(r, f"{r['id'][:10]} {tag(r)}") for r in lit_ho[:16]], os.path.join(C.EVAL, "holdout_lit.jpg"))
    dis = [r for r in ho if r["adapter_fp"] != r["teacher_fp"]]
    sheet([overlay(r, f"{r['id'][:10]} {tag(r)}") for r in dis[:16]], os.path.join(C.EVAL, "holdout_disagree.jpg"))
    good = [r for r in ho if r["teacher_fp"] and r["adapter_fp"]]
    sheet([overlay(r, f"{r['id'][:10]} iou {r['inter'] / max(1, r['union']):.2f}") for r in good[:12]],
          os.path.join(C.EVAL, "holdout_fireplaces.jpg"))
    cf = [r for r in ho + kn if r["coco_colour"] and not r["teacher_lit"]]
    sheet([overlay(r, f"{r['id'][:14]} {tag(r)}") for r in cf[:12]], os.path.join(C.EVAL, "old_colour_false_fires.jpg"))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
