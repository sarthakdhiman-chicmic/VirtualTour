"""Train the fireplace adapter on stored COCO features against the ADE20K teacher.

  python tools/safe_run.py --cpus 8,10,12,14 --memory 7G -- \
      .venv/bin/python training/fireplace_adapter/train.py [--epochs 30]

Train split from selected.json (held-out folders never seen). Seeded, deterministic, CPU. Saves
out/fireplace_adapter/model/{adapter.pt, config.json, adapter.pt.sha256, train_log.jsonl}.
Target: the teacher's fireplace mask only (decision 2026-10-21); lit fires stay fire.py's job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

import adapter as A  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_item(pid: str, feat_dir: str, teacher_dir: str) -> dict:
    """Features + target for one photo."""
    feats = dict(np.load(os.path.join(feat_dir, pid + ".npz")))
    hw8, _ = A.grid_size(feats["valid"])
    item = A.build_inputs(feats)
    item["y"] = A.targets(cv2.imread(os.path.join(teacher_dir, pid + "_fireplace.png"), 0), hw8)
    return item


def dice(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit)
    return 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)


def loss_fn(out: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
    bce = F.binary_cross_entropy_with_logits(out, y, pos_weight=torch.tensor(3.0))
    d = dice(out, y) if float(y.sum()) > 0 else torch.sigmoid(out).mean()   # negatives: push the map down
    return bce + d, {"bce": float(bce.detach()), "dice": float(d.detach())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    seed_all(C.SEED)
    os.makedirs(C.MODEL, exist_ok=True)
    lg = os.path.join(C.LOGS, "train.log")
    sel = json.load(open(os.path.join(C.OUT, "selected.json")))
    train_ids = [h for h in sel["ids"] if sel["split"][h] == "train"]
    t0 = time.time()
    C.log(lg, f"train photos {len(train_ids)}")     # features stay on disk (RAM is short); loaded per step
    model = A.FireplaceAdapter()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(train_ids) // a.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps + 1, pct_start=0.1)
    rng = random.Random(C.SEED)
    hist = open(os.path.join(C.MODEL, "train_log.jsonl"), "w")
    for ep in range(a.epochs):
        model.train()
        order = train_ids[:]
        rng.shuffle(order)
        tot, parts, n = 0.0, {"bce": 0.0, "dice": 0.0}, 0
        te = time.time()
        opt.zero_grad()
        for i, h in enumerate(order):
            it = load_item(h, C.FEATURES, C.TEACHER)
            x = {k: it[k] for k in ("mask8", "dec16", "enc16")}
            y = it["y"]
            if rng.random() < 0.5:                                    # horizontal flip
                x = {k: v.flip(-1) for k, v in x.items()}
                y = y.flip(-1)
            out = model(**x)
            loss, lp = loss_fn(out, y)
            (loss / a.accum).backward()
            if (i + 1) % a.accum == 0:
                opt.step()
                opt.zero_grad()
                sched.step()
            tot += float(loss)
            n += 1
            for k in parts:
                parts[k] += lp[k]
        rec = {"epoch": ep + 1, "loss": round(tot / n, 4), "bce": round(parts["bce"] / n, 4),
               "dice": round(parts["dice"] / n, 4), "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - te, 1)}
        hist.write(json.dumps(rec) + "\n")
        hist.flush()
        C.log(lg, json.dumps(rec))
    path = os.path.join(C.MODEL, "adapter.pt")
    torch.save(model.state_dict(), path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    open(path + ".sha256", "w").write(f"{sha}  adapter.pt\n")
    cfg = {"arch": "FireplaceAdapter", "width": 96, "drop": 0.1, "params": n_par, "inputs": {
        "mask8": "pixel_level_module.decoder_last_feature avg-pooled 2x (1/8, 256)",
        "dec16": "pixel_level_module.decoder_features at 1/16 (256)",
        "enc16": "pixel_level_module.encoder_features at 1/16 (Swin stage 3, 768)"},
        "outputs": ["fireplace"], "lit": "tour_worker/vision/fire_detect.py inside the predicted mask",
        "student": "shi-labs/oneformer_coco_swin_large@3a263017",
        "teacher": "shi-labs/oneformer_ade20k_swin_large@4a5bac8e fireplace regions (segment.segments_to_regions)",
        "epochs": a.epochs, "lr": a.lr, "accum": a.accum, "seed": C.SEED, "train_photos": len(train_ids),
        "sha256": sha, "train_seconds": round(time.time() - t0, 1)}
    json.dump(cfg, open(os.path.join(C.MODEL, "config.json"), "w"), indent=1)
    C.log(lg, f"saved {path} sha256 {sha} params {n_par}")


if __name__ == "__main__":
    main()
