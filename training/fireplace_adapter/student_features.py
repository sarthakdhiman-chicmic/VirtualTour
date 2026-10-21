"""Student features: COCO OneFormer (production model) internal features, stored compactly.

  python tools/safe_run.py --cpus 8,10,12,14 --memory 7G -- \
      .venv/bin/python training/fireplace_adapter/student_features.py [--known]

Which photos: out/fireplace_adapter/selected.json (from select_photos.py) or, with --known, the known
local listings. One COCO pass per photo, exactly as production runs it (processor, panoptic task);
a forward hook on the pixel-level module grabs, as fp16:
  mask8   pixel-decoder mask features (256 ch, made at 1/4, average-pooled to 1/8)
  dec16   pixel-decoder multi-scale feature at 1/16 (256 ch)
  enc16   Swin backbone stage 3 at 1/16 (768 ch)
plus `valid` = the un-padded input size, so features map back to the photo.
The same pass also runs the old COCO colour search (fire.find_fires) on the panoptic result, for the
comparison in evaluate.py: <id>_cocofire.png (union of its "fire" regions) + JSON, and caches the
panoptic map (<id>_pan.png, 16-bit segment ids at MAX_SIDE) + segments so evaluate.py can rerun
find_fires with the then-current fire.py without the model.
Resumable; progress in logs/features.log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tour_worker.vision import segment as S  # noqa: E402
from tour_worker.vision import fire_detect as F  # noqa: E402

from tour_worker.config import Settings  # noqa: E402

MODEL = "coco"
SETTINGS = Settings.from_env(dict(os.environ, SEGMENT_MODEL=MODEL))   # MODELS_DIR / ONEFORMER_DIR


class FeatureHook:
    def __init__(self, seg: S.Segmenter):
        self.torch = seg.torch
        self.grab = None
        seg.model.model.pixel_level_module.register_forward_hook(self._hook)

    def _hook(self, _mod, _inp, out):
        self.grab = out

    def take(self) -> dict[str, np.ndarray]:
        o, nnf = self.grab, self.torch.nn.functional
        m4 = o.decoder_last_feature                      # (1, 256, H/4, W/4)
        mask8 = nnf.avg_pool2d(m4, 2, ceil_mode=True)
        dec = sorted(o.decoder_features, key=lambda t: t.shape[-1])   # 1/32, 1/16, 1/8
        dec16 = dec[1]
        enc16 = [e for e in o.encoder_features if e.shape[-1] == dec16.shape[-1]][0]
        self.grab = None
        return {k: v[0].half().cpu().numpy() for k, v in
                {"mask8": mask8, "dec16": dec16, "enc16": enc16}.items()}


def run_one(seg: S.Segmenter, hook: FeatureHook, bgr: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    bgr = C.at_max_side(bgr, S.MAX_SIDE)
    sh, sw = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs = seg.processor(images=rgb, task_inputs=["panoptic"], return_tensors="pt").to(seg.device)
    with seg.torch.no_grad():
        out = seg.model(**inputs)
    feats = hook.take()
    pm = inputs["pixel_mask"][0]
    valid = [int(pm.any(1).sum()), int(pm.any(0).sum())]          # (h, w) of the un-padded input
    feats["valid"] = np.array(valid + list(inputs["pixel_values"].shape[-2:]), np.int32)
    res = seg.processor.post_process_panoptic_segmentation(out, target_sizes=[(sh, sw)])[0]
    pan = res["segmentation"].cpu().numpy()
    fires = F.find_fires(bgr, pan, res["segments_info"], seg.id2label)
    cf = np.zeros((sh, sw), np.uint8)
    for r in fires:
        cf |= r["mask"]
    scene = S.scene_summary(pan, res["segments_info"], seg.id2label, (sw, sh))
    info = {"space": [sw, sh], "input": valid,
            "segments": [{"id": int(x["id"]), "label_id": int(x["label_id"]), "score": float(x.get("score", 1.0))}
                         for x in res["segments_info"]], "coco_fire": bool(fires), "coco_fire_n": len(fires),
            "coco_fire_area": round((cf > 0).sum() / float(sh * sw), 6),
            "coco_fire_bboxes": [r["bbox"] for r in fires],
            "scene": {k: v["area"] for k, v in scene["classes"].items()}}
    info["_pan"] = pan.astype(np.uint16)
    return info, feats, cf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out_dir = os.path.join(C.KNOWN, "features") if a.known else C.FEATURES
    os.makedirs(out_dir, exist_ok=True)
    lg = os.path.join(C.LOGS, "features_known.log" if a.known else "features.log")
    if a.known:
        items = [(f"{name}__{f[:-4]}", os.path.join(d, f)) for name, d in C.KNOWN_LISTINGS.items()
                 if name != "turkey_mix" for f in sorted(os.listdir(d)) if f.endswith(".jpg")]
    else:
        rows = {r["hash"]: r for r in C.manifest()}
        sel = json.load(open(os.path.join(C.OUT, "selected.json")))
        items = [(h, C.photo_path(rows[h])) for h in sel["ids"]]
    todo = [(p, f) for p, f in items if not os.path.exists(os.path.join(out_dir, p + ".json"))]
    C.log(lg, f"start: {len(items)} items, {len(todo)} to do")
    if not todo:
        return
    seg = S.Segmenter(SETTINGS.oneformer_path(), MODEL, threads=a.threads)
    hook = FeatureHook(seg)
    for i, (pid, path) in enumerate(todo):
        t = time.time()
        bgr = C.read_bgr(path)
        info, feats, cf = run_one(seg, hook, bgr)
        np.savez(os.path.join(out_dir, pid + ".npz"), **feats)
        if cf.any():
            cv2.imwrite(os.path.join(out_dir, pid + "_cocofire.png"), cf)
        cv2.imwrite(os.path.join(out_dir, pid + "_pan.png"), info.pop("_pan"))
        info["seconds"] = round(time.time() - t, 2)
        json.dump(info, open(os.path.join(out_dir, pid + ".json.tmp"), "w"))
        os.replace(os.path.join(out_dir, pid + ".json.tmp"), os.path.join(out_dir, pid + ".json"))
        C.log(lg, f"{i + 1}/{len(todo)} {pid[:16]} {info['seconds']}s coco_fire={int(info['coco_fire'])} "
                  f"mask8={feats['mask8'].shape}")
    C.log(lg, "done")


if __name__ == "__main__":
    main()
