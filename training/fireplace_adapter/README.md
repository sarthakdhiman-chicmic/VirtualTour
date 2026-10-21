# Fireplace adapter (research, not enabled)

Goal: keep ONE segmentation model in production (COCO OneFormer, `tour_worker/vision/segment.py`) and still
find fireplaces, so the lit-fire effect works without loading ADE20K. COCO has no fireplace class;
ADE20K does. The adapter is a small head (about 0.33 M parameters) trained on COCO OneFormer's own
internal features to reproduce ADE20K's fireplace mask (ADE20K = teacher).

Decision 2026-10-21 (user): the adapter predicts the **fireplace mask only**. Whether a fire is lit
stays the job of `tour_worker/vision/fire_detect.py` (`flame_mask` -> `fire_regions`), run inside the
predicted mask exactly as it runs inside ADE20K's mask today. Every fireplace photo, lit or not,
counts as a positive in training.

## Pipeline (all heavy steps go through `tools/safe_run.py --cpus 8,10,12,14 --memory 7G`)

| step | script | output (`out/fireplace_adapter/`) |
|---|---|---|
| list the bucket (read-only) | `s3_list.py` (backend venv) | `s3_index.jsonl` |
| download distinct real photos, by ETag | `s3_fetch.py [--wide]` (backend venv) | `photos/`, `manifest.jsonl` |
| teacher: ADE20K fireplace masks | `teacher.py [--known]` | `teacher/<id>.json`, `<id>_fireplace.png`, flame and zone PNGs |
| recompute "lit" with the current fire.py | `relabel_lit.py [--known]` | `teacher/<id>_lit.json` |
| choose photos and the held-out split | `select_photos.py` | `selected.json`, `phash.json` |
| student: COCO features | `student_features.py [--known]` | `features/<id>.npz` (fp16), `<id>_pan.png`, `<id>.json` |
| train | `train.py` | `model/adapter.pt`, `config.json`, `adapter.pt.sha256` |
| evaluate | `evaluate.py` | `eval/metrics.json`, `eval/*.jpg` |

The known evaluation listings (turkey, downing) are kept out of training. Exact copies (md5) and
near-copies (pHash) are left out, and near-duplicate groups never fall on both sides of the split.

## How it would run in production (sketch; nothing here is enabled)

In `Segmenter.segment()`, with `SEGMENT_MODEL=coco` (the default):

1. Register a forward hook on `self.model.model.pixel_level_module` once, in `__init__`. The COCO
   pass is the same single pass as today. The hook keeps `decoder_last_feature` (1/4,
   average-pooled to 1/8), the 1/16 `decoder_features` level and the 1/16 Swin `encoder_features`
   stage. Also keep the un-padded input size from `inputs["pixel_mask"]`.
2. Run `FireplaceAdapter` on those features. Take the sigmoid, upsample bilinearly to the `space`
   size, threshold at 0.5, and keep connected parts of at least `MIN_PART_AREA`. Each part becomes a
   region with `class_name="fireplace"`.
3. Hand them to `fire.fire_regions(bgr, regions)`. This is the same code path ADE20K takes today
   (`if "fireplace" in CLASS_MAP`). Lit fireplaces become "fire" regions, and unlit ones are dropped.
4. This would replace the `COCO_FIRE_SEARCH` colour search, which stays off.

A flag such as `FIREPLACE_ADAPTER=1` (default off) would control it. The weights would be pinned by
sha256, the same way `ensure_model` pins the OneFormer weights. `tour_worker/vision/segment.py` has **not**
been changed.

### Runtime per photo (CPU, measured on this machine)
- COCO OneFormer pass: about 11-13 s with 4 threads. This cost is unchanged because the pass
  already runs.
- Feature hook: no extra compute. The tensors already exist during the forward pass.
- Adapter head: about 0.2 s with 1 thread on a 100x150 grid (much less with 4 threads), plus
  upsampling.
- fire.py inside the mask: the same cost as on the ADE20K path today (it works on a crop of the
  mask).
- No ADE20K pass: this saves about 7-13 s and about 1-2 GB of resident model per photo batch when
  only one model is loaded.

## Risks
- **Small training set.** The bucket only has about 1,027 distinct real photos, and about 7% of
  them show a fireplace. Many are near-duplicates of the same few rooms. Unusual fireplace styles
  will be under-represented: linear gas inserts, double-sided fireplaces, outdoor fire pits and
  stoves.
- **Teacher errors are learned.** ADE20K's own fireplace mistakes (dark doorways, TV niches) are
  reproduced.
- **Lit detection is not solved by the adapter.** Lit fires depend on fire.py. ADE20K + fire.py
  already missed Downing's glass-fronted gas fires (03, 06, 11, 17), because the flame-patch filter
  rejects pale flames behind glass. The adapter can only match the teacher there. fire.py is being
  revised separately, so `relabel_lit.py` recomputes lit before every evaluation.
- **Feature hook coupling.** The hook depends on the internals of the HF `OneFormerForUniversalSegmentation`
  (`model.pixel_level_module` output fields). A transformers upgrade can break it, so it needs a
  pinned version and a test.
- **Licensing.** The adapter is distilled from ADE20K OneFormer, and ADE20K's training images are for
  non-commercial research and education (MIT CSAIL terms). The adapter therefore inherits the
  legal question still pending for both models: COCO's Flickr images, and ADE20K. Sign-off is
  needed before production.
