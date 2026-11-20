# ADR 0001: OneFormer COCO replaces Mask2Former ADE20K for segmentation

- **Status:** Accepted by the product owner, 2026-11-20. A departure from client review item 19, so
  it goes to the client for written approval (review item 23).
- **Date:** 2026-11-20
- **Scope:** `tour_segment` (photo tours: motion masks, scene summaries, automatic room typing) and
  `scene_label` (3D tours: the classes voted onto every splat).

## Context

Since 2026-11-11 the worker has segmented with Mask2Former Swin-L trained on ADE20K panoptic
(`facebook/mask2former-swin-large-ade-panoptic`). ADE20K was chosen over COCO for one thing COCO
does not have: a fireplace class, which the lit-fire effect needed to find flames.

Three things have changed:

1. **The fire effect is off.** Lit fires are held still (client review 2026-11-20, item 15), so the
   fireplace class is no longer needed.
2. **Mask2Former is no longer shared.** The platform's other pipelines have moved to models that do
   not take Mask2Former's masks, so the tour worker is its only user.
3. **Our own checkpoint is far off.** Review item 19 asks for our own panoptic Mask2Former (Swin-B)
   trained on our data with the motion classes added. Collecting and labelling that data and
   training it will take a long time; tours need a segmenter now.

## Decision

Segment with **OneFormer Swin-L trained on COCO panoptic** (`shi-labs/oneformer_coco_swin_large`,
revision `3a263017ca5c`, sha256-pinned in `tour_worker/vision/segment.py`), for photo tours and 3D
labelling alike.

- The Segmenter renames COCO's labels once, at load (`vision/segment.COMMON`): `couch` -> `sofa`,
  `ceiling-merged` -> `ceiling`, `water-other` -> `water`, `fence-merged` -> `fence`,
  `window-other` -> `windowpane`, the `wall-*` / `floor-*` classes -> `wall` / `floor`, and so on.
  Scene summaries, motion regions and 3D labels all use these names.
- Motion classes: curtain, window blind, potted plant, tree, water (sea and river too), sky.
- The fireplace / flame pass is removed. The water refinement (`vision/refine.py`, surface only) stays.
- `SEGMENT_VERSION` is 4: cached `tour_segment` results from Mask2Former are not reused.

## Alternatives considered

- **Keep Mask2Former ADE20K.** Its one advantage over COCO was the fireplace class, which is no
  longer used.
- **Train our own Mask2Former now (item 19).** Not possible on the current timeline; it stays the
  long-term target.
- **Mask2Former COCO panoptic.** The same COCO classes in the architecture item 19 names, but it
  brings no benefit to the worker now that no other pipeline shares Mask2Former.

## Consequences

- **Narrower indoor vocabulary.** COCO has 133 classes and no stove, hood, kitchen island,
  bathtub, shower, desk, stool, armchair, wardrobe, painting, lamp or swimming-pool class. The
  automatic room typing (`scoring/autoroom.py`) was rewritten for what COCO sees: kitchens from an
  oven, microwave or toaster, or a fridge at a counter; bathrooms from a toilet or a towel by a
  sink; offices from a computer, keyboard or mouse; pools from water; balconies from COCO's
  fence class, which includes railings. Its accuracy must be checked again on real listings.
- **3D labels.** Scenes labelled with Mask2Former keep working (the 3D vocabulary only grew:
  `toaster`, `keyboard`, `mouse` were appended), but they must be relabelled (`scene_label`) to
  get OneFormer's classes. The 3D room discovery, framing and views lose the ADE-only cues; the
  ones COCO keeps (bed, sofa, table, counter, oven, sink, fridge, microwave, toilet, TV) still
  drive them. A TV wall can no longer be labelled a "painting" (review item 16): COCO has no
  painting class.
- **Water.** COCO's water masks cover the surface, not the coping or walls (seen when COCO was the
  model, October 2026), which helps review item 13.
- **Speed.** OneFormer was about 8.7x slower than Mask2Former on CPU in the November benchmark;
  production runs on the A100, where it must be timed again (review item 22).
- **Cache.** Every photo is segmented again once (`SEGMENT_VERSION` 4).
- **Reversible.** `MODELS` in `vision/segment.py` holds one entry per checkpoint; our own
  Mask2Former goes in beside it when it exists, with its labels mapped through `COMMON`.
