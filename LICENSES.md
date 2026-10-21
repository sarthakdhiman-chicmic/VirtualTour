# Licences

## OneFormer (tour_segment)
- **Weights:** `shi-labs/oneformer_coco_swin_large` and `shi-labs/oneformer_ade20k_swin_large`,
  MIT. The revisions and sha256 are pinned in `tour_worker/vision/segment.py`.
- **COCO** (the primary model's training data): the annotations are CC BY 4.0. The images are
  under their Flickr licences, and COCO disclaims them. This dataset question needs a legal
  sign-off before production use.
- **ADE20K** (the fireplace-aware model): the dataset images are for non-commercial research and
  education (MIT CSAIL terms). The product owner accepted this, pending legal review.

## Music analysis (the `music` extra, offline)
- `librosa` (ISC), `scipy` (BSD-3-Clause), `soundfile` (BSD-3-Clause).

## Codecs
- H.264 is encoded with ffmpeg / x264, which are system tools and not shipped in this package.
  Codec patents are a separate question from software licences.

## Removed (2026-10)
The Remotion renderer (Remotion licence: paid "Automators" tier for automated rendering) and the
Depth Anything V2 parallax experiment are no longer part of the worker. Both remain on the branch
`archive/remotion-renderer`.
