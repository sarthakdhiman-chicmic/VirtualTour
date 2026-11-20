# Licences

## Mask2Former (tour_segment, scene_label)
- **Code:** Mask2Former (Meta) is MIT.
- **Production checkpoint (required, SOW §3.2):** our own panoptic Mask2Former, Swin-B backbone
  pretrained on ImageNet-1K, trained on our data with the motion classes added. Not yet trained;
  it replaces the interim weights below in both photo tours and 3D labelling.
- **Interim weights, development only:** `facebook/mask2former-swin-large-ade-panoptic` (revision
  and sha256 pinned in `tour_worker/vision/segment.py`; model card licence "other"). They are
  trained on **ADE20K**, whose images are licensed for non-commercial research and education (MIT
  CSAIL terms). **Acceptance of the ADE20K weights has not been given** (client review 2026-11-20,
  item 19); the licence is under review, and these weights must not be in the production path.
- Replaced OneFormer (COCO + ADE20K, two passes) on 2026-11-11. `training/fireplace_adapter/` is
  an offline OneFormer experiment and is not used by any job.

## Music tracks (SOW §5.3C, MU-06)
Each track's licence comes from the music library with the track (`music_track.licence`) and is
recorded in the plan (`music.licence`) and echoed in the plan job's `music_licence`; a track sent
without one is warned `music_licence_missing`. The demo track `poradovskyi` ("Real Estate Luxury
Music", Pixabay download 530233) is under the Pixabay Content License: commercial use allowed, no
attribution, no standalone redistribution of the audio. Its use inside an automated video product
is not yet confirmed and it is used for demos only.

## 3D tours (plan 1.6)
- **gsplat** (Apache-2.0, nerfstudio-project): the CUDA rasteriser for Gaussian splat scenes
  (`tour_worker/render/splat_shot.py`). It is imported only on the GPU host, from the RunPod
  image's pinned requirements; the worker does not install it.
- **Not used, on purpose:** Inria's `diff-gaussian-rasterization` and the original 3DGS code
  (non-commercial research licence). The CPU reference rasteriser (`render/splat_cpu.py`) is our
  own code, written from the published method, and uses only `torch` (BSD-3-Clause) and `numpy`.
- Scene files (`.ply`, `.splat`) are the customer's data; no third-party scene or model ships here.

## Music analysis (the `music` extra, offline)
- `librosa` (ISC), `scipy` (BSD-3-Clause), `soundfile` (BSD-3-Clause).

## Room-label fonts (`tour_worker/render/fonts`)
All under the SIL Open Font License 1.1; each font's licence text is next to it (`OFL-*.txt`).
Bundling them in the worker and drawing them into videos is allowed; the fonts themselves may not
be sold on their own.
- **Sacramento** (default): Brian J. Bonislawsky, Astigmatic.
- **Mr Dafoe:** Alejandro Paul (Sudtipos).
- **Pacifico:** The Pacifico Project Authors.
- **Kaushan Script:** Pablo Impallari, Igino Marini.
- **Lato** (the room index, the "Virtually Staged" label and the title / end card text): Łukasz
  Dziedzic (tyPoland).

## Codecs
- H.264 is encoded with ffmpeg / x264, which are system tools and not shipped in this package.
  Codec patents are a separate question from software licences.

## Not supported: HEIC / HEIF photos (2026-10)
The ready-made decoder, `pillow-heif`, ships binary wheels under GPLv2 because they bundle the x265
encoder. Its own code is BSD-3, and the decoders are LGPLv3 (libheif, libde265). A decode-only
build would avoid the GPL: pillow-heif built from source against system libheif + libde265, with no
x265. That option was deferred, so HEIC photos are left out of tours for now. HEVC decoding also
carries its own patent question, separate from the software licences.

## Removed (2026-10)
The Remotion renderer (Remotion licence: paid "Automators" tier for automated rendering) and the
Depth Anything V2 parallax experiment are no longer part of the worker. Both remain on the branch
`archive/remotion-renderer`.
