# Progress

Checkpoint: 2026-11-13, branch `feat/splat-tours` (the current pipeline, pushed). Update this file
at each checkpoint: what changed, what is decided, what is open. How to run and call the worker
is in README.md.

## Where the worker stands

The worker turns a listing into a cinematic tour video. The backend (`articulait_backend`,
`app/domains/virtual_tour/service.py`) orchestrates the jobs over a RunPod-compatible HTTP API
(the worker runs as a plain service on a RunPod A100 pod, `tools/runpod/`):

- photo tours: `tour_segment` (Mask2Former masks) -> `tour_plan` (scoring + shot plan) -> `tour_render`
- 3D tours (a Gaussian-splat scene, `.ply` / `.splat`): `scene_label` (once per scene) ->
  `tour_plan` (the 3D planner; photos are the fallback) -> `tour_render`

Renders are byte-identical for a given plan. Every result carries per-stage timings.

| Area | State | Where |
| --- | --- | --- |
| Photo scoring, room ranking, narrative order, 2x hero screen time | Done | `scoring/` |
| Shot lengths by type and pacing (room / hero / detail; Slow / Normal / Fast), one table for photo and 3D tours | Done (2026-11-13, client standards) | `planner/lengths.py`, `planner/allocate.py` |
| Beat sync v2: cuts on hero moments; room shots only lengthen; cuts on the audible attack; music slid under half a frame | Done (2026-11-13) | `planner/music_sync*.py` |
| Cinematic camera engine for photos (2D moves) | Done | `planner/cinematic.py` |
| 3D tours: true camera travel through a splat scene (gsplat on the GPU) | Done (2026-11-13) | `planner3d/`, `render/splat_*.py` |
| 3D planner: floor plan, rooms from a point + type, walkways, push / pull, hallway links, reveals, look-back closing, occlusion and room-to-breathe checks | Done; green-house v3 is the shot standard | `planner3d/` |
| Scene labels: Mask2Former classes voted onto every splat; objects; curtain / plant / flame motion from them | Done | `scene/semantics.py`, `scene/motion_label.py`, `jobs/scene_label.py` |
| Segmentation: Mask2Former ADE20K panoptic (one pass, fireplaces included), water v4 / flame refinements | Done (2026-11-12) | `vision/segment.py`, `vision/refine.py` |
| Procedural motion (photos): curtains, plants, trees, water, lit fires, sky drift | Done | `motion/` |
| Finishing: title / end cards, corner logo, room labels, branded + unbranded (MLS) copies, voiceover with ducking | Done (2026-11-11) | `render/overlays.py`, `render/direct.py` |
| Tour controls: presets, pacing, manual order | Done (2026-11-11) | `planner/` |
| Content-chosen transitions (cut default, whip / push at a stage change, fade for dusk; 3D: cuts, fades between parts of the home) | Done | `planner/transitions.py`, `planner3d/plan.py` |
| Audio: -14 LUFS, true peak <= -1 dBTP, 3 s fade, checked in `verify()`; no MP4 edit lists (in sync in every player) | Done (2026-11-13) | `render/direct.py`, `render/encode.py` |
| BT.709 conversion and tags for HD / 4K | Done | `render/` |
| 4K master (3840x2160, H.264 Level 5.1) | Done; 3D 4K rendered on the A100 | `render/encode.py` |
| RunPod pod deployment, readable progress logs | Done | `tools/runpod/`, `progress.py` |
| Renderer design doc (`docs/renderer.md`) | Not started | - |
| Social / vertical exports | Parked (vertical framing, 2026-11-05) | - |
| Drone video, cloud shadows | Not started | - |
| Depth parallax for photo tours | Dropped for photos (2026-11-02: bends straight lines); 3D tours give real camera travel | - |

Tests: 476 pass, 4 skipped (`pytest -q`; heavy suites under `tools/safe_run.py`).

### Latest timings (A100 pod, green house, 64.9 s tour)

| Stage | 1080p | 4K |
| --- | --- | --- |
| `scene_label` (once per scene, cold) | 155 s (model load 62 s, labelling 27 s) | - |
| `tour_plan` planning | 77-105 s | 257 s (CPU; see Open) |
| `tour_render` draw + encode | 143 s | 335 s (GPU rasterise 284 s) |

## Branches

| Branch | What | State |
| --- | --- | --- |
| `feat/splat-tours` | this checkpoint: client-review pass + 3D tours + A/V sync | pushed |
| `feat/beat-sync-v2` | the 2026-10-27 checkpoint | pushed |
| `feat/camera-feel` | camera feel + motion blur, flags off | local, not merged |
| `archive/remotion-renderer` | the old Remotion renderer | archived |

## Decisions to keep

- Room order is product design: sync may retime shots, never reorder rooms. 3D tours: hallways,
  corridors, landings and stairs are links (a pan onto them, then a walk into the next room), not
  stops; the tour opens in the living room.
- Shot lengths: one table for photo and 3D tours (`planner/lengths.py`). The sync only lengthens
  room and hero shots; close-ups flex. Tours 60-120 s.
- 3D shots: the green-house v3 tour is the standard. One shot a room (a hero room's second angle
  only with room to breathe), at most 2 close-ups a tour, no pans inside rooms, no aims above eye
  level, >= 2.2 m clear in front at both ends of a shot, turns <= 10 deg/s, travel <= 0.8 m/s.
- Transitions are hard cuts on the music by default; no zoom punches.
- Beat sync: librosa, not aubio. Any sync change must pass `tools/music/sync_regression.py
  --compare tools/music/sync_baseline.json` (but see Open: the baseline is stale).
- Audio and video must stay in sync in every player: no MP4 edit lists (`encode.mux_aligned`).
- Segmentation: Mask2Former only (OneFormer dropped). Bump `SEGMENT_VERSION` when regions change.
  ADE20K's licence is under legal review (LICENSES.md).
- Motion runs in the deterministic OpenCV prewarp for photos; for splats, per-Gaussian offsets.
- The worker runs as a plain service (no Docker images); RunPod is reached via the backend's
  `RUNPOD_ENDPOINT_URL`.

## Open

- Planning on the pod is ~8x slower than on a dev machine (257 s vs 32 s): most likely numpy
  threads oversubscribing the container's ~14-CPU quota (nproc says 128). Check with
  OPENBLAS/OMP threads = 1; then cap the maths threads in the worker.
- The sync regression baseline has failed since the client length rules (1d70034): photo-tour
  cases went 2 -> 21 cuts after a harder sound, drops cut 18/18 -> 15/18. Decide the trade-off
  before re-saving the baseline.
- 3D: the hallway link turns only 25 deg (turn-speed cap); a bedroom lost its walk-in to the
  room-to-breathe rule; one label sits in the wrong room (a bed on the balcony).
- Photo-tour 4K timing run on the A100 from a cold cache (the client's next-pass ask) not yet done.
- Renderer design doc, cloud shadows, drone video, social exports (vertical framing parked).
- Camera feel / motion blur: on by default or not.
