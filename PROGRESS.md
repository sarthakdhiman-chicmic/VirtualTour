# Progress

Checkpoint: 2026-10-27, branch `feat/beat-sync-v2` (the current pipeline). Update this file at
each checkpoint: what changed, what is decided, what is open. How to run and call the worker is
in README.md.

## Where the worker stands

The worker turns a listing's photos into a cinematic tour video. The backend
(`articulait_backend`, `app/domains/virtual_tour/service.py`) orchestrates three jobs over the
RunPod-compatible API: `tour_segment` (OneFormer masks) -> `tour_plan` (scoring + shot plan) ->
`tour_render` (H.264 MP4). Renders are byte-identical for a given plan.

| Area | State | Where |
| --- | --- | --- |
| Photo scoring: framing, lighting, depth of field, coverage | Done | `scoring/` |
| Room ranking, narrative order, screen time (top 3 rooms 2x, exteriors 1.5x) | Done | `scoring/rooms.py` |
| Cinematic camera engine (tilt, pan, pull-back, slider, crane, rise, aerial, gimbal, push-ins) | Done | `planner/cinematic.py` |
| Procedural motion: curtains, plants, trees, water, lit fires | Done | `motion/` |
| Moving sky (`sky_drift`): clouds drift 2%/s and billow; thin things stay put | Done (2026-10-27) | `motion/sky.py` |
| Beat sync v2: cuts on hero moments, late music start, music edits | Done (2026-10-23) | `music/`, `planner/music_sync_v2.py` |
| Segmentation cache support (`segment_version`, `known_version`) | Done (2026-10-27) | `jobs/segment.py` |
| Camera feel + motion blur (plan 1.6) | Built, flags off, awaiting A/B decision | branch `feat/camera-feel` |
| Direct renderer (OpenCV frames, FFmpeg encode) replacing Remotion | Done | `render/` |
| Docker images, RunPod deploy, GPU timings | Not started | `docker/` (never built) |
| Social formats, branding / overlays, H.265 | Not started | - |
| Depth-based parallax, drone video | Not started | - |

Tests: 325 pass (`pytest -q`, run heavy suites under `tools/safe_run.py`).

## Branches

| Branch | What | State |
| --- | --- | --- |
| `feat/tour-v2` | the line before this checkpoint | base of the branches below |
| `feat/beat-sync-v2` | beat sync v2 + `sky_drift` + segment cache version | this checkpoint |
| `feat/camera-feel` | camera feel + motion blur, flags off | committed, not merged |
| `archive/remotion-renderer` | the old Remotion renderer | archived |

## Decisions to keep

- Room order is product design: sync may retime shots, never reorder rooms.
- Tours are capped at 120 s, with no minimum and no stretching; shots run 2.8-5.5 s.
- Transitions are hard cuts on the music; no zoom punches.
- Beat sync: cut the biggest hero of a run. librosa, not aubio (worse beats, GPLv3+). Any sync
  change must pass `tools/music/sync_regression.py --compare tools/music/sync_baseline.json`.
- Motion runs in the deterministic OpenCV prewarp (no shaders: WebGL was non-deterministic).
  Run `tools/quality/perceptual_quality.py` after any effect change.
- Fire: lit fireplaces only. Sky: 2%/s drift, no cloud shadows.
- Segmentation: OneFormer (COCO, plus an ADE20K pass for fireplaces). Bump `SEGMENT_VERSION` in
  `jobs/segment.py` whenever regions change for the same models: the backend's mask cache
  depends on it. The backend keeps the masks for tours only (never the frontend, never empty
  room or other pipelines).
- HEIC input is not supported (licensing, see LICENSES.md). COCO / ADE20K use needs legal
  sign-off.

## Open

- Camera feel / motion blur: on by default or not (A/B renders delivered 2026-10-22).
- Small static fringes can remain at twig tips while a cloud passes behind them; light thin
  objects (white trim, a white flagpole) against blue sky are not yet detected.
- Production path: build the Docker images, deploy a GPU worker, time a 90 s 4K tour.
- Music library on the admin side (upload, analysis stored once); the backend does not pass a
  track yet, so API tours are silent.
- SOW v12 tour items not started: social exports, branding / MLS versions, depth parallax,
  drone video, budget and cost tracking (see the delivery-status doc).
