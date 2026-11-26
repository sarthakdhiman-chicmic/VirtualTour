# Progress

Checkpoint: 2026-11-26, branch `feat/splat-tours` (the current pipeline, pushed). The client review of
2026-11-20 (23 items) is answered item by item below; departures from it are ADRs 0001-0006. Update this file
at each checkpoint: what changed, what is decided, what is open. How to run and call the worker
is in README.md.

## Where the worker stands

The worker turns a listing into a cinematic tour video. The backend (`articulait_backend`,
`app/domains/virtual_tour/service.py`) orchestrates the jobs over a RunPod-compatible HTTP API
(the worker runs as a plain service on a RunPod A100 pod, `tools/runpod/`):

- photo tours: `tour_segment` (OneFormer COCO masks) -> `tour_plan` (scoring + shot plan) -> `tour_render`
- 3D tours (a Gaussian-splat scene, `.ply` / `.splat`): `scene_label` (once per scene) ->
  `tour_plan` (the 3D planner; photos are the fallback) -> `tour_render`

Renders are byte-identical for a given plan. Every result carries per-stage timings.

| Area | State | Where |
| --- | --- | --- |
| Photo scoring, room ranking, narrative order, 2x hero screen time | Done | `scoring/` |
| Shot lengths by type and pacing (room / hero / detail; Slow / Normal / Fast), one table for photo and 3D tours | Done (2026-11-13, client standards) | `planner/lengths.py`, `planner/allocate.py` |
| Beat sync v2: cuts on hero moments; room shots only lengthen; cuts on the audible attack; music slid under half a frame | Done (2026-11-13) | `planner/music_sync*.py` |
| Cinematic camera engine for photos (2D moves) | Done | `planner/cinematic.py` |
| Continuous cuts (client review items 1-5): every cut measured (speed in % frame width/s, direction change, beat offset) and reported in the render result; moves planned a 1 s handle long at a steady pace and cut by the sync inside it (no time warp); ease in / out only at the tour's ends; one speed band per pacing, matched within 25% at every cut; direction kept within 45 deg (reversals only between interior and exterior, recorded as `cut_reason`); a 3D move that would turn back across a cut is offered played in reverse | Done. Photo tours: every cut within the limits (Turkey Trot 15/15, 2026-11-26). 3D: green house 9-11/13 (local plans; small-room back-outs, ADR 0006) | `render/continuity.py`, `planner/speed_band.py`, `planner/cinematic.py`, `planner/music_sync.py`, `planner3d/moves.py` |
| 3D room changes (item 6): one flight from inside the room through every door into the next, narrow doors included; a matched cut at a door only when the flight fails; the walk takes the nearest room next within a stage | Done (2026-11-26, ADR 0006) | `planner3d/plan.py` `_fly_on`, `planner3d/moves.py` `flight` |
| Motion blur (item 7): 180 deg shutter, from 1 px of motion a frame, on the CPU and GPU photo paths | Done (2026-11-26) | `render/frames.py`, `render/frames_gpu.py` |
| 3D tours: true camera travel through a splat scene (gsplat on the GPU) | Done (2026-11-13) | `planner3d/`, `render/splat_*.py` |
| 3D planner: floor plan, rooms from a point + type, walkways, push / pull, hallway links, reveals, look-back closing, occlusion and room-to-breathe checks; eye level only inside (no rise / crane, item 17); coverage shots show each room's walls at the main shot's bar, never ending on a door | Done | `planner3d/` |
| 3D room discovery on OneFormer labels: en-suite / dressing nooks fold into their bedroom; a sofa among chairs in an open corner is no bedroom; a second living area with its own seating group is its own room; a counter-height table is no dining room; a lone speck of a sink is no bathroom | Done (2026-11-26; green house = the user's v18 rooms, Home Scan = v17's) | `planner3d/discover.py` |
| Scene labels: segmentation classes voted onto every splat; objects; curtain / plant motion from them | Done | `scene/semantics.py`, `scene/motion_label.py`, `jobs/scene_label.py` |
| Segmentation: OneFormer COCO panoptic (one pass, common class names), water v4 refinement; replaced Mask2Former ADE20K (docs/adr/0001) | Done (2026-11-20) | `vision/segment.py`, `vision/refine.py` |
| Procedural motion (photos): curtains, plants, trees, water, sky drift and cloud shadows, one wind per shot (clouds <= 0.7% fw/s, trees and water with it, near-miss trees strengthened); fixtures, blinds and plants outdoors still; fire held still (ADR 0004); pool fittings ripple (ADR 0003) | Done (items 12-16, 2026-11-26) | `motion/`, `planner/perceptual.py` |
| Finishing: title / end cards (the intro text wraps and scales to fit, up to 500 characters), the brokerage logo at the bottom right, room labels (the name only), branded + unbranded (MLS) copies, voiceover with ducking - photo and 3D tours alike | Done (overlays reworked 2026-11-26; 3D demo with a placeholder logo) | `render/overlays.py`, `render/direct.py` |
| Tour controls: presets, pacing, manual order | Done (2026-11-11) | `planner/` |
| Transitions (item 8): hard cuts on the beat; a fade only between daylight and twilight; whip / push removed | Done | `planner/transitions.py` |
| Audio: -14 LUFS, true peak <= -1 dBTP, 3 s fade, checked in `verify()`; no MP4 edit lists (in sync in every player) | Done (2026-11-13) | `render/direct.py`, `render/encode.py` |
| BT.709 conversion and tags for HD / 4K | Done | `render/` |
| 4K master (3840x2160, H.264 Level 5.1) | Done; 3D 4K rendered on the A100 | `render/encode.py` |
| RunPod pod deployment, readable progress logs | Done | `tools/runpod/`, `progress.py` |
| Renderer design doc (`docs/renderer.md`) | Not started | - |
| Social / vertical exports | Parked (vertical framing, 2026-11-05) | - |
| Drone clips (3D, entry / rear) | Built; off in this delivery (ADR 0005) | `render/clip.py` |
| Depth parallax for photo tours (item 10) | Built, parked: layers tear at segmentation boundaries (ADR 0002; `PARALLAX_ENABLED=1` turns it on) | `parallax/` |
| True 4K source (item 11): original uploads, source size and enlargement per shot, smaller sources flagged | Done | `io/images.py`, `planner/planner.py` |
| Music library (item 20): the backend resolves the request's track to the licensed library; the licence is recorded in the plan | Done (backend change pending commit) | `planner/music_sync.py`, backend `music/` |

Tests: `pytest -q` (heavy suites under `tools/safe_run.py`); the 3D render / splat tests need torch.

### Latest timings (A100 80GB pod, cold scene cache, 4K, 2026-11-26 - review item 22)

GPU photo drawing (`PHOTO_BACKEND=gpu`), each frame drawn once for both copies, x264 `medium`,
maths threads capped to the container's CPU quota (14), the scene cache (`SCENE_CACHE_DIR`).

| Tour | `scene_label` | `tour_plan` | `tour_render` | Total |
| --- | --- | --- | --- | --- |
| Turkey Trot (photos, 1536 px sources) | - | 56-107 s | 278-304 s (draw + encode 225-236 s) | ~5-6 min |
| Photo tour from 8K originals | - | 66 s | 377 s (draw + encode 296 s) | ~7.4 min |
| Home Scan (3D) | 108 s | ~14 min (see Open) | 93-161 s | ~17 min |
| Green house (3D) | ~5 min | 138-155 s | 154-254 s | - |

Targets (item 22): under 3 min, within 6 for complex listings. Photo tours are bound by x264 on the
pod's 14 CPUs (the A100 has no hardware encoder).

## Branches

| Branch | What | State |
| --- | --- | --- |
| `feat/splat-tours` | this checkpoint: client-review pass + 3D tours + A/V sync | pushed |
| `feat/beat-sync-v2` | the 2026-10-27 checkpoint | pushed |
| `feat/camera-feel` | camera feel + motion blur; the blur is ported to `feat/splat-tours` (on by default), the camera-feel speed profiles are not (they contradict items 2-3) | local, in the bundle |
| `archive/remotion-renderer` | the old Remotion renderer | archived |

## Decisions to keep

- Room order is product design: sync may retime shots, never reorder rooms. 3D tours: hallways,
  corridors, landings and stairs are links (a pan onto them, then a walk into the next room), not
  stops; the tour opens in the living room.
- Shot lengths: one table for photo and 3D tours (`planner/lengths.py`). The sync only lengthens
  room and hero shots; close-ups flex. Tours 60-90 s (the 90 s cap includes the music sync, 2026-11-20).
- 3D shots: eye level only inside; up to 3 wide shots a room until its walls are seen (coverage is
  a feature: long takes give time back before coverage is dropped for length); no aims above eye
  level; room to breathe at both ends; turns <= 10 deg/s, travel <= 0.8 m/s; room changes are one
  flight through the doors (ADR 0006). Room types come from the planner (`scene_label`), never by hand.
- Transitions are hard cuts on the music; a fade only daylight <-> twilight; no zoom punches.
- Departures from the client's 2026-11-20 review need the client's written approval and an ADR
  (`docs/adr/`): 0001 segmentation, 0002 parallax parked, 0003 pool fittings / spa, 0004 fire,
  0005 no drone clips, 0006 3D room changes and the lens limit.
- Beat sync: librosa, not aubio. Any sync change must pass `tools/music/sync_regression.py
  --compare tools/music/sync_baseline.json` (but see Open: the baseline is stale).
- Audio and video must stay in sync in every player: no MP4 edit lists (`encode.mux_aligned`).
- Segmentation: OneFormer COCO (docs/adr/0001-segmentation-oneformer-coco.md). Bump `SEGMENT_VERSION`
  when regions change. Class names reach every reader through `vision/segment.COMMON`.
- Motion runs in the deterministic OpenCV prewarp for photos; for splats, per-Gaussian offsets.
- The worker runs as a plain service (no Docker images); RunPod is reached via the backend's
  `RUNPOD_ENDPOINT_URL`.

## Open

- Render time (item 22) above target: photo tours ~5-7 min (x264 on 14 CPUs); 3D planning on a scan
  with no ceiling (the Home Scan) ~14 min - most rendered candidates fail the void check; needs a
  cheap pre-estimate of unscanned views so they are rendered last. Maths threads are capped
  (`tour_worker/__init__.py`).
- Depth parallax (ADR 0002): resume after the delivery.
- The sync regression baseline has failed since the client length rules (1d70034): photo-tour
  cases went 2 -> 21 cuts after a harder sound, drops cut 18/18 -> 15/18. Decide the trade-off
  before re-saving the baseline.
- 3D: the hallway link turns only 25 deg (turn-speed cap); a bedroom lost its walk-in to the
  room-to-breathe rule; one label sits in the wrong room (a bed on the balcony).
- Renderer design doc, social exports (vertical framing parked).
- Segmentation trained on our own data (item 19): needs labelled data (ADR 0001).

## 2026-11-26 — client review of 2026-11-20, item by item

| # | Item | State | Where / check |
|---|---|---|---|
| 1-4 | One continuous move; moving through cuts; sync trims, never retimes; one speed band | Done | continuity report per cut; `verify()` |
| 5 | Screen direction within 45 deg | Done; photo tours 15/15; 3D 9-11/13 (small-room back-outs, ADR 0006) | continuity report |
| 6 | 3D through the doorway | Done: one flight through the doors; matched cut only at tight doors (ADR 0006) | `planner3d/plan.py` |
| 7 | Motion blur on | Done: 180 deg, from 1 px/frame, CPU and GPU paths | render warnings `motion_blur_budget` |
| 8 | Transitions | Done: cuts; fade only daylight <-> twilight | `planner/transitions.py` |
| 9 | Shot length | Done: shorten to a hit, never below 3 s | `planner/lengths.py` |
| 10 | Depth on photo tours | Built, parked (ADR 0002) | `parallax/` |
| 11 | True 4K source | Done: source size / enlargement in the plan, smaller sources flagged | plan `source`, `enlargement`, `source_below_output` |
| 12 | One wind per shot | Done | render result `sky` (wind, cloud speed) |
| 13 | Water stays on the water | Done for tile / coping / walls; fittings ripple, spa still (ADR 0003) | - |
| 14 | Cloud shadows | On outdoors (never inside) | render result `sky` (shadow strength) |
| 15 | Fire | Held still (ADR 0004) | - |
| 16 | Fixtures never animated | Done: no pendant-as-curtain, blinds still, plants still outdoors | `planner/config.py`, `planner/perceptual.py` |
| 17 | Eye-level 3D, coverage at the hero bar, lens | Done except the "sliver" for flights through doors (ADR 0006); drone clips off (ADR 0005) | `planner3d/moves.py` |
| 18 | 3D finishing | Done: title / end card, labels, MLS copy, narrative order, skip reasons; corner logo shown with a placeholder | demo render |
| 19 | Segmentation checkpoint | OneFormer COCO (ADR 0001); ADE20K weights out of the production path | `LICENSES.md`, ADR 0001 |
| 20 | Music licence | Done: library track's licence in the plan | plan `music.licence` |
| 21 | 3D scene source | Open: capture details to document | - |
| 22 | Time to render | Measured (timings above); above target - see Open | render `timings` |
| 23 | Records | This file; ADRs 0001-0006 | `docs/adr/` |

## 2026-11-17 — SOW specification sheet: every item started

Against "Articulait Video Tour — SOW Specification Sheet" (Remotion and 2.5D parallax excluded by
the user). Audit + the work of this pass:

| ID | Status | Where |
|---|---|---|
| IN-04 auto room classification | started (untyped photos named from the OneFormer scene summary as TourRoomType values, flagged room_auto) | scoring/autoroom.py |
| PS-07 calibration | started (tool; needs 20-30 listings with human hero picks) | tools/scoring/calibrate.py |
| CM-04 drone clips | done for 3D tours: drone_clips in tour_plan, video_clip shots (entry before the interior, rear after) | render/clip.py, planner3d/plan.py |
| PM-04 shadows | built, off (cloud shadows in sky_drift; user: "lets not do shadows") | motion/sky.py |
| MU-06 licence per track | started (track licence echoed per plan; backend keeps it per job) | contract/jobs.py MusicLicence |
| OR-08 render cache | backend: a completed render with the same plan hash is reused | backend virtual_tour/service.py |
| EX-03 Instagram stills | done (4:5 JPEG a room from each render) | render/stills.py |
| EX-02 Stories 20 s | fixed (the 60 s floor never passes the tour's ceiling) | planner/allocate.py |
| EL-02 static gallery | started (every plan carries its gallery; backend delivers it when a render fails) | jobs/plan.py _gallery |
| BR-03 MLS | started (neutral container metadata on the MLS copy; watermark rules open) | render/direct.py mls_metadata |
| PT-01/02 player + timeline | backend timeline endpoint from the rendered plan | backend GET /virtual-tour/{id}/timeline |
| PT-03 timeline edits | started (re-render with reorder / labels / branding / pacing / preset / trim / music) | backend POST /virtual-tour/{id}/rerender |
| IN-01 / EL-01 / MU-02 backend wiring (3D scene, budget, music library) | not yet | — |
| CM-09 easings on keyframe / compound paths, TR-01 room adjacency for photos | not yet | — |

Deliberate departures from the sheet (earlier user / client decisions): 3 s music fade (MU-04),
2.5 s sync drift (MU-03), 0.8 s fades at <= 0.5x the shorter shot (TR-06), detail shots 0.85x
(TM-06).
