# tour-worker

The Articulait cinematic tour worker. It runs three jobs for the backend, on RunPod (serverless)
or locally. A tour is `tour_segment` -> `tour_plan` -> `tour_render`; the backend only orchestrates.

- **`tour_plan`**: scores a tour's photos (coverage, framing, lighting, depth of field, motion
  signals), picks heroes and the room order, and plans every shot: lengths, camera moves (the
  cinematic engine), motion regions from `tour_segment`'s masks and beat sync to a music track.
  It returns the **plan** (JSON, validated with the renderer's own checks) and its `plan_hash`.
- **`tour_render`**: renders a tour **plan** (made by `tour_plan`) into an H.264 MP4.
  It fetches the photos, motion-region masks and music, draws every frame with OpenCV (the
  camera moves, beat-sync retiming and zoom punches, and perceptual motion: curtains, plants,
  trees, water and fire), encodes with ffmpeg, checks that the file plays smoothly, and uploads
  `<output_prefix>master.mp4`. The same plan always gives a byte-identical file. A plan with
  `output.hevc` also gets an H.265 copy, `<output_prefix>master_hevc.mp4`. A plan with
  `output.room_labels` draws each room's name; with `output.mls_copy` it also gets
  `<output_prefix>master_mls.mp4`, the same video without the room labels.
- **`tour_segment`**: runs Mask2Former on a tour's photos before planning. It uploads one PNG mask
  per object that can move (curtain, blinds, potted plant, tree, water, lit fire) and returns
  JSON only: the regions and a scene summary per photo.

## Install

System tools: **`ffmpeg` and `ffprobe`** on `PATH` (the renderer pipes frames to ffmpeg, and the
playback checks use ffprobe). Python 3.10 or later.

```bash
uv sync                          # the render job, plus the dev group (pytest)
uv sync --extra segment          # + Mask2Former on CPU PyTorch (dev machines)
uv sync --extra segment-gpu      # + Mask2Former on CUDA 12.6 PyTorch (GPU host)
uv sync --extra music            # + music analysis (librosa 0.10.2.post1, pinned)
uv sync --extra http             # + the HTTP service (fastapi, uvicorn)
pip install -e ".[segment,dev]"  # the same with pip
```

`segment` and `segment-gpu` exclude each other: they pull torch from different indexes.

## Jobs and their JSON

RunPod calls `handler.handler(event)` with `event["input"]` set to the request. Every job goes
through `tour_worker.jobs.router.run_job`, which validates the request
(`tour_worker/contract/jobs.py`) and returns either the job's result or **one failure shape**:

```json
{"status": "failed", "error": {"code": "invalid_plan", "message": "invalid_plan: ..."}}
```

| code | when |
|---|---|
| `invalid_job_type` | `job_type` is not `tour_render`, `tour_segment` or `tour_plan` |
| `invalid_request` | the request is missing required fields (for example `output_prefix`) |
| `invalid_plan`, `unsupported_camera_type`, `unsupported_easing` | the plan failed validation (`tour_worker/contract/plan.py`); for `tour_plan`, the plan it made did |
| `image_fetch_failed` | a shot's photo could not be fetched; for `tour_plan`, no photo could be fetched and decoded |
| `planning_failed` | `tour_plan`: scoring or planning raised (a bug or an input the planner cannot handle; not transient) |
| `prewarp` | a photo cannot be read, or its aspect ratio does not match the plan |
| `encode_failed`, `verification_failed` | ffmpeg failed, or the file failed the playback checks (codec, size, fps, duration, faststart, keyframe gap, VBV, level) |
| `unknown` | anything unexpected |

### `tour_render`

Request (full example: `examples/tour_render.json`):

```json
{
  "job_type": "tour_render",
  "job_id": "optional",
  "plan": { "plan_version": "1.5", "...": "the plan, exactly as tour_plan returned it (hashed)" },
  "output_prefix": "s3://bucket/virtual_tour/<job>/",
  "image_urls": { "<shot image_uri or mask_uri>": "<URL to fetch it from, e.g. presigned https>" }
}
```

- `output_prefix` is **required**. The video is written to `<output_prefix>master.mp4`.
- **Delivery (client review 2026-11-05), all checked by `render/probe.py verify()`:**
  - **Colour:** BT.709 conversion and tags.
  - **Audio:** normalised to -14 LUFS with a true peak at or below -1 dBTP. This is a two-pass,
    linear gain applied once and stream-copied into every copy.
  - **Music fade:** 3 s at the end.
  - **Cuts:** on the beat, with no lead.
- `output.hevc` (optional, plan field): also writes an H.265 copy to
  `<output_prefix>master_hevc.mp4` (the SOW's optional H.265).
  - The same drawn frames go to a second encoder (x265) in the same ffmpeg process, so nothing is
    drawn twice and the copy is not a re-encode of the H.264 file.
  - HEVC Main, tagged `hvc1` for Apple players, at the master's level, bitrate cap and keyframe
    spacing. It passes the same playback checks and is byte-identical for the same plan.
  - Each render worker needs more memory (x265: about 0.43 GB at 1080p, 1.35 GB at 4K), so fewer
    workers start on the same machine.
- `image_urls` is optional. It maps a URI in the plan (a shot's `image_uri`, a region's
  `mask_uri`, or `music.uri`) to the URL the worker should fetch instead.
- Plan versions 1.0 to 1.5 are accepted:
  - 1.0: ken_burns cameras and cuts only.
  - 1.1: adds fades.
  - 1.2: adds easing.
  - 1.3: adds `keyframe_path` and `compound` cameras, the `rotate` projection and roll.
  - 1.4: adds `motion_regions`, with the effects `fabric_sway`, `foliage_sway`, `canopy_flutter`, `calm_water`, `water_ripple`, `fire_flicker` and `sky_drift`.
    - `sky_drift` (2026-10-27): the clouds drift with the shot's wind (2% of the photo width per second at the top) and slowly billow (`amplitude_px` / `freq_hz`).
    - The sky is a layer under everything. It is refined to the pixel, so roofs and leaves stay put, and the sky hidden behind them is filled with nearby cloud texture.
    - The segmentation's sky mask takes in thin things whole: twigs, loose leaves, wires, a weathervane, part of a gutter. Anything darker than the clear blue sky nearby that hangs off the foreground is kept out, so it stays still instead of drifting with the clouds.
    - At their soft edges only the sky's share of each pixel changes, so no outline of the old sky shows when a cloud passes behind them.
    - A sky whose mask cannot be read stays still, with a warning.
  - 1.5: adds the `music` block, `time_warp` and `punches`.

  `provenance.plan_hash` must be the sha256 of the plan's canonical JSON with the hash itself
  removed (`compute_plan_hash`).

Result:

- **Room labels** (`output.room_labels`: `font`, `scale`, `placement`; `render/overlays.py`):
  each shot with a `caption` (a room's first shot) gets the room's name in a script font after a
  small gold index and a thin rule, e.g. "02 | Kitchen", over a soft gradient.
  - **Animation:** 0.4 s after the room's first cut the rule draws down and the index fades in.
    The name writes on left to right over 0.9 s. It holds for at most 3.6 s, or until 0.35 s
    before the room ends, then fades out over 0.5 s. A room too short to write its name out gets
    no label.
  - **`font`:** `sacramento` (default), `mr_dafoe`, `pacifico` or `kaushan_script`. The fonts ship
    in `tour_worker/render/fonts` (SIL OFL, `LICENSES.md`).
  - **`scale`:** 0.6–1.0 of the reviewed maximum size; the default is 0.8. Sizes are set for a
    1080 px short side and scale with the output.
  - **`placement`:** `bottom_left` (default), `bottom_center`, `bottom_right`, `top_left`,
    `top_center` or `top_right`. Right placements mirror the label ("Kitchen | 02"); top ones
    move the gradient to the top.
- **"Virtually Staged":** every shot with `staged: true` shows the disclosure at the bottom right
  for the whole shot (bottom left when the labels sit bottom right). It is drawn with or without
  room labels.
- `output.mls_copy` (optional): also writes `<output_prefix>master_mls.mp4`. The same drawn frames
  go to a second x264 before the room labels are added; the staged disclosure stays. It passes
  the same checks; each render worker budgets a second encoder's memory.

```json
{"status": "completed", "video_uri": "s3://bucket/virtual_tour/<job>/master.mp4",
 "duration_s": 71.6, "width": 1920, "height": 1080, "fps": 30, "codec": "h264",
 "render_seconds": 412.3, "plan_hash": "...", "warnings": ["motion_region_dropped:s17:seg_002:faint_0.3px"],
 "hevc_uri": null, "mls_uri": null, "error": null}
```

A warning never fails the job. These are the warnings:
- `image_resized:<shot>:<w>x<h>-><w>x<h>`
- `motion_region_dropped:<shot>:<region>:<reason>`, where the reason is one of `missing`, `unreadable`, `aspect`, `empty`, `mismatch` or `faint_<px>px`
- `motion_mask_fetch_failed:...`
- `music_fetch_failed`
- the photo-size notice from image preparation

### `tour_segment`

Request (example: `examples/tour_segment.json`):

```json
{"job_type": "tour_segment", "job_id": "job1",
 "images": [{"photo_id": "living.jpg", "uri": "s3://bucket/listings/L123/photos/living.jpg"}],
 "output_prefix": "s3://bucket/listings/L123/segments/job1/"}
```

Result:
- One region per moving object. Its mask is uploaded to `<output_prefix><photo_id>/<object_id>.png`.
- A photo that cannot be fetched or segmented gets `null` and a `segment_failed:<photo_id>:<error>` warning. The tour still renders, without motion on that photo.

```json
{"status": "completed",
 "results": {"living.jpg": [{"object_id": "seg_000", "class_name": "curtain", "confidence": 0.98,
                             "bbox": [x1, y1, x2, y2], "space": [w, h],
                             "mask_url": "s3://.../segments/job1/living.jpg/seg_000.png"}]},
 "scenes": {"living.jpg": {"space": [w, h], "classes": {"bed": {"area": 0.12, "count": 1,
                                                               "bbox": [...], "centroid": [...]}}}},
 "warnings": [], "model": "facebook/mask2former-swin-large-ade-panoptic@ae53ed7ab3", "seconds": 13.2}
```

`bbox` and the masks are in `space` pixels: the photo scaled to at most 2048 px on its long side.
The scene summary lists every Mask2Former class in the photo; the planner uses it to choose camera
moves.

### `tour_plan`

Request (example: `examples/tour_plan.json`; contract `TourPlanRequest` in
`tour_worker/contract/jobs.py`):

```json
{"job_type": "tour_plan", "job_id": "job1", "listing_id": "L123",
 "photos": [{"photo_id": "img_0", "uri": "s3://bucket/listings/L123/photos/living.jpg", "room_type": "Living Room"}],
 "image_urls": {"s3://bucket/listings/L123/photos/living.jpg": "https://...presigned..."},
 "segments": {"img_0": [{"object_id": "seg_000", "class_name": "curtain", "...": "a tour_segment region"}]},
 "scenes": {"img_0": {"space": [2048, 1365], "classes": {"...": "..."}}},
 "music_track": {"track_id": "solarflex.mp3", "s3_uri": "s3://bucket/music/solarflex.mp3",
                 "sha256": "...", "analysis": {"...": "tour_worker.music analysis"}},
 "options": {"preset": "property_showcase", "budget_blocked": false,
             "output_width": 1920, "output_height": 1080, "max_total_s": 120}}
```

- `listing_id` (required) goes into the plan and seeds the motion regions.
- `photos` (at least one, unique `photo_id`s). `uri` is the photo's canonical URI: it becomes the
  shot's `image_uri` and is hashed with the plan. `room_type` is the user's choice at upload, or null.
- `image_urls` (optional) maps a photo's `uri` to the URL to fetch instead, as for `tour_render`;
  fetch URLs never enter the plan.
- `segments` / `scenes` (optional): `tour_segment`'s `results` / `scenes`, by `photo_id`. Without
  them there are no motion regions, and camera moves come from the photos' saliency alone. A photo
  whose segments are `null` gets a `perceptual_unavailable:<shot>` plan warning.
- `music_track` (optional): a music-library track; shots are cut on its beats (plan 1.5).
  - **Analysis v4 (sync v2):** with a v4 analysis (`heroes`, `phrases`), cuts go on the music's hero
    moments.
    - tier 1: drops;
    - tier 2: entries, dropouts, whooshes, big hits.

    How it works:
    - The strongest of a close run gets the cut; the others get a zoom punch sized by their tier.
    - The music may start at a phrase or a change in the first 45% of the track
      (`music.start_s`). Opening on a change is preferred; opening just before one is avoided.
    - **Music edits:** the music may be trimmed to fit the tour, up to 2 edits.
      - Whole bars come out so a drop lands on a cut, or whole phrases of repetition are skipped
        so a later drop arrives inside the tour.
      - A join is used only where both sides sound alike (analysis `bar_prints`), as seamless as
        two neighbouring bars.
      - The plan's `music.segments` lists the track pieces played back to back. `tour_render`
        joins them with a `crossfade_s` crossfade centred on each join, so timing stays exact.
      - `music.edits` says which bars came out and why.
    - **Cut quality:** a cut prefers a hard sound, and avoids landing up to a beat after a harder
      beat it could have taken.
    - The plan's `music.hooks` lists every hero with `cut` / `accent` / `covered` / `missed` and
      its tier.
  - **Fallback:** a v3 analysis, or a v2 sync with no solution, syncs as before
    (`music_sync_v2_fallback:<reason>`).
- `options` (optional, no other keys): `max_total_s` is the tour's ceiling; `budget_blocked`
  makes a slideshow. Everything else is the planner's default (`tour_worker/planner/config.py`).
  - **Shot lengths (client review 2026-11-05):**
    - Each room gets a budget of `room_base_s` (4 s) x its screen-time multiplier x the pacing. The
      multiplier is 2x for the best living room, kitchen and primary bedroom, and 1.5x for exteriors.
    - The budget is split into shots of 3-8 s.
    - The opening shot is at least 5 s and the closing shot at least 5 s.
    - Tours land at 60-120 s by stretching shots up to 8 s, never by adding padding shots. With Fast
      pacing the floor is 45 s.
    - Beat sync scales each shot by the music's loudness under it: x1.3 through a breakdown,
      x0.85 into a drop.
  - **Transitions** (`planner/transitions.py`):
    - **Cut on the beat:** within a part of the home.
    - **Whip:** into a new part of the home when the incoming camera moves sideways.
    - **Push:** into a new part of the home when it moves into depth.
    - **Fade:** daylight to twilight.
    - Motion transitions last 0.33 s and are centred on the cut, so the cut stays on the beat.
  - `max_bytes` (optional): a platform's file-size limit, e.g. Instagram Reels' 100 MB. It
    becomes the plan's `output.max_bytes`; the encoder caps its bitrate so the file fits, and
    the playback check fails a file over it. Left out of the plan when unset.
  - `hevc` (optional, default false): also render the H.265 copy (the plan's `output.hevc`,
    left out of the plan when false).
  - `room_labels` (optional): `{"font", "scale", "placement"}` (defaults `sacramento`, 0.8,
    `bottom_left`). The plan gets `output.room_labels`, and each room's first shot a `caption`:
    its display name (`planner/captions.py`, e.g. `primary_bedroom` -> "Primary Suite").
    Left out of the plan when unset. An unknown font or placement, or a scale outside 0.6–1.0,
    is `invalid_request`.
  - `mls_copy` (optional, default false): also render the MLS copy (the plan's `output.mls_copy`).
    It is the unbranded version: no room labels, agent card or logo.
  - `preset` (default `property_showcase`): `planner/tour_presets.py`.
    - **`property_showcase`:** Normal pacing.
    - **`just_listed`:** Fast pacing, "Just Listed" above the address, and the music opening in a
      high-energy section if one fits.
    - **`lifestyle`:** Slow pacing and 1.5x time for exteriors, outdoor living and amenities.
  - `pacing` (optional): `slow`, `normal` or `fast`, scaling every room's time across the 3-8 s
    shot range. It overrides the preset's pacing.
  - `photo_order` (default `narrative`): `manual` plays the rooms and photos in request order
    (drag-to-reorder).
  - `branding` (optional): `{kicker?, address?, locality?, title?, agent?: {name, phone, email,
    photo_uri}, logo_uri?, corner_logo?}`.
    - The title card, end card and corner logo are drawn by `render/overlays.py`.
    - The images are fetched like the photos.
    - With an end card, the final shot holds at least 6 s.
  - `voiceover` (optional): `{uri, balance}`. Narration over the tour; the music ducks under it,
    more as `balance` (0-1) rises.
  - A photo with `"staged": true` (a virtually staged generation) marks its shots `staged`.
  - Frame size picks the encode (`render/encode.py` `video_profile`): up to 1080p, H.264 High
    Level 4.1 with a 10 Mbps cap; larger frames (4K), Level 5.1 with a 40 Mbps cap.

Result (`TourPlanResult`):

```json
{"status": "completed",
 "plan": {"plan_version": "1.5", "listing_id": "L123", "shots": [...], "provenance": {"plan_hash": "..."}},
 "plan_hash": "...",
 "scoring": {"config_version": "0.3.0", "photos": 21,
             "rooms": [{"room_key": "living_room", "room_type": "living_room", "narrative_order": 3,
                        "hero_photo_id": "img_4", "photo_ids": ["img_4", "img_3"]}],
             "sequence": ["front_exterior", "entry", "living_room"], "warnings": []},
 "warnings": ["photo_fetch_failed:img_7:RuntimeError"], "seconds": 2.4}
```

- `plan` is checked with `contract.plan.parse_plan`, the validator `tour_render` runs; send it to
  `tour_render` unchanged. The same request always gives the same plan and `plan_hash`.
- A photo that cannot be fetched (`photo_fetch_failed:<photo_id>:<error>`) or decoded
  (`photo_unreadable:<photo_id>`) is left out with a warning; with none left the job fails with
  `image_fetch_failed`. The plan's own warnings are in `plan.warnings`.
- Photos are decoded with OpenCV: JPEG, PNG, WebP, TIFF and BMP. **HEIC / HEIF is not supported**
  and arrives as `photo_unreadable` (and `segment_failed:<photo_id>:ValueError` in `tour_segment`);
  see LICENSES.md for why.
- It needs no GPU: run it on the render image. Scoring takes about 0.1 s a photo.

## Layout

```
handler.py                RunPod entry (thin) -> tour_worker.jobs.router
tour_worker/
  config.py               Settings: read once from the environment (table below)
  errors.py               WorkerError(code, message), the failure shape
  contract/plan.py        plan schema, plan_hash, validators
  contract/jobs.py        request / result models of the three jobs
  jobs/router.py          dispatch by job_type, validate, run in a temp folder
  jobs/plan.py            tour_plan (run_plan)
  jobs/render.py          tour_render (run_tour)
  jobs/segment.py         tour_segment (run_segment)
  scoring/                photo scoring: coverage, framing, lighting, depth of field, motion
                          signals; rooms, heroes and narrative order (pipeline.score_photos)
  planner/                the shot planner (planner.build_shot_plan): allocation, the cinematic
                          camera engine, perceptual motion regions, beat sync, scene context
  service/jobs.py         HTTP service: job store + executor (states, records, TTL, webhook)
  service/http.py         HTTP service: RunPod's serverless API (FastAPI)
  service/redact.py       keeps signed URLs and keys out of the service's logs
  io/storage.py           S3 / HTTPS / file / local-folder storage, routed by URI scheme
  io/images.py            fetch + prepare photos and masks
  render/camera_path.py   keyframe_path / compound camera math, rotate projection
  render/timeline.py      frame-exact shot spans and fades
  render/frames.py        ShotWarp: every frame of a shot (camera + motion regions)
  render/direct.py        the renderer: frames -> ffmpeg in parallel chunks, audio
  render/encode.py        delivery encode settings, playback statistics
  render/probe.py         the playback checks a finished file must pass
  motion/                 perceptual motion: region.py (core, apply_regions), masks.py,
                          loader.py, one module per effect: fabric, foliage, canopy, water, fire
  vision/segment.py       Mask2Former: pinned checkpoint, Segmenter, scene summary
  vision/refine.py        water (above-waterline structure out) and flame-core refinements of its masks
  vision/fire_detect.py   lit fires inside fireplace masks
  music/                  beat analysis for beat sync (the `music` extra; offline, not a job);
                          structure.py: sections, phrases and hero moments (analysis v4)
tools/render.py           local CLI: plan + images folder -> MP4
tools/runpod/             GPU pod: setup.sh, preflight.py, start.sh, bench.py (3D timing), worker.env.example
tools/quality/            perceptual_quality.py (motion quality gate), sync_quality.py (beat sync score,
                          incl. hero metrics), sync_timeline.py (picture of cuts / punches vs the music)
tools/music/              analyze_library.py (re-analyse a library folder as v4), tap_heroes.html
                          (mark hero moments by ear), score_heroes.py (detector vs your taps),
                          review_cuts.html (mark late / early cuts in a render),
                          sync_regression.py (every library track x regression_tours.json, scored;
                          --compare sync_baseline.json fails when any case gets worse)
tools/samples/            fireplace / motion sample reels for review
tools/safe_run.py         run a heavy command pinned to some CPUs, frozen when the CPU gets hot
training/fireplace_adapter/   research: a fireplace head on COCO OneFormer's features (not enabled)
examples/                 one request per job type (plan 1.5)
docker/                   render.Dockerfile, segment.Dockerfile, compose.yaml (local dev)
tests/                    mirrors tour_worker/
```

## Configuration

`tour_worker.config.get_settings()` reads the environment once per process. No other code reads
`os.environ`.

| variable | default | meaning |
|---|---|---|
| `WORKER_STORAGE` | (unset) | `local`: dev mode for `tour_render` and `tour_plan`. Every image, mask and output URI that is not `file://` or an absolute path (s3://, https://, bare names) is served from `IMAGES_DIR` by basename. A missing image becomes a red placeholder. Unset: URIs are routed by scheme (`s3://` to S3, `https://` to HTTPS, `file://` or a path to the filesystem). `tour_segment` always uses S3 / HTTPS. |
| `IMAGES_DIR` | `/tmp` | the folder `WORKER_STORAGE=local` reads from and writes to |
| `PREWARP_WORKERS` | auto | processes that draw and encode frames. Auto is min(8, CPUs available - 1), capped by available memory at about 600 MB each. Output is identical for any count. |
| `MODELS_DIR` | `<repo>/models` | folder holding the model weights, one subfolder per model. Set it when the package is installed outside the repo. |
| `SEGMENT_MODEL` | `m2f_ade20k` | The `tour_segment` model: Mask2Former ADE20K panoptic, one pass for every motion class, fireplaces included. Water and fireplace masks are refined after it (`tour_worker/vision/refine.py`). |
| `SEGMENT_DIR` | `MODELS_DIR/mask2former_swin_large_ade_panoptic` | Folder of its weights. |
| `SPLAT_BACKEND` | `auto` | 3D tours (plan 1.6): `gsplat` (CUDA), `cpu` (the reference rasteriser: seconds a frame, for checking a scene without a GPU) or `auto` (gsplat when CUDA is available, else the render fails with `scene_render_failed`). |
| `SPLAT_SUPERSAMPLE` | `1.5` | 3D frames are rasterised at this x the output size and reduced (INTER_AREA): less shimmer. 1.0-3.0. |
| `SPLAT_SH_DEGREE` | `3` | View-dependent colour detail kept from the scene (0-3); lower saves memory. |
| `SCENE_CACHE_DIR` | `~/.cache/tour-worker/scenes` | Scene files by sha256: fetched and checked once, reused by later renders. |

The HTTP service (`python -m tour_worker.service`) adds these; `handler.py` ignores them:

| variable | default | meaning |
|---|---|---|
| `WORKER_API_KEY` | (unset) | the key callers send as `Authorization: Bearer <key>` on every endpoint but `/health`. Unset: the service refuses to start, unless `WORKER_ALLOW_NO_AUTH=1`. |
| `WORKER_ALLOW_NO_AUTH` | `0` | `1`: run without a key (local development only) |
| `JOB_CONCURRENCY` | `1` | jobs run at once. Keep 1 for renders: one render already uses every CPU it is given (`PREWARP_WORKERS`). |
| `JOBS_DIR` | (unset) | folder for job records (one JSON per job) so statuses survive a restart. Unset: memory only. The images set `/var/lib/tour-worker/jobs`. |
| `JOB_TTL_S` | `86400` | seconds a finished job's status is kept; then `/status` is 404 |
| `RUNSYNC_TIMEOUT_S` | `300` | seconds `POST /runsync` waits before it returns the job's current status |
| `HOST` | `127.0.0.1` | listen address (the images set `0.0.0.0`) |
| `PORT` | `8000` | listen port |

boto3 reads the AWS credentials and region itself (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`, or an instance role).

## Running as a service

The worker has two front doors over one job router (`tour_worker.jobs.router.run_job`) and one
wire protocol, RunPod's serverless API. The backend uses one client for both; only the base URL
and the key change:

| deployment | entry | base URL | key |
|---|---|---|---|
| RunPod serverless | `python -u handler.py` | `https://api.runpod.ai/v2/<endpoint_id>` | the RunPod API key |
| service container | `python -m tour_worker.service` | `http://<host>:8000` | `WORKER_API_KEY` |

```bash
WORKER_API_KEY=dev-key python -m tour_worker.service           # needs the http extra
```

One process per container: jobs and their statuses live in the process (`JOBS_DIR` keeps the
records). Scale with more containers; each one is its own base URL.

### Endpoints

Every endpoint but `/health` needs `Authorization: Bearer <WORKER_API_KEY>`. Times are in
milliseconds, ids are 32 hex characters.

`POST /run`: queue a job. The body is RunPod's: `input` is the job request (see
[Jobs and their JSON](#jobs-and-their-json)); `webhook` is optional; `policy` is accepted and
ignored.

```json
{"input": {"job_type": "tour_render", "plan": {...}, "output_prefix": "s3://bucket/tours/t1/"},
 "webhook": "https://backend.example.com/hooks/tour"}
```
```json
{"id": "5f0c2a9e4b7d4c35a2d1e0f9b8c7a6d5", "status": "IN_QUEUE"}
```

`POST /runsync`: the same body. Waits up to `RUNSYNC_TIMEOUT_S` and returns the `/status` body:
the result if the job finished, else `IN_QUEUE` / `IN_PROGRESS` and the id to poll.

`GET /status/{id}`:

```json
{"id": "5f0c...", "status": "COMPLETED", "delayTime": 12, "executionTime": 412300,
 "output": {"status": "completed", "video_uri": "s3://bucket/tours/t1/master.mp4", "duration_s": 71.6,
            "width": 1920, "height": 1080, "fps": 30, "codec": "h264", "render_seconds": 412.3,
            "plan_hash": "...", "warnings": []}}
```
```json
{"id": "5f0c...", "status": "FAILED", "delayTime": 12, "executionTime": 2100,
 "error": {"code": "image_fetch_failed", "message": "image_fetch_failed: ..."}}
```

- `status`: `IN_QUEUE`, `IN_PROGRESS`, `COMPLETED`, `FAILED` or `CANCELLED` (RunPod's names).
- `delayTime` once the job started (queue wait); `executionTime` once it finished.
- `output` only when `COMPLETED`: the job's result without its `error` key (as RunPod's SDK
  passes a handler's result on).
- `error` only when `FAILED`: `{"code", "message"}`, the codes in the table above, plus
  `worker_restarted` (the service restarted before the job finished: submit it again).

`POST /cancel/{id}`: `{"id", "status"}`. Only an `IN_QUEUE` job is cancelled (`CANCELLED`). A
running job is left running and the body says `IN_PROGRESS`: a render is many ffmpeg and worker
processes with an upload at the end, and there is no safe point to stop it. A finished job is
unchanged.

`GET /health` (no key):

```json
{"jobs": {"inQueue": 0, "inProgress": 1, "completed": 12, "failed": 1, "cancelled": 0},
 "workers": {"running": 1, "idle": 0, "concurrency": 1},
 "version": "0.3.0", "jobTypes": ["tour_render", "tour_segment", "tour_plan"],
 "segmentModel": "facebook/mask2former-swin-large-ade-panoptic@ae53ed7ab3", "workerId": "<hostname>"}
```

**Errors.** Every HTTP error has the worker's failure shape,
`{"status": "failed", "error": {"code", "message"}}`:

| HTTP | code | when |
|---|---|---|
| 400 | `invalid_job_type`, `invalid_request`, `invalid_plan`, `unsupported_camera_type`, `unsupported_easing` | the request cannot run: checked at submit (request shape, and the plan's schema, `plan_hash` and camera rules), nothing is queued. The codes are the ones the job would fail with. A body that is not JSON, has no `input`, or has a non-http(s) `webhook` is `invalid_request`. |
| 401 | `unauthorized` | missing or wrong key |
| 404 | `job_not_found` | unknown id, or a finished job older than `JOB_TTL_S` |

**Webhook.** When a job finishes (or is cancelled), the service POSTs its `/status` body to the
webhook: three tries, 10 s timeout each. It is best effort: a webhook that fails never fails the
job, so poll `/status` as well.

**Restarts.** With `JOBS_DIR` set, finished jobs survive a restart. A job that was queued or
running is `FAILED` with `worker_restarted`. Records never hold the request (it has presigned
URLs) or the webhook.

**Logs.** The service never logs a request body. Error messages and webhook targets are logged
without query strings, and HTTP client libraries are kept at WARNING (they log URLs).

### Docker

The Dockerfiles have not been built yet (they were written without a Docker daemon). Build them from the repo root:

```bash
docker build -f docker/render.Dockerfile  -t tour-worker-render .     # python 3.12-slim + ffmpeg, no torch
docker build -f docker/segment.Dockerfile -t tour-worker-segment .    # CUDA 12.6 torch, COCO weights baked in
docker build -f docker/segment.Dockerfile --build-arg BAKE_ADE20K=1 -t tour-worker-segment .   # + ADE20K
docker run -p 8000:8000 -e WORKER_API_KEY=... -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... \
    -e AWS_DEFAULT_REGION=... tour-worker-render
docker compose -f docker/compose.yaml up --build                      # local: render on :8000, key dev-key
docker compose -f docker/compose.yaml --profile segment up --build    # + segment (CPU torch) on :8001
```

Both images run as a non-root user, default to the HTTP service on port 8000 and have a
`HEALTHCHECK` on `/health`. The segment image sets `HF_HUB_OFFLINE=1`: the weights are in the
image, and a missing checkpoint fails at once instead of downloading.

### RunPod

**The FastAPI service on a RunPod GPU pod (how the backend uses it).** The backend's Celery task
calls the worker's HTTP API: `POST /run`, `GET /status/{id}`, `POST /cancel/{id}`, `GET /health`,
and `GET /ping`. No container image is involved: the worker is installed on the pod, on top of the
pod's own pinned GPU stack (torch, CUDA, gsplat), and runs as a plain process. `tools/runpod/`:

| file | what |
|---|---|
| `setup.sh` | ffmpeg if missing; a virtualenv (`.venv-pod`) that sees the pod's packages; the worker and its `http` extra installed under a constraints file frozen from the pod, so torch / gsplat are never replaced (pip stops instead). `WITH_SEGMENT=1` adds transformers etc. for `tour_segment` if the pod lacks them. Ends by checking torch / torchvision / gsplat / numpy are unchanged. |
| `preflight.py` | one report: ffmpeg + libx264; GPU, VRAM, CUDA, driver, torch; a gsplat render (its first call, which may build the CUDA kernels, and ms per 1080p frame); optional S3 write/read (`--s3-prefix`), scene HEAD (`--scene`), segmentation weights (`--segment`), and the GPU / 3D test suites (`--tests`). Exit 1 when anything failed. |
| `start.sh` | `start` / `stop` / `status` / `logs` of the service; settings from `~/.config/tour-worker/worker.env` (`worker.env.example` lists them; keep the real file out of the repo). Waits for `/ping`, prints `/health`. |
| `bench.py` | times a 3D tour through the real `tour_render` job: the sample apartment, a push-in and a 180-degree pan (`examples/tour_render_3d.json` is its request). |

```bash
git clone <worker repo> && cd tour-worker && git checkout <branch>
tools/runpod/setup.sh                                  # WITH_SEGMENT=1 for tour_segment as well
.venv-pod/bin/python -m tools.runpod.preflight --tests \
    --s3-prefix s3://<bucket>/virtual_tour/splat_demo/preflight/ \
    --scene s3://<bucket>/virtual_tour/splat_demo/scenes/green_model_house_3dgs.ply
mkdir -p ~/.config/tour-worker && cp tools/runpod/worker.env.example ~/.config/tour-worker/worker.env
$EDITOR ~/.config/tour-worker/worker.env               # WORKER_API_KEY, AWS keys, cache dirs
tools/runpod/start.sh                                  # then: status | logs | stop
```

`/health` reports `splat`: the 3D backend and the installed gsplat / torch versions.

**Timing a 3D tour on the pod** (the AWS keys exported, or the env file sourced):

```bash
.venv-pod/bin/python -m tools.runpod.bench \
    --scene s3://<bucket>/virtual_tour/splat_demo/scenes/green_model_house_3dgs.ply \
    --sha256 68dfddac0b602ed1af698dad0411de43551c6dd48cc788af908b90a2bfa1d028 \
    --output-prefix s3://<bucket>/virtual_tour/splat_demo/renders/<run>/   # --width 3840 --height 2160 for 4K
```

The first run downloads the scene to `SCENE_CACHE_DIR`; a second run shows the warm-cache time.

- **Backend settings:**
  - `TOUR_WORKER_TRANSPORT="http"`: the backend calls `TOUR_RENDER_WORKER_URL` /
    `TOUR_SEGMENT_WORKER_URL` (e.g. `http://<host>:8000`) with `TOUR_*_WORKER_KEY`.
  - `TOUR_WORKER_TRANSPORT="runpod"`: it calls `RUNPOD_ENDPOINT_URL` + `RUNPOD_TOUR_RENDER_PATH` /
    `RUNPOD_TOUR_SEGMENT_PATH` with `RUNPOD_API_KEY`. The service answers under those prefixes
    (`SERVICE_PREFIXES`).
- **One process per host.** Jobs and their statuses live in the service process; `JOBS_DIR`
  keeps the records across restarts. Behind a proxy with several hosts, every `/status` for a
  job must reach the host that took it. Either use one host, or share `JOBS_DIR` between hosts.
- **Models:** the Mask2Former weights are fetched to `MODELS_DIR` on first use. Pre-fetch them once on
  the host so a timed run does not include the download.
- **AWS:** the credentials must allow reading the photos and writing the outputs.
- **Timing a run:** every job result carries `timings` (stages, total, host). For a cold-cache run,
  restart the service and use a listing the mask cache has never seen, then run
  `python -m tools.timing_report` on the tour job's `extra["timings"]`.

**Serverless (queue) endpoint.** Deploy the same image as a serverless endpoint with the container start command
`python -u handler.py` (RunPod runs the handler; `WORKER_API_KEY` and the service settings are
not used). The segment image needs a GPU worker; the render image runs on CPU. The request and
result JSON are identical; two differences the client must allow for:

- A request that cannot run is not refused at submit: RunPod queues it and the job ends `FAILED`
  with the same code.
- `error` may arrive as a JSON string rather than an object, and a handler crash is RunPod's own
  error JSON (`error_type`, `error_message`, ...). Parse a string as JSON; without a `code`, treat
  it as `unknown`.

### Contract for callers

1. **Submit** `POST {base}/run` with `{"input": <job request>}`. Keep the returned `id`.
   A 400 means the request is wrong: do not retry it.
2. **Poll** `GET {base}/status/{id}` until the status is terminal: `COMPLETED`, `FAILED` or
   `CANCELLED`. Suggested interval: 2 s for `tour_segment` (about 13 s a photo on CPU, much less
   on GPU), 5 to 10 s for `tour_render` (minutes), with backoff. A webhook can replace most polls,
   but keep a slow poll in case it is lost.
3. **Read** `output` on `COMPLETED`: `video_uri` for a render, `results` / `scenes` for a
   segmentation, `plan` / `plan_hash` for a plan. Show `warnings`; they never fail a job.
   On `FAILED` read `error.code`. Retry only transient codes (`image_fetch_failed`,
   `encode_failed`, `worker_restarted`, `unknown`); the plan codes and `invalid_request` will fail
   again, and so will `planning_failed`.
4. **Time out** on the caller's side: give up (and `POST /cancel/{id}`, which only helps while it
   is queued) after a deadline that fits the job, e.g. 30 min for a render. A 404 on `/status`
   means the job is gone (TTL, or a service without `JOBS_DIR` restarted): resubmit.
5. **Idempotency.** `/run` is not idempotent: each call is a new job. Retrying is still safe: a
   plan always renders to the same bytes at `<output_prefix>master.mp4` (masks likewise go to
   the same keys), so a repeat overwrites the same objects. Use a fresh `output_prefix` per job if
   two runs must not overwrite each other. `input.job_id` is passed through for the caller's logs.

**Inputs.** The worker reads `s3://` URIs with its own AWS credentials (boto3) and `https://`
URLs (presigned is fine; 15 s timeout, 50 MB limit, no redirect to http). It writes outputs only
to `s3://` (`output_prefix` must be `s3://...` in production; `https://` cannot be written). For
`tour_render`, a plan's URIs stay as tour_plan hashed them; `image_urls` maps a shot's
`image_uri` or a region's `mask_uri` to a URL to fetch instead (for example a presigned https
URL when the worker has no read access to the bucket). `music.uri` is mapped the same way (fetched
as written when `image_urls` has no entry for it); a failed fetch renders without music
(`music_fetch_failed`). `tour_segment` reads `images[].uri` (s3:// or https://) and writes masks
under its `output_prefix`. `tour_plan` reads `photos[].uri` (or its `image_urls` entry) and writes
nothing.

## Run locally

Render a plan from a folder of photos. Images are found by the basename of each `image_uri`, and
masks as `<images-dir>/masks/<shot_id>_<region_id>.png`, the way a tour workdir has them:

```bash
python -m tools.render --plan plan.json --images-dir ./images --out ./tour.mp4 [--music-file track.mp3] [--workers 3]
```

This runs the production `tour_render` path: plan checks, image preparation, render and playback
checks. It prints the job result.

To run the RunPod handler locally (dev storage), put every photo and mask the request names in
`IMAGES_DIR` by basename (a missing one becomes a red 1920x1080 placeholder). The example requests
show the shape of a request; their photos are not in the repo:

```bash
WORKER_STORAGE=local IMAGES_DIR=./images python handler.py --test_input "$(cat my_request.json)"
```

On a machine that overheats, wrap heavy commands:
`python tools/safe_run.py --cpus 8,10,12,14 --memory 6G -- <cmd>`.

## Segmentation models

`tour_segment` needs the `segment` (CPU) or `segment-gpu` extra and the pinned Mask2Former
checkpoint:

| `SEGMENT_MODEL` | Hugging Face model | revision | folder |
|---|---|---|---|
| `m2f_ade20k` | `facebook/mask2former-swin-large-ade-panoptic` | `ae53ed7ab37f` | `mask2former_swin_large_ade_panoptic` |

- **Download:** if the weights are missing, the first job downloads them from the pinned revision
  into `SEGMENT_DIR`. Every load checks `model.safetensors` against its pinned sha256.
- **Pre-fetching:** fetch the folder once on the host so no job downloads it.
- **Speed:** about 2.2 s per photo on CPU (the 2026-11-11 benchmark); the model loads once per
  warm worker.
- **Refinements** (`tour_worker/vision/refine.py`):
  - A pool or water mask keeps the model's mask, so reflections, submerged shelves and steps,
    dark water and sunlit water keep moving. Only what stands above the waterline stays still:
    dark walls with neutral, clearly brighter stone directly above them (a spa wall under its
    coping), thin dark rims along the edge, grey or pale coping that is not water-hued, and a thin
    non-water-coloured lip. A reflection mirrors what stands above it (dark trees over a dark tree
    reflection), so it fails the wall test; a dark part bigger than a third of the water is dark
    water. Thresholds are relative to the pool's own water (the brighter half of its pixels).
    Checked on 104 varied photos (pools, spas, indoor, night, lakes, ponds, sea).
  - Known limits: water the model does not label at all (a dark, mirror-like raised spa) cannot be
    recovered here; mid-tone tile bands on a spa's outer face can ripple; the model's own false
    positives (a railing, a tub shell, steam) pass through.
  - A fireplace becomes a `fire` region holding only its flame core, and only when it is lit;
    the mantel and surround stay still.

## 3D tours (plan 1.6)
A tour can be rendered from a Gaussian splat scene instead of photos: the backend passes the scene
file's link (`plan.scene`: `uri`, `sha256`, `format` `ply` | `splat`, optional `up_axis`), and
every shot is a `bezier_path` camera flying through it.

- **Scene files** (`tour_worker/scene/splat_io.py`): `.ply` as 3DGS training writes it
  (spherical-harmonic colour to degree 3, logit opacity, log scale, w x y z rotation) and the
  web-viewer `.splat`. Loaded in chunks and validated; a few broken splats are dropped, more than
  1 % is `scene_invalid`. Cached on disk by sha256 (`scene/cache.py`).
- **Orientation** (`scene/orient.py`): with no `scene.json`, up, floor and ceiling come from the
  geometry (floor and ceiling planes; furniture mass for the sign, the file's axis convention when
  that is ambiguous). An explicit `scene.up_axis` wins.
- **Cameras** (`scene/camera.py`): OpenCV axes, intrinsics from `hfov_deg`, cubic Beziers
  reparameterised by arc length so the camera moves evenly before easing and the beat-sync
  `time_warp`.
- **Rendering** (`render/splat_shot.py`): `SplatShot` has ShotWarp's frame interface, so 3D shots
  go through the same composition (fades, whip, push, overlays), chunk encoders, audio and join.
  3D plans draw in one process (one GPU). gsplat on CUDA in production; `render/splat_cpu.py` is
  a CPU reference rasteriser (same maths) for checking scenes without a GPU and as the reference
  the GPU path is compared against (PSNR >= 45 dB; tests skip without CUDA).
- `scene.json` (rooms and camera anchors marked by an operator) is optional; when given it is
  validated strictly (`scene/scene_json.py`).

## Tests

```bash
uv run pytest                 # or: python -m pytest   (needs ffmpeg / ffprobe)
```

`tests/` mirrors the package. The render tests encode real videos, so a full run takes a minute
or two (use `tools/safe_run.py` on a hot machine).

## Licences

See `LICENSES.md`: the Mask2Former weights and their training data (ADE20K), and librosa.
