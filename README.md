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
  `<output_prefix>master.mp4`. The same plan always gives a byte-identical file.
- **`tour_segment`**: runs OneFormer on a tour's photos before planning. It uploads one PNG mask
  per object that can move (curtain, blinds, potted plant, tree, water, lit fire) and returns
  JSON only: the regions and a scene summary per photo.

## Install

System tools: **`ffmpeg` and `ffprobe`** on `PATH` (the renderer pipes frames to ffmpeg, and the
playback checks use ffprobe). Python 3.10 or later.

```bash
uv sync                          # the render job, plus the dev group (pytest)
uv sync --extra segment          # + OneFormer on CPU PyTorch (dev machines)
uv sync --extra segment-gpu      # + OneFormer on CUDA 12.6 PyTorch (RunPod GPU worker)
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
- `image_urls` is optional. It maps a URI in the plan (a shot's `image_uri`, a region's
  `mask_uri`, or `music.uri`) to the URL the worker should fetch instead.
- Plan versions 1.0 to 1.5 are accepted:
  - 1.0: ken_burns cameras and cuts only.
  - 1.1: adds fades.
  - 1.2: adds easing.
  - 1.3: adds `keyframe_path` and `compound` cameras, the `rotate` projection and roll.
  - 1.4: adds `motion_regions`, with the effects `fabric_sway`, `foliage_sway`, `canopy_flutter`, `calm_water`, `water_ripple` and `fire_flicker`.
  - 1.5: adds the `music` block, `time_warp` and `punches`.

  `provenance.plan_hash` must be the sha256 of the plan's canonical JSON with the hash itself
  removed (`compute_plan_hash`).

Result:

```json
{"status": "completed", "video_uri": "s3://bucket/virtual_tour/<job>/master.mp4",
 "duration_s": 71.6, "width": 1920, "height": 1080, "fps": 30, "codec": "h264",
 "render_seconds": 412.3, "plan_hash": "...", "warnings": ["motion_region_dropped:s17:seg_002:faint_0.3px"],
 "error": null}
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
 "warnings": [], "model": "shi-labs/oneformer_coco_swin_large@3a263017ca", "seconds": 13.2}
```

`bbox` and the masks are in `space` pixels: the photo scaled to at most 2048 px on its long side.
The scene summary lists every OneFormer class in the photo; the planner uses it to choose camera
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
- `options` (optional, no other keys): `max_total_s` is the tour's ceiling; `budget_blocked`
  makes a slideshow. Everything else is the planner's default (`tour_worker/planner/config.py`).

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
  vision/segment.py       OneFormer: pinned checkpoints, Segmenter, scene summary
  vision/fire_detect.py   lit fires inside fireplace masks
  music/                  beat analysis for beat sync (the `music` extra; offline, not a job)
tools/render.py           local CLI: plan + images folder -> MP4
tools/quality/            perceptual_quality.py (motion quality gate), sync_quality.py (beat sync score)
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
| `SEGMENT_MODEL` | `coco` | OneFormer checkpoint for `tour_segment`: `coco` (primary) or `ade20k` (knows fireplaces) |
| `ONEFORMER_DIR` | `MODELS_DIR/<checkpoint dir>` | folder of that checkpoint: `oneformer_coco_swin_large` or `oneformer_ade20k_swin_large` |
| `COCO_FIRE_SEARCH` | `0` | `1`: look for lit fires by colour with the COCO model. Off by default because it failed validation (it found 4 of 11 fires). |
| `FIREPLACE_MODEL` | `ade20k` | A second OneFormer pass in `tour_segment` that finds fireplaces (COCO has no fireplace class); lit fires are detected inside them and added as `fire` regions. Empty: off. About 7 s per photo on CPU. |
| `FIREPLACE_DIR` | (unset) | That checkpoint's folder; default `MODELS_DIR/<its dir name>`. |

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
 "segmentModel": "shi-labs/oneformer_coco_swin_large@3a263017ca", "workerId": "<hostname>"}
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

Deploy the same image as a serverless endpoint with the container start command
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

`tour_segment` needs the `segment` (CPU) or `segment-gpu` extra and a pinned OneFormer checkpoint
(MIT weights):

| `SEGMENT_MODEL` | Hugging Face model | revision | folder |
|---|---|---|---|
| `coco` | `shi-labs/oneformer_coco_swin_large` | `3a263017ca5c` | `oneformer_coco_swin_large` |
| `ade20k` | `shi-labs/oneformer_ade20k_swin_large` | `4a5bac8e64f8` | `oneformer_ade20k_swin_large` |

If the weights are missing, the first job downloads them from the pinned revision into
`ONEFORMER_DIR`. Every load checks `pytorch_model.bin` against its pinned sha256. Bake the folder
into the worker image so no job downloads. One photo takes about 13 s on 4 CPU threads; the model
loads once per warm worker.

## Tests

```bash
uv run pytest                 # or: python -m pytest   (needs ffmpeg / ffprobe)
```

`tests/` mirrors the package. The render tests encode real videos, so a full run takes a minute
or two (use `tools/safe_run.py` on a hot machine).

## Licences

See `LICENSES.md`: the OneFormer weights and their training data (COCO, ADE20K), and librosa.
