# tour-worker

The Articulait cinematic tour worker. It runs two jobs for the backend, on RunPod (serverless)
or locally:

- **`tour_render`**: renders a tour **plan** (made by the backend planner) into an H.264 MP4.
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
| `invalid_job_type` | `job_type` is not `tour_render` or `tour_segment` |
| `invalid_request` | the request is missing required fields (for example `output_prefix`) |
| `invalid_plan`, `unsupported_camera_type`, `unsupported_easing` | the plan failed validation (`tour_worker/contract/plan.py`) |
| `image_fetch_failed` | a shot's photo could not be fetched |
| `prewarp` | a photo cannot be read, or its aspect ratio does not match the plan |
| `encode_failed`, `verification_failed` | ffmpeg failed, or the file failed the playback checks (codec, size, fps, duration, faststart, keyframe gap, VBV, level) |
| `unknown` | anything unexpected |

### `tour_render`

Request (full example: `examples/tour_render.json`):

```json
{
  "job_type": "tour_render",
  "job_id": "optional",
  "plan": { "plan_version": "1.5", "...": "the plan, exactly as the backend hashed it" },
  "output_prefix": "s3://bucket/virtual_tour/<job>/",
  "image_urls": { "<shot image_uri or mask_uri>": "<URL to fetch it from, e.g. presigned https>" }
}
```

- `output_prefix` is **required**. The video is written to `<output_prefix>master.mp4`.
- `image_urls` is optional. It maps a URI in the plan to the URL the worker should fetch instead.
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

## Layout

```
handler.py                RunPod entry (thin) -> tour_worker.jobs.router
tour_worker/
  config.py               Settings: read once from the environment (table below)
  errors.py               WorkerError(code, message), the failure shape
  contract/plan.py        plan schema, plan_hash, validators
  contract/jobs.py        request / result models of both jobs
  jobs/router.py          dispatch by job_type, validate, run in a temp folder
  jobs/render.py          tour_render (run_tour)
  jobs/segment.py         tour_segment (run_segment)
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
tests/                    mirrors tour_worker/
```

## Configuration

`tour_worker.config.get_settings()` reads the environment once per process. No other code reads
`os.environ`.

| variable | default | meaning |
|---|---|---|
| `WORKER_STORAGE` | (unset) | `local`: dev mode for `tour_render`. Every image, mask and output URI that is not `file://` or an absolute path (s3://, https://, bare names) is served from `IMAGES_DIR` by basename. A missing image becomes a red placeholder. Unset: URIs are routed by scheme (`s3://` to S3, `https://` to HTTPS, `file://` or a path to the filesystem). `tour_segment` always uses S3 / HTTPS. |
| `IMAGES_DIR` | `/tmp` | the folder `WORKER_STORAGE=local` reads from and writes to |
| `PREWARP_WORKERS` | auto | processes that draw and encode frames. Auto is min(8, CPUs available - 1), capped by available memory at about 600 MB each. Output is identical for any count. |
| `MODELS_DIR` | `<repo>/models` | folder holding the model weights, one subfolder per model. Set it when the package is installed outside the repo. |
| `SEGMENT_MODEL` | `coco` | OneFormer checkpoint for `tour_segment`: `coco` (primary) or `ade20k` (knows fireplaces) |
| `ONEFORMER_DIR` | `MODELS_DIR/<checkpoint dir>` | folder of that checkpoint: `oneformer_coco_swin_large` or `oneformer_ade20k_swin_large` |
| `COCO_FIRE_SEARCH` | `0` | `1`: look for lit fires by colour with the COCO model. Off by default because it failed validation (it found 4 of 11 fires). |

boto3 reads the AWS credentials and region itself (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`, or an instance role).

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
