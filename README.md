# Articulait Cinematic Tour Worker

## Run
Run the handler locally using SDK test mode:
`WORKER_STORAGE=local IMAGES_DIR=/tmp python3 handler.py --test_input test_input.json`

Or using the CLI:
`python3 run_tour.py --plan plan.json --images-dir ./images --out ./out`

## Test
Run Python tests with pytest:
`uv run pytest tests/`

Run TypeScript tests:
`cd remotion && npm test`

## Benchmark
Run the benchmark script:
`uv run python bench.py`
`uv run python sweep.py`
## Environment Variables
- `WORKER_STORAGE`: Set to `local` to use local storage mock.
- `IMAGES_DIR`: Directory for local images when `WORKER_STORAGE=local`.
- `AWS_ACCESS_KEY_ID`: AWS access key.
- `AWS_SECRET_ACCESS_KEY`: AWS secret key.
- `AWS_DEFAULT_REGION`: AWS region.
- `TOUR_PARALLAX`: Set to `1` to enable experimental depth parallax (off by default: the per-pixel warp bends straight lines and rigid objects next to near objects).
- `DEPTH_MODEL_PATH`: Depth model location (default `models/depth_anything_v2_small.onnx`; downloaded from a pinned revision and checksum-verified if missing — bake it into the image).
- `REMOTION_GL`: GL backend for parallax shots (default `swangle`, software). Use `angle-egl` on a GPU worker.

## Transitions & Plan Contracts
The worker supports `plan_version` `1.0`, `1.1`, and `1.2`.
- In `1.0`: Only cut transitions and default `"ease_in_out"` easing are supported.
- In `1.1`: The `transition_in` block on any shot after the first can specify `"type": "fade"`. Fade duration (`duration_s`) cannot exceed 50% of the shot's individual `duration_s`. Only default `"ease_in_out"` easing is allowed.
- In `1.2`: Supports camera easing via `camera.easing` (`"linear"`, `"ease_in"`, `"ease_out"`, `"ease_in_out"`). Also supports optional `camera.preset` metadata string. Unknown easing values reject with `unsupported_easing`.

## Camera Motion Presets
Reference implementation in `tools/motion_presets.py`, a mirror of the backend planner's
`app/domains/virtual_tour/planner/presets.py` (the rationale for the values lives there; keep
the two in sync). All presets are `linear`; `roll_deg` is the gimbal roll reached by the end
of the shot, in the direction of travel.

| Preset | zoom | shift_x | shift_y | roll_deg | max_zoom |
|---|---|---|---|---|---|
| `push_in_gentle` | +0.12 | -0.09 | -0.03 | 0.5 | |
| `push_in_strong` | +0.20 | -0.07 | +0.05 | 0.8 | |
| `push_slide` | +0.10 | -0.18 | -0.02 | 1.0 | 1.35 |
| `slide` | -0.08 | -0.20 | 0.00 | 1.2 | 1.35 |
| `hero_sweep` | +0.16 | -0.18 | -0.04 | 1.5 | 1.35 |
| `tilt` | +0.06 | 0.00 | +0.14 | 0.4 | |
| `pull_back` | -0.13 | -0.10 | -0.04 | 0.6 | |

Roll is rendered inside the planner's validated window (`rollFitScale` in `remotion/src/math.ts`
scales the rolled frame up to fit), so it never exposes anything outside the photo.

## Depth Parallax
Experimental, off by default. When `TOUR_PARALLAX=1`, `modules/depth.py` runs Depth Anything V2 Small on each shot image
and writes `<image>.depth.png` plus a focus depth. Those shots render through
`remotion/src/ParallaxShot.tsx`: the same window and roll as `KenBurnsShot`, plus a per-pixel
shift by depth around the focus depth (a full-screen shader, no mesh). A shot whose depth fails
renders as Ken Burns, with a `depth_unavailable:` warning. Only the Small checkpoint may be used:
Base/Large/Giant are non-commercial.

Fallback logic:
Tries original preset with shifts scaled by `1.0`, `0.75`, `0.5`. If no valid camera window fits in `[0,1]^2`, downgrades to `push_in_gentle`. If still invalid, falls back to a static camera (`zoom=1.0`, `center=(0.5, 0.5)`).

## Motion Analyzer (`tools/motion_analyzer.py`)
CLI syntax:
```bash
python -m tools.motion_analyzer clip.mp4 [--shot-frames N | --cuts auto] [--json out.json] [--heatmaps dir]
```
Performs Farneback optical flow on 332px grayscale frames, masked borders, least-squares affine fit, per-step displacements and zoom steps, 3-band zoom, cumulative speed profiles, and auto-cut detection.

> **Caveat on Optical Flow**: Optical flow is unreliable on flat regions and heavily compressed video. Calibrate on textured synthetic images and treat real-photo numbers as approximate.

## Motion Lab (`tools/motion_lab.py`)
One-shot preview tool and analyzer comparison CLI:
```bash
python -m tools.motion_lab --image photo.jpg --preset push_in_strong [--scale 0.25] [--out out.mp4]
python -m tools.motion_lab --image-dir ./photos --all-presets [--scale 0.25]
```
Renders a 5-second cinematic preview using `preset_to_camera`, analyzes the output with the motion analyzer, produces a 3-frame contact sheet JPG, performs black-edge validation (outermost 4 px > 10 luminance), and prints comparison tables against tolerances (zoom 0.03, shift 0.015, profile 0.07).
