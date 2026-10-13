# Tour v2: recovered baseline, hard cuts, multi-step camera moves

Branch `feat/tour-v2`, created from `park/m3-and-retune-2026-10-12` (`1b64a33`) on 2026-10-13.
`master`, `park/*` and `exp/*` are not modified, and nothing was merged or deleted. All renders
are 1920×1080, 30 fps, H.264 yuv420p, with Remotion's silent AAC track. The renders are not committed.

## Summary

- **Baseline recovered to mean SSIM 0.9763.** The target was ≥ 0.98, so this is the closest match,
  not a full recovery.
  - No plan file on disk reproduces the liked video, so the cameras were measured from it frame
    by frame.
  - Frame count, duration, shot order, photos and all 12 shot boundaries match.
  - The remaining difference is renderer noise, not camera error (§1.5).
- **Determinism:** two v2 renders are identical, 2100/2100 frames.
  - Passing `--gl angle` or `--gl swangle` made two renders of the same plan differ in 839 and
    690 frames.
  - Renders with `--gl` unset are identical. All final renders leave it unset.
- **v2 constraints:**
  - The window stays ≥ 1 output px inside the image on every frame (minimum 1.999 px).
  - Pixel density stays ≥ 0.75 on every frame (minimum 0.7812). No path needed shrinking and no
    zoom was capped.
- **`reveal_sweep` stops at K1 and K2.** With the monotone (Fritsch–Carlson) interpolation the
  brief specifies, the camera comes to a near-stop at both keyframes: 0.17 px/frame at K1 and
  0.05 px/frame at K2. The move as defined reverses cx at K1 and K2 and holds zoom between them,
  and a monotone cubic gives zero velocity there by construction. The brief's expectation of "no
  stop and restart mid-shot" cannot hold for this move (§3.3).
- **Direction rule conflict:** in 4 of 6 `push_to_subject` shots the chosen direction could not be
  honoured, because the thirds framing fixes where the window ends (§4.3).

## 1. Step 1: recovering the baseline

### 1.1 Reference

`out/reference/liked_baseline.mp4` was missing.
- Six different files in `~/Downloads` matched the description: 70.059 s, 2100 frames, AAC.
- I asked which one was the liked baseline. The answer was `~/Downloads/virtual_tour_1080p.mp4`.
- It was copied with `cp -p`: md5 `20bbd6cb09d551e4d385df597100ec2e`, 133,701,351 bytes,
  container comment "Made with Remotion 4.0.141".
- Its byte content differs from every file in `old_generations/`. Per-second SSIM against those
  files is only 0.41–0.49: `virtual_tour_1080p.mp4` 0.4473, `(1)` 0.4528, `_v1_2` 0.4331,
  `_prev_v1_1` 0.4067, `parallax_preview` 0.4926.

### 1.2 Leads checked (no plan reproduces the liked video)

The liked video's 13 shots are, in order: exterior, four living-area views, dining nook, blue
dining room, den, canopy bedroom, bathroom, blue bedroom, dark bedroom and white bedroom.

**Photos.** Registering frames to the 19 Downing photos (ORB + RANSAC) shows these photos per
shot: `imgi_2286, 2416, 2478, 2366, 2431, 2326, 2638, 2446, 2590, 2346, 2510, 2542, 2526`. That is
the order used by both "restored" plans.

**Plan candidates:**

| Candidate | Transitions | Cameras vs liked |
|---|---|---|
| `Downloads/…/downing_tour_plan_restored.json` (planner 1.4.0, 2026-10-13 12:09) | all cuts | different values; has roll |
| `old_generations/downing_tour_plan_restored.json` (planner 1.1.0) | 0.5 s fades | different values and easings |
| `old_generations/downing_tour_plan_v1_2.json` (planner 1.0.0) | 0.5 s fades | different; s07 uses `imgi_2606` (not on disk) |
| backend scratch `downing_plan.json`, `downing_full_plan.json`, `downing_full_base.json` | all cuts | same as the 1.4.0 plan |
| Trash `plan.json` (20 shots, 80 s) | cuts | different tour |

**Render logs:** none found for this render. Only npm debug logs from 2026-10-13 exist.

**Liked camera, measured.** Fitting the registrations from 13 samples per shot shows three
properties:
- cx and cy are linear in time (residual ≈ 1e-4).
- ln(zoom) is linear in time, so zoom is interpolated in log space.
- Rotation is ±0.04°, i.e. no roll.

**Renderer state that produced it:**
- `master` (`e3025be`) forces smoothstep easing with linear zoom, so it cannot produce this.
- `park/m3-and-retune` (`1b64a33`) has linear easing, log-space zoom and fades, so it is the
  renderer state.

**Repo state found during this run.** `HEAD` had been switched to the park branch at 12:08, and
the B2 work was stashed as `stash@{0}: b2_experiment_wip` at 12:07. I did neither, and left both
untouched.

### 1.3 Recovery method (`tools/tour_v2/recover_baseline.py`)

- **Skeleton:** shot list, durations (5.5 / 5.2 s, total 70.0 s) and 0.5 s fades from
  `old_generations/downing_tour_plan_restored.json`. These match the liked video's 2100 frames and
  its boundaries.
- **Sampling:** every 6th frame outside the dissolves, 23–26 samples per shot, registered with
  ORB + RANSAC.
- **Fit:** cx and cy linear in the renderer's progress t over each shot's render span (which
  includes the fade overlap, as in `modules/timeline.py`). ln(zoom) linear in t.
- **Edge clamp:** each end is clamped into the image. The largest clamp shift is 1.1e-4 (s04
  end, about 0.2 px).

**Evidence the fit is right.** The recovered values sit on the park preset table:
- start zooms 1.035 / 1.0 / 1.13 / 1.185 / 1.255, on the solver's 0.005 grid
- zoom ratios ×1.12, ×1.20, ×0.92, ×1.10, ×1.06 and ×0.87, which equal the `zoom_total` of
  `push_in_gentle`, `push_in_strong`, `slide`, `push_slide`, `tilt` and `pull_back`
  (`tools/motion_presets.py`)
- speed 1 and roll 0

Re-solving each shot with `_solve_camera` reproduces the measured values to ≤ 2.2e-4.

**Variants tried** (SSIM on liked frames 60–89; exact settings in §1.5):

| Plan variant | Settings | SSIM |
|---|---|---|
| ORB fit (committed) | default JPEG, default GL | 0.9648 |
| ECC-refined fit (`--ecc`) | default JPEG, default GL | 0.9571 |
| Solver-snapped | default JPEG, default GL | 0.9599 |

The ORB fit is kept because it scores highest.

### 1.4 Re-render of the recovered plan and comparison

**v0 vs liked (ffprobe):**

| | liked_baseline.mp4 | v0_baseline.mp4 |
|---|---|---|
| frames | 2100 | 2100 |
| duration | 70.059 s | 70.059 s |
| resolution, fps | 1920×1080, 30/1 | 1920×1080, 30/1 |
| codec, pix_fmt | h264, yuv420p | h264, yuv420p |
| audio | aac 48 kHz 2 ch, −91.0 dB (silent) | aac 48 kHz 2 ch, −91.0 dB (silent) |
| x264 settings | crf 18, preset options identical | same |

**Shot boundaries.** All 12 nominal boundaries (frames 165, 330, 486, 642, 807, 972, 1128, 1293,
1458, 1623, 1788 and 1944) show the expected photo 9 frames before and after, in both videos.
The table is at the end of §5.

**SSIM**, grayscale, Gaussian 11×11 σ 1.5, middle frame of each second:

| second | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–9 | 0.9674 | 0.9687 | 0.9689 | 0.9697 | 0.9697 | 0.9670 | 0.9789 | 0.9793 | 0.9812 | 0.9800 |
| 10–19 | 0.9793 | 0.9736 | 0.9746 | 0.9750 | 0.9757 | 0.9744 | 0.9794 | 0.9795 | 0.9791 | 0.9792 |
| 20–29 | 0.9785 | 0.9719 | 0.9764 | 0.9778 | 0.9783 | 0.9796 | 0.9785 | 0.9787 | 0.9789 | 0.9787 |
| 30–39 | 0.9789 | 0.9781 | 0.9730 | 0.9754 | 0.9768 | 0.9772 | 0.9763 | 0.9691 | 0.9810 | 0.9810 |
| 40–49 | 0.9813 | 0.9811 | 0.9800 | 0.9767 | 0.9781 | 0.9795 | 0.9803 | 0.9804 | 0.9778 | 0.9765 |
| 50–59 | 0.9777 | 0.9758 | 0.9774 | 0.9747 | 0.9758 | 0.9753 | 0.9779 | 0.9792 | 0.9769 | 0.9670 |
| 60–69 | 0.9742 | 0.9764 | 0.9765 | 0.9778 | 0.9721 | 0.9760 | 0.9748 | 0.9740 | 0.9728 | 0.9707 |

Mean **0.9763**, min 0.967 (70 samples, frame 15 + 30·k).

#### Recovered baseline cameras (fixtures/downing/baseline_plan.json)

| shot | photo | room | dur s | fade in | start (cx, cy, zoom) | end (cx, cy, zoom) | zoom ratio | samples | max fit residual (cx/cy/ln z) | rot ° range |
|---|---|---|---|---|---|---|---|---|---|---|
| s01 | imgi_2286 | exterior | 5.5 | cut 0.0 | 0.4831, 0.4225, 1.0350 | 0.5654, 0.4457, 1.1592 | ×1.1200 | 26 | 5.9e-05 / 0.000105 / 0.000171 | -0.0101..0.0082 |
| s02 | imgi_2416 | living_room | 5.5 | fade 0.5 | 0.4006, 0.5350, 1.2549 | 0.5669, 0.5350, 1.1546 | ×0.9201 | 25 | 5.8e-05 / 0.000126 / 0.000223 | -0.0152..0.019 |
| s03 | imgi_2478 | living_room | 5.2 | fade 0.5 | 0.4831, 0.5566, 1.0350 | 0.5654, 0.5799, 1.1592 | ×1.1200 | 23 | 6.1e-05 / 0.000148 / 0.000185 | -0.0203..0.0293 |
| s04 | imgi_2366 | living_room | 5.2 | fade 0.5 | 0.5543, 0.5357, 1.1300 | 0.4023, 0.5500, 1.2430 | ×1.1000 | 23 | 9e-05 / 0.000113 / 0.00038 | -0.0256..0.0391 |
| s05 | imgi_2431 | kitchen | 5.5 | fade 0.5 | 0.5000, 0.5608, 0.9999 | 0.4358, 0.5221, 1.2001 | ×1.2002 | 25 | 4.6e-05 / 0.000126 / 0.000254 | -0.0172..0.018 |
| s06 | imgi_2326 | dining_room | 5.5 | fade 0.5 | 0.4006, 0.5415, 1.2552 | 0.5669, 0.5415, 1.1545 | ×0.9198 | 25 | 6.1e-05 / 7.1e-05 / 0.000548 | -0.0145..0.0183 |
| s07 | imgi_2638 | dining_room | 5.2 | fade 0.5 | 0.4830, 0.4904, 1.0351 | 0.5654, 0.5136, 1.1589 | ×1.1196 | 23 | 3.8e-05 / 6.8e-05 / 0.000301 | -0.0273..0.0238 |
| s08 | imgi_2446 | basement | 5.5 | fade 0.5 | 0.5575, 0.5627, 1.1300 | 0.4054, 0.5769, 1.2431 | ×1.1002 | 25 | 0.000103 / 0.000178 / 0.000544 | -0.0166..0.0228 |
| s09 | imgi_2590 | primary_bedroom | 5.5 | fade 0.5 | 0.5000, 0.4772, 1.0000 | 0.4358, 0.4387, 1.2000 | ×1.2000 | 25 | 8.2e-05 / 0.000112 / 0.000578 | -0.0091..0.0242 |
| s10 | imgi_2346 | primary_bathroom | 5.5 | fade 0.5 | 0.5000, 0.5781, 0.9999 | 0.4999, 0.4633, 1.0601 | ×1.0602 | 25 | 6.4e-05 / 7.4e-05 / 0.000298 | -0.0101..0.0242 |
| s11 | imgi_2510 | bedroom | 5.5 | fade 0.5 | 0.5000, 0.5781, 1.0000 | 0.5641, 0.5395, 1.2000 | ×1.1999 | 25 | 3.4e-05 / 5.6e-05 / 0.000239 | -0.0189..0.0283 |
| s12 | imgi_2542 | bedroom | 5.2 | fade 0.5 | 0.5993, 0.5907, 1.2551 | 0.4331, 0.5908, 1.1546 | ×0.9199 | 23 | 5.9e-05 / 0.000147 / 0.000338 | -0.035..0.0101 |
| s13 | imgi_2526 | bedroom | 5.2 | fade 0.5 | 0.4243, 0.5495, 1.1852 | 0.5150, 0.5800, 1.0309 | ×0.8698 | 25 | 5.4e-05 / 7e-05 / 0.000329 | -0.0145..0.0083 |


Mean 0.9763 is below 0.98, so the baseline is **not fully recovered**. v0 is the closest match and
was used as the baseline for v1 and v2.

### 1.5 What differs (measured)

**Geometry.** ECC-aligning v0 frames to liked frames gives:
- scale 1.000003
- centre offset of 0.08 px in x and 0.05–0.06 px in y
- no rotation

**Noise floor.** On frame 60:
- v0 aligned exactly to liked: SSIM 0.9678.
- Liked against itself shifted by 0.08 px and re-encoded at JPEG 80: 0.9867.
- An ideal warp of the original photo scores 0.9640 against liked and 0.9615 against v0.

Both renders sit equally far from the same source photo. What's left is each render's own
resampling and JPEG-capture noise, which differ between the two renders.

**Settings tried** (frames 60–89):

| Setting | SSIM |
|---|---|
| `--jpeg-quality 95` (park default) | 0.9599 |
| Remotion default JPEG (80) | 0.9648 |
| `--gl angle` | 0.9695 |
| `--gl swangle` | 0.9691 |

Liked frame 75 has Laplacian variance 1578. Default JPEG gives 1577, JPEG 95 gives 1500. So the
liked render was captured at the default JPEG quality, from before the park commit set it to 95.
The `--gl` options were rejected because they are not deterministic (§5.2). The final settings
are default JPEG and default GL.

**Rejected hypotheses:**
- An identity rotate/scale wrapper in KenBurnsShot: no change.
- Image preparation: 1920 px sources are not resized, since the limit is 2304.
- Higher-resolution sources: the trashed folder holds the same 1920×1280 files.

### 1.6 Branch and parallax

- **Branch:** `feat/tour-v2` from `1b64a33`.
- **Out of the render path:** parallax no longer appears in `remotion/src/TourComposition.tsx`
  (ParallaxShot is not imported), `modules/remotion_render.py` (no depth maps, no `--gl` for
  depth) or `run_tour.py` (no `build_depth_maps`, no `TOUR_PARALLAX`).
- **Kept but unreferenced:** `modules/depth.py`, `remotion/src/ParallaxShot.tsx` and
  `tests/test_depth.py`. The latter is skipped on this branch with a stated reason.
- **Park branch:** unchanged.
- **Dependencies:** `remotion/package.json` is unchanged (it still lists three.js, which is
  unused).

### 1.7 Fixture

`fixtures/downing/baseline_plan.json` (plan_hash `50e29ed9ada8…`), recovered cameras, `preset: "recovered"`.

| | |
|---|---|
| plan_version | 1.2 |
| total | 70.0 s |
| fades | 0.5 s on s02–s13 |
| easing | linear |
| roll | 0 |
| provenance | `recovered_from: virtual_tour_1080p.mp4` |

## 2. Step 2: saliency targets

**Implementation** (`tools/tour_v2/saliency.py`):
- opencv-contrib's `cv2.saliency` is not installed, and new dependencies are not allowed, so
  spectral residual (Hou & Zhang 2007) is ported to NumPy and OpenCV.
- Steps: EXIF transpose, grayscale, 64×64, FFT, log amplitude minus its 3×3 box blur, inverse
  FFT with the original phase, squared magnitude, Gaussian 5×5 σ 8, resize to 480 px wide,
  Gaussian smoothing σ = 2% of width, normalise.

**UNVERIFIED:** exact parity with opencv-contrib and with the backend's scoring code. Neither
was run or read.

**Metrics:**
- **peak:** centroid of the top 5% of pixels inside a 5% border
- **side:** whichever half holds more saliency mass, with the mass ratio
- **spread:** fraction of pixels above the mean

Output: `fixtures/downing/saliency.json` (all 19 photos) and `out/tour_v2/saliency_sheet.png`.

#### Saliency targets (plan photos)

| shot | photo | peak (x, y) | side | mass ratio | spread |
|---|---|---|---|---|---|
| s01 | imgi_2286 | 0.457, 0.406 | left | 1.051 | 0.464 |
| s02 | imgi_2416 | 0.461, 0.579 | right | 1.568 | 0.437 |
| s03 | imgi_2478 | 0.522, 0.624 | right | 1.017 | 0.482 |
| s04 | imgi_2366 | 0.409, 0.607 | left | 1.137 | 0.400 |
| s05 | imgi_2431 | 0.457, 0.619 | left | 1.359 | 0.402 |
| s06 | imgi_2326 | 0.724, 0.503 | right | 2.118 | 0.425 |
| s07 | imgi_2638 | 0.454, 0.511 | right | 1.044 | 0.354 |
| s08 | imgi_2446 | 0.517, 0.780 | left | 1.055 | 0.404 |
| s09 | imgi_2590 | 0.504, 0.353 | left | 1.292 | 0.363 |
| s10 | imgi_2346 | 0.478, 0.586 | left | 1.130 | 0.375 |
| s11 | imgi_2510 | 0.627, 0.697 | right | 1.354 | 0.352 |
| s12 | imgi_2542 | 0.483, 0.716 | right | 1.044 | 0.413 |
| s13 | imgi_2526 | 0.491, 0.633 | right | 1.200 | 0.391 |


**Observations on the saliency output:**
- **Peaks between hot spots:** when a photo has more than one hot spot, the top-5% centroid falls
  between them.
  - s01: peak (0.457, 0.406), between the left facade and the lit garage.
  - s09: peak (0.504, 0.353), on the wall above the bed.
- **Peak and side can disagree:** s02 has `side` right (×1.568) while its peak is at x = 0.461.
- **Narrow spread:** spread runs 0.352–0.482 across the 13 plan photos.
- **Duplicate photo:** `imgi_2366` and `imgi_2494` are the same photo (same content hash in the
  file name).

## 3. Step 3: contract addendum 1.3

### 3.1 Changes

**`modules/plan_io.py`:**
- plan_version 1.3 is accepted. Camera type `keyframe_path` is allowed only in 1.3; other
  versions reject it with `unsupported_camera_type`.
- Unknown camera types still fail with `unsupported_camera_type`. The type check now runs before
  the bounds check, so a malformed camera cannot crash it.
- Keyframe rules: 2–5 keyframes, first t = 0, last t = 1, strictly increasing, plus a move name.
- The window is checked inside the image at every rendered frame (`check_path_bounds`).
- `compute_plan_hash` is factored out.
- ken_burns is unchanged.

**`modules/camera_path.py`** (new, the single source of truth for the maths):
- the time profile
- Fritsch–Carlson tangents and evaluation
- window margin in output px
- pixel density

**`remotion/src/math.ts`:** `hermiteProfile`, `monotoneSlopes`, `monotoneEval` and
`keyframePathAt`, mirroring the Python. `KenBurnsShot.tsx` branches on `camera.type`.
`remotion_render.py` passes the keyframes to the renderer.

**Transitions and audio:** v2 uses `cut` throughout. Audio is unchanged: Remotion's silent AAC
track, the same as the liked video.

**Time profile:** cubic Hermite with s0 = 1.6 and s1 = 0.8. It is asserted monotonic, with a
minimum slope of 0.7667 at t = 5/6.

### 3.2 Tests

`tests/test_camera_path.py` covers:
- profile endpoints and monotonicity
- no overshoot between keyframes
- **Python ↔ TypeScript parity**: math.ts is bundled with esbuild and run in node over 4 paths
  × 65 t values. The difference is < 1e-12.
- 1.3 accepted; 1.2 rejected
- unknown type rejected
- each keyframe rule
- a path whose keyframes are all inside but which leaves the image between keyframes is rejected

```
$ pytest -q -rs tests/
...............ssss..........................                            [100%]
SKIPPED [1] tests/test_depth.py: parallax is out of the render path on feat/tour-v2
SKIPPED [1] tests/test_depth.py:32: depth model not downloaded (modules.depth.ensure_model)
SKIPPED [2] tests/test_depth.py:44: parallax is out of the render path on feat/tour-v2
41 passed, 4 skipped in 103.35s
```

### 3.3 Two properties of per-channel monotone cubics

1. **Zero velocity at reversals and flat segments.** Fritsch–Carlson sets a channel's tangent to
   0 wherever its direction reverses or it holds. As specified, `reveal_sweep` reverses cx at K1
   (push to one side, then scroll the other way) and at K2 (scroll, then a zoom-out with the edge
   fixed). Zoom holds between K1 and K2. So all three channels reach zero velocity at K1 and K2.

   Measured on v2 (grid speed in px/frame):

   | shot | frames | speed at K1 (frame) | speed at K2 (frame) | peak speed (frame) |
   |---|---|---|---|---|
   | s02 | 145–369 | 0.174 (202) | 0.053 (283) | 10.01 (235) |
   | s06 | 790–1014 | 0.169 (847) | 0.053 (928) | 10.01 (880) |
   | s08 | 1152–1376 | 0.188 (1209) | 0.053 (1290) | 10.01 (1242) |

   Velocity is continuous, but the camera stops and restarts twice within each sweep.
2. **Leaving the image between keyframes.** The path can exit the image even when every keyframe
   is inside, because zoom can sit still at a keyframe while cx is still moving. Example
   (in the tests): keyframes (cx, zoom) of (0.30, 1.7) → (0.5, 1.0) → (0.70, 1.7). Every
   keyframe is inside, yet frame 73 is 32 px outside. The every-frame check catches it.

## 4. Step 4: move selection (`tools/tour_v2/planner.py`)

### 4.1 Rules as implemented

**Move per shot:**
- Exterior first shot: `rise`.
- Final shot: `pull_back_reveal`, with a duration ≥ 5 s.
- Bathroom, or any shot under 4 s: `settle`.
- Hero with spread above the plan median (**0.4022**): `reveal_sweep`. Other heroes:
  `push_to_subject`. The brief gives no threshold; the median is my choice.

**Limits:**
- `reveal_sweep` on at most 40% of shots and never on two consecutive shots. Where a cut is
  needed, the highest spreads are kept.
- "Other" shots choose `slide` or `push_to_subject` together, checked against both neighbours:
  0 consecutive repeats.

**Durations:**
- `reveal_sweep` gets 7.5 s. The rest are scaled to keep the 70.0 s total and clamped to 3–8 s
  (final ≥ 5 s).
- Result: 4.57–5.0 s for the others, total 70.0 s (baseline 70.0 s).
- No shot is under 4 s, so no shot was converted to `settle`.

**Lateral direction:**
- The direction is constant within a narrative stage and flips at each stage change.
- living is −1 (taken from s02's natural direction); private is +1.
- Where a shot's natural direction (its salient side) conflicts, continuity wins unless the
  photo's mass ratio is above 1.5. No conflicting shot had a ratio above 1.5.

**Limits:**
- 1 px margin: keyframes are placed at 2 px, and every frame is checked. If a path fails, its
  keyframes shrink toward (0.5, 0.5) in 10% steps.
- Zoom cap: 1.3333 for these 1920 px sources, from source px per output px ≥ 0.75.

### 4.2 Move parameters

| Move | Keyframes and values |
|---|---|
| `reveal_sweep` | K0 zoom 1.035; K1 t 0.35, zoom 1.28; K2 t 0.70, zoom 1.28, same cy; K3 zoom 1.08, keeping the side edge and the nearer vertical edge of K2 fixed |
| `push_to_subject` | zoom 1.02 → 1.1985 (+17.5%); ends with the peak on its nearest thirds point; start offset 6% of window width against the travel direction; mid keyframe with x at 70% and y at 40% of the way (the curve) |
| `slide` | travel 10% of window width; push +4%; lowest zoom that fits (1.105) |
| `pull_back_reveal` | 1.175 on the peak → 1.02 centred |
| `rise` | 1.01 → 1.0756 (+6.5%), from the lowest to the highest window position, cx halfway to the peak |
| `settle` | 1.02 → 1.0914 (+7%), 35% of the way toward the peak |

### 4.3 Direction results

The chosen and achieved directions agree for `slide` and `reveal_sweep`. For `push_to_subject`
they disagree in four shots: **s05, s07** (chosen −1, achieved +1) and **s09, s11** (chosen +1,
achieved −1; s11 moves only dx = −0.0037).

The move must end with the peak on its nearest thirds point, which fixes the final window. Its
start is near full frame (zoom 1.02), so there is no room to start on the other side. The
continuity rule cannot be honoured without breaking the thirds rule. I kept the move as defined
and recorded `direction.conflict` in `v2_plan_notes.json`.

## 5. Step 5: renders and checks

### 5.1 Renders

| File | Plan | Render s |
|---|---|---|
| `v0_baseline.mp4` | fixtures/downing/baseline_plan.json | 150.36 |
| `v1_cuts.mp4` | `v1_cuts.plan.json` (baseline cameras, every transition a cut) | 149.10 |
| `v2_moves.mp4` | `v2_moves.plan.json` | 147.79 |
| `determinism/v2_moves.run2.mp4` | `v2_moves.plan.json` | 149.26 |

All four use concurrency 6, default JPEG and default GL.

```
ffprobe (all four renders and the reference):
duration 70.059 s, 2100 frames, 30/1, 1920x1080, h264, yuv420p, audio aac 48000 Hz 2 ch
volumedetect: mean_volume -91.0 dB, max_volume -91.0 dB (silent, same as liked_baseline.mp4)
compare_grid.mp4: 960x1620, 2100 frames, 30/1, h264 (v0 / v1 / v2 stacked, labelled, frame counter)
```

### 5.2 Determinism (v2, decoded-frame md5 via `ffmpeg -f framemd5`)

```
default JPEG, --gl unset (final):  frames [2100, 2100] identical=True  n_differing=0    digest 86f37112aedc04d5
JPEG 95, --gl unset (park):        frames [2100, 2100] identical=True  n_differing=0    digest a6156cc1772ba90a
default JPEG, --gl angle:          frames [2100, 2100] identical=False n_differing=839  (frames 491-1151, 1635-1812)
default JPEG, --gl swangle:        frames [2100, 2100] identical=False n_differing=690  (frames 475-644, 1002-1521)
```

With `--gl` set, differences reach 25–40 levels (mean |Δ| about 1.5) over the whole frame, with no
geometric shift. The longest run (520 frames) exceeds the x264 keyframe interval (250), so the
captured frames themselves differ, not only the encoding.

The renders that used `--gl` are kept for evidence in `out/tour_v2/angle_nondeterministic/` and
`out/tour_v2/swangle_nondeterministic/`.

### 5.3 Inside-image margin, every frame (output px)

| Render | frames checked | min margin | where | min source px / output px |
|---|---|---|---|---|
| v0 | 2280 (includes the dissolve overlaps) | −0.133 | s10, frame 1451 | 0.7967 |
| v1 | 2100 | −0.133 | s10, frame 1458 | 0.7967 |
| v2 | 2100 | **1.999** | s02, frame 369 | **0.7812** |

v0 and v1 are the recovered baseline cameras. The baseline planner placed windows exactly on the
image edge, and the fit puts them within 0.13 px of it (≈ 0.0001 normalised, inside the plan
validator's tolerance). The 1 px rule applies to `keyframe_path` and is met by v2.

**Projected to 3840×2160:** worst v2 pixel density is 0.3906 (the `reveal_sweep` shots at zoom
1.28).

### 5.4 Per shot

#### v2 per shot

| shot | role / room / stage | move | why | frames | dur s (baseline) | keyframes (t: cx, cy, zoom) | zoom range | min margin px | min spp (frame) | 4K spp | v start / end px/f | min speed inside px/f (frame) | max px/f | direction nat/stage/chosen/achieved | shrink | zoom cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s01 | hero / exterior / exterior_open | rise | exterior, first shot | 0–144 | 4.84 (5.5) | 0.0: 0.4961, 0.5808, 1.0100<br>1.0: 0.4783, 0.3937, 1.0756 | 1.01–1.0756 | 2.0 (f144) | 0.9297 (f144) | 0.4648 | 2.734 / 1.455 | 1.3861 (f117) | 2.734 | +0 / – / +1 / -1 | 1.0 | none |
| s02 | hero / living_room / living | reveal_sweep | hero, spread 0.437 > median 0.402 | 145–369 | 7.5 (5.5) | 0.0: 0.5000, 0.5393, 1.0350<br>0.35: 0.6086, 0.5787, 1.2800<br>0.7: 0.3914, 0.5787, 1.2800<br>1.0: 0.4639, 0.5177, 1.0800 | 1.035–1.28 | 1.999 (f369) | 0.7812 (f205) | 0.3906 | 5.009 / 2.28 | 0.0528 (f283) | 10.006 | -1 / -1 / -1 / -1 | 1.0 | none |
| s03 | detail / living_room / living | push_to_subject | other shot, slide/push chosen against both neighbours | 370–507 | 4.58 (5.2) | 0.0: 0.4912, 0.5000, 1.0200<br>0.5: 0.4532, 0.5028, 1.1057<br>1.0: 0.4181, 0.5070, 1.1985 | 1.02–1.1985 | 1.999 (f507) | 0.8344 (f507) | 0.4172 | 1.899 / 1.001 | 0.9389 (f480) | 1.899 | -1 / -1 / -1 / -1 | 1.0 | none |
| s04 | detail / living_room / living | slide | other shot, slide/push chosen against both neighbours | 508–644 | 4.58 (5.2) | 0.0: 0.5439, 0.6072, 1.1050<br>1.0: 0.4534, 0.6072, 1.1492 | 1.105–1.1492 | 7.6 (f508) | 0.8702 (f644) | 0.4351 | 2.254 / 1.175 | 1.1221 (f620) | 2.254 | -1 / -1 / -1 / -1 | 1.0 | none |
| s05 | hero / kitchen / living | push_to_subject | hero, spread 0.402 <= median 0.402 | 645–789 | 4.84 (5.5) | 0.0: 0.5088, 0.5000, 1.0200<br>0.5: 0.5468, 0.5007, 1.1057<br>1.0: 0.5819, 0.5018, 1.1985 | 1.02–1.1985 | 1.999 (f789) | 0.8344 (f789) | 0.4172 | 1.805 / 0.95 | 0.8905 (f761) | 1.805 | +1 / -1 / -1 / +1 | 1.0 | none |
| s06 | hero / dining_room / living | reveal_sweep | hero, spread 0.425 > median 0.402 | 790–1014 | 7.5 (5.5) | 0.0: 0.5000, 0.5013, 1.0350<br>0.35: 0.6086, 0.5026, 1.2800<br>0.7: 0.3914, 0.5026, 1.2800<br>1.0: 0.4639, 0.4416, 1.0800 | 1.035–1.28 | 1.999 (f1014) | 0.7812 (f850) | 0.3906 | 4.883 / 2.28 | 0.0528 (f928) | 10.006 | -1 / -1 / -1 / -1 | 1.0 | none |
| s07 | detail / dining_room / living | push_to_subject | other shot, slide/push chosen against both neighbours | 1015–1151 | 4.57 (5.2) | 0.0: 0.5088, 0.5000, 1.0200<br>0.5: 0.5468, 0.4573, 1.1057<br>1.0: 0.5819, 0.3934, 1.1985 | 1.02–1.1985 | 1.999 (f1151) | 0.8344 (f1151) | 0.4172 | 2.345 / 1.557 | 1.5076 (f1125) | 2.345 | +1 / -1 / -1 / +1 | 1.0 | none |
| s08 | hero / basement / living | reveal_sweep | hero, spread 0.404 > median 0.402 | 1152–1376 | 7.5 (5.5) | 0.0: 0.5000, 0.5909, 1.0350<br>0.35: 0.6086, 0.6692, 1.2800<br>0.7: 0.3914, 0.6692, 1.2800<br>1.0: 0.4639, 0.6079, 1.0800 | 1.035–1.28 | 1.999 (f1376) | 0.7812 (f1212) | 0.3906 | 5.375 / 2.282 | 0.0528 (f1290) | 10.006 | +1 / -1 / -1 / -1 | 1.0 | none |
| s09 | hero / primary_bedroom / private | push_to_subject | hero, spread 0.363 <= median 0.402 | 1377–1521 | 4.84 (5.5) | 0.0: 0.4912, 0.5000, 1.0200<br>0.5: 0.4532, 0.4883, 1.1057<br>1.0: 0.4181, 0.4708, 1.1985 | 1.02–1.1985 | 1.999 (f1521) | 0.8344 (f1521) | 0.4172 | 1.838 / 0.999 | 0.9414 (f1492) | 1.838 | -1 / 1 / +1 / -1 | 1.0 | none |
| s10 | hero / primary_bathroom / private | settle | bathroom | 1522–1667 | 4.84 (5.5) | 0.0: 0.5000, 0.5000, 1.0200<br>1.0: 0.4923, 0.5301, 1.0914 | 1.02–1.0914 | 19.2 (f1522) | 0.9163 (f1667) | 0.4582 | 0.631 / 0.319 | 0.3053 (f1642) | 0.631 | +0 / 1 / +1 / -1 | 1.0 | none |
| s11 | hero / bedroom / private | push_to_subject | hero, spread 0.352 <= median 0.402 | 1668–1812 | 4.84 (5.5) | 0.0: 0.4912, 0.5000, 1.0200<br>0.5: 0.4886, 0.5317, 1.1057<br>1.0: 0.4875, 0.5793, 1.1985 | 1.02–1.1985 | 1.999 (f1668) | 0.8344 (f1812) | 0.4172 | 1.401 / 0.945 | 0.9349 (f1791) | 1.401 | -1 / 1 / +1 / -1 | 1.0 | none |
| s12 | detail / bedroom / private | slide | other shot, slide/push chosen against both neighbours | 1813–1949 | 4.57 (5.2) | 0.0: 0.4534, 0.6168, 1.1050<br>1.0: 0.5439, 0.6315, 1.1492 | 1.105–1.1492 | 1.999 (f1813) | 0.8702 (f1949) | 0.4351 | 2.268 / 1.182 | 1.1288 (f1925) | 2.268 | +1 / 1 / +1 / +1 | 1.0 | none |
| s13 | detail / bedroom / private | pull_back_reveal | final shot | 1950–2099 | 5.0 (5.2) | 0.0: 0.4911, 0.6332, 1.1750<br>1.0: 0.5000, 0.5000, 1.0200 | 1.02–1.175 | 11.667 (f1950) | 0.8511 (f1950) | 0.4255 | 2.29 / 1.042 | 1.0117 (f2077) | 2.29 | +0 / 1 / +1 / +1 | 1.0 | none |

#### v0_baseline per shot

| shot | frames (render span) | zoom range | min margin px (frame) | min spp | v start / end px/f |
|---|---|---|---|---|---|
| s01 | 0–172 | 1.035–1.1592 | 0.0 (f0) | 0.8627 | 0.994 / 1.106 |
| s02 | 158–337 | 1.1546–1.2549 | 0.043 (f337) | 0.7969 | 2.243 / 2.065 |
| s03 | 323–493 | 1.035–1.1592 | -0.001 (f323) | 0.8627 | 1.006 / 1.12 |
| s04 | 479–649 | 1.13–1.243 | 0.0 (f649) | 0.8045 | 1.953 / 2.146 |
| s05 | 635–814 | 0.9999–1.2001 | -0.118 (f635) | 0.8332 | 0.875 / 0.987 |
| s06 | 800–979 | 1.1545–1.2552 | -0.0 (f979) | 0.7967 | 2.243 / 2.064 |
| s07 | 965–1135 | 1.0351–1.1589 | 0.044 (f965) | 0.8629 | 1.007 / 1.12 |
| s08 | 1121–1300 | 1.13–1.2431 | 0.022 (f1121) | 0.8044 | 1.855 / 2.039 |
| s09 | 1286–1465 | 1.0–1.2 | -0.0 (f1286) | 0.8333 | 0.874 / 0.986 |
| s10 | 1451–1630 | 0.9999–1.0601 | -0.133 (f1451) | 0.9433 | 0.841 / 0.889 |
| s11 | 1616–1795 | 1.0–1.2 | 0.001 (f1616) | 0.8334 | 0.874 / 0.986 |
| s12 | 1781–1951 | 1.1546–1.2551 | -0.001 (f1951) | 0.7968 | 2.36 / 2.173 |
| s13 | 1937–2099 | 1.0309–1.1852 | -0.001 (f2099) | 0.8438 | 1.339 / 1.176 |

#### v1_cuts per shot

| shot | frames (render span) | zoom range | min margin px (frame) | min spp | v start / end px/f |
|---|---|---|---|---|---|
| s01 | 0–164 | 1.035–1.1592 | 0.0 (f0) | 0.8627 | 1.043 / 1.16 |
| s02 | 165–329 | 1.1546–1.2549 | 0.043 (f329) | 0.7969 | 2.448 / 2.254 |
| s03 | 330–485 | 1.035–1.1592 | -0.001 (f330) | 0.8627 | 1.104 / 1.228 |
| s04 | 486–641 | 1.13–1.243 | 0.0 (f641) | 0.8045 | 2.142 / 2.354 |
| s05 | 642–806 | 0.9999–1.2001 | -0.118 (f642) | 0.8332 | 0.955 / 1.077 |
| s06 | 807–971 | 1.1545–1.2552 | -0.0 (f971) | 0.7967 | 2.448 / 2.253 |
| s07 | 972–1127 | 1.0351–1.1589 | 0.044 (f972) | 0.8629 | 1.104 / 1.228 |
| s08 | 1128–1292 | 1.13–1.2431 | 0.022 (f1128) | 0.8044 | 2.025 / 2.225 |
| s09 | 1293–1457 | 1.0–1.2 | -0.0 (f1293) | 0.8333 | 0.954 / 1.077 |
| s10 | 1458–1622 | 0.9999–1.0601 | -0.133 (f1458) | 0.9433 | 0.918 / 0.97 |
| s11 | 1623–1787 | 1.0–1.2 | 0.001 (f1623) | 0.8334 | 0.954 / 1.076 |
| s12 | 1788–1943 | 1.1546–1.2551 | -0.001 (f1943) | 0.7968 | 2.588 / 2.383 |
| s13 | 1944–2099 | 1.0309–1.1852 | -0.001 (f2099) | 0.8438 | 1.399 / 1.23 |

#### Speed across each cut (px/frame at 1920×1080: outgoing last step → incoming first step)

| cut | v0 (dissolve) | v1 | v2 |
|---|---|---|---|
| s01→s02 (v2 f145) | 1.106 → 2.243 | 1.16 → 2.448 | 1.455 → 5.009 |
| s02→s03 (v2 f370) | 2.065 → 1.006 | 2.254 → 1.104 | 2.28 → 1.899 |
| s03→s04 (v2 f508) | 1.12 → 1.953 | 1.228 → 2.142 | 1.001 → 2.254 |
| s04→s05 (v2 f645) | 2.146 → 0.875 | 2.354 → 0.955 | 1.175 → 1.805 |
| s05→s06 (v2 f790) | 0.987 → 2.243 | 1.077 → 2.448 | 0.95 → 4.883 |
| s06→s07 (v2 f1015) | 2.064 → 1.007 | 2.253 → 1.104 | 2.28 → 2.345 |
| s07→s08 (v2 f1152) | 1.12 → 1.855 | 1.228 → 2.025 | 1.557 → 5.375 |
| s08→s09 (v2 f1377) | 2.039 → 0.874 | 2.225 → 0.954 | 2.282 → 1.838 |
| s09→s10 (v2 f1522) | 0.986 → 0.841 | 1.077 → 0.918 | 0.999 → 0.631 |
| s10→s11 (v2 f1668) | 0.889 → 0.874 | 0.97 → 0.954 | 0.319 → 1.401 |
| s11→s12 (v2 f1813) | 0.986 → 2.36 | 1.076 → 2.588 | 0.945 → 2.268 |
| s12→s13 (v2 f1950) | 2.173 → 1.339 | 2.383 → 1.399 | 1.182 → 2.29 |

#### v0 vs liked baseline: shot identity at each boundary (±9 frames, ORB matches)

| boundary | expected | liked before/after | v0 before/after | ok |
|---|---|---|---|---|
| 165 | s01→s02 | s01/s02 | s01/s02 | True |
| 330 | s02→s03 | s02/s03 | s02/s03 | True |
| 486 | s03→s04 | s03/s04 | s03/s04 | True |
| 642 | s04→s05 | s04/s05 | s04/s05 | True |
| 807 | s05→s06 | s05/s06 | s05/s06 | True |
| 972 | s06→s07 | s06/s07 | s06/s07 | True |
| 1128 | s07→s08 | s07/s08 | s07/s08 | True |
| 1293 | s08→s09 | s08/s09 | s08/s09 | True |
| 1458 | s09→s10 | s09/s10 | s09/s10 | True |
| 1623 | s10→s11 | s10/s11 | s10/s11 | True |
| 1788 | s11→s12 | s11/s12 | s11/s12 | True |
| 1944 | s12→s13 | s12/s13 | s12/s13 | True |


## 6. Per-shot notes (facts, output frame numbers)

- **s01 rise (frames 0–144):**
  - Starts at its fastest, 2.73 px/frame.
  - Window rises from cy 0.581 to 0.394 (about 240 source px).
  - Ends at 1.46 px/frame into the cut at frame 145.
- **s02, s06, s08 reveal_sweep (7.5 s each):**
  - Stops at K1 (frames 202, 847, 1209) and K2 (283, 928, 1290).
  - The scroll between them peaks at 10.0 px/frame, about 300 px/s (frames 235, 880, 1242).
  - Lowest pixel density in the tour: 0.7812 at zoom 1.28.
  - s08's natural direction (+1) was overridden by continuity (mass ratio 1.055).
- **s03, s05, s07, s09, s11 push_to_subject:**
  - End speeds 0.95–1.56 px/frame.
  - Direction conflicts in s05, s07, s09 and s11 (§4.3).
- **s04, s12 slide:**
  - Zoom 1.105 → 1.149.
  - Start speed about 2.26 px/frame, end speed 1.18 px/frame.
- **s10 settle (bathroom, frames 1522–1667):**
  - The slowest shot: 0.63 px/frame at the start, 0.32 px/frame at the end.
  - Margin 19.2 px.
- **s13 pull_back_reveal (frames 1950–2099, 5.0 s):**
  - Zoom 1.175 → 1.02.
  - Ends at 1.04 px/frame on the last frame.
- **Speed at the cuts:** every v2 cut goes from an outgoing end speed of 0.32–2.28 px/frame (lowest: s10 settle into s11) to an
  incoming start speed of 0.63–5.38 px/frame. The highest is into s08 at frame 1152, at 5.375.
  Full table below.

## 7. Open questions

1. **Recovery threshold:** v0 reaches mean SSIM 0.9763 against the liked video, which is short of
   0.98. The gap is renderer noise; camera error is under 0.1 px. Is v0 accepted as the baseline,
   or is pixel-identical recovery needed? That would require the original plan file and the
   exact render environment.
2. **`reveal_sweep` stops:** it stops at K1 and K2 under Fritsch–Carlson. Options:
   - accept the "arrive, then scroll" pauses
   - drop the hold (K1 and K2 at different zooms) so zoom keeps moving through them
   - use a non-monotone spline for this move only, at the risk of overshoot
3. **Scroll speed:** the `reveal_sweep` scroll peaks at 10 px/frame (300 px/s). Is that too fast?
   A 7.5 s shot cannot make it slower without a smaller sweep.
4. **`push_to_subject` direction:** continuity and the thirds framing conflict in 4 of 6 shots.
   Which rule wins: framing on the far third, or exempting push moves from continuity?
5. **"High spread" threshold:** the brief gives none. I used the plan median (0.4022). With
   spreads of 0.352–0.482, the hero/push split is close: s05 is at exactly the median and s08 is
   above it by 0.002.
6. **Saliency peaks:** the top-5% centroid lands between hot spots on multi-subject photos (s01,
   s09). Should the peak use the largest connected region instead?
7. **Rendering knobs:** `--jpeg-quality` default (80) matches the liked render, but park had moved
   to 95 to reduce texture crawl. Which should production use? And should `--gl` stay unset given
   the determinism results?
8. **Edge-touching windows:** v0 and v1 windows touch the image edge (−0.13 px). Should the 1 px
   rule apply to ken_burns as well?

## 8. Files

`out/tour_v2/`:
- **Renders:** `v0_baseline.mp4`, `v1_cuts.mp4`, `v2_moves.mp4`, `compare_grid.mp4`,
  `determinism/v2_moves.run2.mp4`, and `determinism_test/` (the gl-unset pairs).
- **Images:** `saliency_sheet.png`, and `v2_contact_sheet.png` (first and last frame of every v2
  shot).
- **Plans:** `v1_cuts.plan.json`, `v2_moves.plan.json`, `v2_plan_notes.json` (every selection,
  direction and enforcement decision).
- **Checks:** `checks.json`, `checks.log`, `baseline_fits.json`, `recover_baseline.log`,
  `planner.log`, `saliency.log`, `timings.jsonl`.
- **Evidence for the `--gl` determinism finding:** `angle_nondeterministic/` and
  `swangle_nondeterministic/`.

Committed on `feat/tour-v2`:
- code
- `fixtures/downing/{baseline_plan.json, saliency.json}`
- the text deliverables under `out/tour_v2/`

Not committed: the mp4s, the PNG sheets (4.5 MB and 6.1 MB) and the source photos.

Reproduce:
```
D=<Downing photo folder>
python -m tools.tour_v2.recover_baseline --video out/reference/liked_baseline.mp4 \
  --skeleton "$D/old_generations/downing_tour_plan_restored.json" --images-dir "$D" \
  --out fixtures/downing/baseline_plan.json --fits-out out/tour_v2/baseline_fits.json
python -m tools.tour_v2.saliency --images-dir "$D" --plan fixtures/downing/baseline_plan.json \
  --out fixtures/downing/saliency.json --sheet out/tour_v2/saliency_sheet.png
python -m tools.tour_v2.planner --baseline fixtures/downing/baseline_plan.json \
  --saliency fixtures/downing/saliency.json --out-dir out/tour_v2
python -m tools.tour_v2.render --plan <plan> --images-dir "$D" --out <mp4> --jpeg-quality default
python -m tools.tour_v2.checks all --out-dir out/tour_v2 --images-dir "$D" \
  --reference out/reference/liked_baseline.mp4 --baseline fixtures/downing/baseline_plan.json
```
