# Tour v5: rotate reveals, smooth pull-backs, opposite motion at cuts, varied directions

Branch `feat/tour-v2`, uncommitted. Built on `checkpoint/v4-compound` (`848ede5`); `checkpoint/v3-tilt-reveal`
is `c306c44`.

The render is 1920×1080, 30 fps, 2250 frames, 75.0 s, with hard cuts.
- Determinism: two renders, 0 / 2250 frames differ (framemd5).
- Tests: 63 passed, 4 skipped.

## Review points, and what changed

1. **Use the v3 rotate reveal, not the crop one.**
   - `tilt_reveal_rotate` (a camera turning in place) is now the preferred reveal:
     - wide rooms: 5, against 2 for the crop version
     - medium rooms: 3.5
     - final shot: 3
   - It is used in s02 and s05.
   - **Render fix.** Chrome's CSS 3D path was non-deterministic:
     - 300–408 of 2250 frames differed between two renders.
     - Crop shots sharing a render chunk with a rotate shot were affected too.
     - `will-change: transform` and remounting the layer every frame did not fix it.
   - The worker now warps rotate shots itself (`modules/prewarp.py`):
     - OpenCV `warpPerspective` with the exact homography, one JPEG (q95) per frame.
     - Remotion only shows those frames.
     - Linear interpolation matches Chrome's sampling of crop shots: SSIM 0.96–0.97, same
       sharpness (Laplacian variance 439/550 vs Chrome's 456/570).
     - Cost: render time 160 s → 205 s.
2. **Jerky pull-back.**
   - **Cause, measured.** v4 swung the camera's direction by 45–57° within one frame at full speed
     (4.3–4.7 px/frame), exactly where the pull-back starts.
     - v4's ported moves overlapped their phases by only 0.06.
     - The distance-based timing then held full speed through the near-corner.
     - v3 slowed to 1.25 px/frame there (4°/frame).
   - **Fix, part 1:** a phase overlap of 0.25 (0.18 for `reveal_sweep`, whose three corners sit
     close together) rounds the corners.
   - **Fix, part 2:** new optional `timing.turn` (addendum 1.3.2a), set to 60 px per radian. Turning
     costs time as well as distance, smoothed over ±40 of 512 samples, so the camera slows into a
     turn and speeds up out of it.
   - **Result:** the maximum turn is 0.6–1.4°/frame on most shots and at most 4.0°/frame (the
     v2 sweeps; v3 had 4°/frame). The rotate reveal's slowest point is 1.13 px/frame.
3. **After a zoom-out, the next room should move the opposite way, with no hard jumps.** The
   cut rule is new (`continuity` in `tools/tour_v3/planner.py`):
   - **Framing carries across the cut.** The next room opens at about the zoom the last one closed
     on: −30 per unit of |ln zoom ratio| beyond 0.03. Cut zoom ratios are now 0.955–1.038. In v4,
     a 1.012 close often cut to a 1.32 open.
   - **Motion reverses.**
     - Zoom-out → zoom-in scores +2, and the same direction −2.
     - Pan left → pan right scores +1, and the same direction −0.5.
     - Tilt: +0.3 / −0.3.
4. **Varied directions.** Repeating "start on the right, zoom in, tilt, pull back" is penalised:
   - −4 for two reveal-family moves in a row, −1.5 two shots apart.
   - At most 4 reveal-family moves per tour.
   - −1 for opening on the same side of the frame as the previous shot, and −1 for the same first
     motion.
   - −2 per earlier use of the same move with the same mirror and flip, and −1.5 per earlier use
     with the same pan direction.
   - Result: the two rotate reveals go opposite ways (s02 left and tilting down, s05 right and up),
     the two `reveal_sweep`s are mirrored, and pans alternate.

## Per shot

Opening and closing motion, and the cut zoom ratio (this shot's first zoom ÷ the previous shot's
last zoom).

| Shot | Room | Category | Move | Opens | Closes | Cut zoom ratio | Seconds | Max px/frame | Max turn °/frame | Shrink |
|---|---|---|---|---|---|---|---|---|---|---|
| s01 | exterior | exterior | crane | right up | in right up | – | 5.34 | 3.06 | 0.94 | 1.0 |
| s02 | living_room | wide | tilt_reveal_rotate | left down | out right | 1.037 | 7.42 | 1.82 | 0.99 | 0.65 |
| s03 | living_room | wide | reveal_sweep | in left | out left | 1.027 | 7.47 | 4.85 | 3.85 | 0.9 |
| s04 | living_room | medium | push_pan | in left | in left down | 0.972 | 4.95 | 2.94 | 0.64 | 1.0 |
| s05 | kitchen | medium | tilt_reveal_rotate | right up | out left | 0.955 | 7.42 | 1.82 | 0.99 | 0.65 |
| s06 | dining_room | wide | reveal_sweep | in right | out right | 1.027 | 7.47 | 4.82 | 4.03 | 0.9 |
| s07 | dining_room | tight | slide_push | left | in left | 1.019 | 4.64 | 3.38 | 1.13 | 1.0 |
| s08 | basement | wide | rise_reveal | right up | out left down | 1.032 | 7.47 | 4.57 | 3.18 | 1.0 |
| s09 | primary_bedroom | tight | push_pan | in right | in right up | 1.038 | 5.28 | 2.82 | 0.62 | 1.0 |
| s10 | primary_bathroom | bath | pull_back | out left up | out right | 0.992 | 4.12 | 2.64 | 0.91 | 1.0 |
| s11 | bedroom | tight | settle_push | in left | in left | 1.018 | 3.5 | 1.56 | 0.03 | 1.0 |
| s12 | bedroom | wide | slide_push | right | in right | 0.965 | 4.91 | 3.35 | 1.36 | 1.0 |
| s13 | bedroom | final | pull_back | out right up | out left up | 0.992 | 5.0 | 2.35 | 0.79 | 1.0 |

**Constraints:**
- Inside-image margin ≥ 1.999 px on every frame.
- Pixel density ≥ 0.7551 (floor 0.75).
- Every shot enters and leaves at 1.55–1.62 px/frame.
- No shot exceeds 5 px/frame.

## Open questions and limits

- **How I read "opposite the motion":** both the zoom and the pan reverse at each cut. If you meant
  only the zoom, set the lateral weights in `continuity` to 0.
- **The rotate reveals are subtle,** running at 65% size and a peak of 1.8 px/frame. Rotation reaches
  about 40% of what a crop does on 1920 px photos, and the 0.75 sharpness floor caps the zoom. A
  wider assumed lens (`hfov_deg`) or larger source photos would make them bigger.
- **Rotate shots (s04→s05) do not alternate pan direction.** The push in s04 travels left and the
  s05 reveal opens to the right. That pan reversal is the rule working; the zoom continues
  (in → neutral).
- **The total is 75 s** (cap 70 + 5). The reveals need the time to stay slow.
- **Pre-warp disk use:** 223 frames per rotate shot, about 0.3 MB each, about 140 MB of temporary
  files per tour.

## Files

- `v5.plan.json`, `v5_plan_notes.json`, `v5*.framemd5`.
- `v5.mp4` and `compare_v4_v5.mp4` (not committed). They are copied to the Downing folder as
  `virtual_tour_v5.mp4` and `virtual_tour_v5_vs_v4_compare.mp4`.
- **Code:**
  - `modules/camera_path.py`: turn-aware timing
  - `remotion/src/math.ts`: mirror, parity-tested
  - `modules/prewarp.py`: new
  - `modules/remotion_render.py`, `remotion/src/KenBurnsShot.tsx`, `remotion/src/TourComposition.tsx`
  - `tools/tour_v3/planner.py`: v5 rules
  - `tests/test_prewarp.py`: new

**Reproduce:**

    python -m tools.tour_v3.planner --baseline fixtures/downing/baseline_plan.json \
        --saliency fixtures/downing/saliency.json --images-dir <photos> --out-dir out/tour_v5
    python -m tools.tour_v2.render --plan out/tour_v5/v5.plan.json --images-dir <photos> \
        --out out/tour_v5/v5.mp4 --jpeg-quality default

## Playback fix (smooth playback in media players)

**Problem:** the file stalled, buffered, played for a second, then stalled again. Measured on the
old `v5.mp4` (now `v5_unconstrained.mp4`):

- **Bitrate:** Remotion encodes at constant quality (CRF 18) with no ceiling. The video averaged
  14.8 Mbps and peaked at 36.1 Mbps in a single second (the exterior's foliage, seconds 0–4). The
  file was tagged H.264 High Level 4.0, whose limit is 25 Mbps.
- **Keyframes:** 13 in 75 s, with the longest gap 7.5 s. Players restart slowly after a stall
  or seek.
- **Buffer test:** a player receiving 10 Mbps into a 20 Mbit buffer drops to a fullness of −17.1,
  so it would run dry repeatedly. That is the stall-and-buffer pattern.

**Fix:** a delivery encode after Remotion (`modules/encode.py`).

- Remotion now writes a high-quality master at CRF 16. ffmpeg/x264 then re-encodes it:
  - `-crf 20` under a VBV cap (`-maxrate 10M -bufsize 20M`, `nal-hrd=vbr`)
  - High profile, Level 4.1
  - a keyframe at least every 2 s (`-g 60 -keyint_min 30`), plus scene-cut keyframes
  - `+faststart`
  - AAC 128k
  - colour passed through unchanged (BT.601 limited range, as tagged before). Converting to
    BT.709 shifted colour by about 2 levels, so it was left out.
- **x264 runs with sliced threads.** Default frame threading with a VBV cap was not deterministic:
  374 of 600 frames differed across two encodes of the same master. Sliced threading is
  byte-identical run to run and about 5.7× faster than one thread, for files about 1.6% larger.
- **`modules/probe.py` now rejects** a delivery file that has any of these:
  - no faststart
  - a keyframe gap over 2 s (including the stretch from the last keyframe to the end)
  - H.264 level above 4.1
  - a buffer underrun for a 10 Mbps / 20 Mbit player (leaky-bucket simulation in decode order)

| | old v5 | new v5 |
|---|---|---|
| size | 141.7 MB | 84.9 MB |
| mean / peak 1 s bitrate | 14.8 / 36.1 Mbps | 9.0 / 14.0 Mbps |
| buffer fullness, worst (10 Mbps, 20 Mbit) | −17.08 (stalls) | 0.434 (never below 43%) |
| keyframes / longest gap | 13 / 7.5 s | 39 / 2.0 s |
| H.264 level | 4.0 (exceeded) | 4.1 |
| faststart | yes | yes |

**Quality** of new vs old: SSIM mean 0.9854, median 0.9861, min 0.9679 (frame 58, exterior
foliage); 2 of 2250 frames are below 0.97.

**Determinism:** two full renders give byte-identical files (0 / 2250 frames differ).

**Render time:** about 213 s, including about 10 s of encode.

**Tests:** 65 passed, 4 skipped. New: `tests/test_encode.py` covers the cap, keyframes,
faststart, determinism and the buffer simulation.

**Downing folder:** `virtual_tour_v5_smooth.mp4`. The old `virtual_tour_v5.mp4` is left as it was.
