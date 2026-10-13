# Tour v3: tilt_reveal (tight start, pan + tilt, zoom out to full coverage)

Branch `feat/tour-v2`, uncommitted. All renders are 1920×1080, 30 fps, 2100 frames, 70.0 s, with hard cuts.

## The move

`tilt_reveal` (in `tools/tour_v2/planner.py`) has three keyframes:

| Keyframe | t | Zoom | Aim |
|---|---|---|---|
| K0 | 0 | 1.32 | low on the side away from the travel direction |
| K1 | 0.62 | 1.24 | the salient side, tilted up to eye level |
| K2 | 1 | 1.012 | the whole room; the K1 side's window edge stays fixed, so it reads as pulling back from that corner |

Every channel is monotone except cx at K1, and zoom is still moving there, so the camera never stops.

**Speed** (crop, s03, output px/frame):

| Point | Speed |
|---|---|
| First frame | 6.8 |
| Peak | 7.8 |
| Slowest, at K1 | 1.3 |
| At the cut | 3.5 |

For comparison, v2's `reveal_sweep` dropped to 0.05 px/frame.

**Where it is used:** the widest rooms, meaning saliency spread above the plan median.
- Excluded: the first and last shots, exteriors and bathrooms.
- At most 40% of shots, never two in a row.
- Each gets 7.5 s. The others are scaled to keep the 70 s total.
- In this plan it lands on s03, s06, s08 and s12.

## Projection modes (addendum 1.3.1, optional fields on keyframe_path)

- `projection: "crop"` (the default) slides a window over the photo.
- `projection: "rotate"` turns a virtual camera in place.
  - Rotating a camera about its lens is exactly a homography of a single photo, so it needs no
    depth and keeps straight lines straight.
  - Vertical aim becomes a real tilt (verticals lean) and horizontal aim a real pan.
  - `hfov_deg` (default 75) is the assumed lens field of view. The real one is unknown, because
    Zillow strips EXIF.
- **Code:** `modules/camera_path.py` (`rotate_homography`, `frame_margin_out_px`, `frame_spp`) and
  `remotion/src/math.ts` (`rotateHomography`), rendered as CSS `matrix3d`.
- **Checks:**
  - Python and TypeScript parity is tested.
  - Rendered frames vs an OpenCV `warpPerspective` reference: SSIM 0.968–0.978. That is the same
    level as encoding noise on the baseline re-render (0.976).

## Measured results

| | crop | rotate (hfov 75) |
|---|---|---|
| s03 K0 → K1 pan (fraction of width) | 0.194 | 0.077 |
| s03 tilt (fraction of height) | 0.161 | 0.064 |
| start zoom after the density floor | 1.32 | 1.206 |
| min px density (floor 0.75) | 0.7576 | 0.7585 |
| min inside-image margin | ≥ 1.999 px | ≥ 1.998 px |
| determinism (two full renders, framemd5) | **0 / 2100 differ** | **306 / 2100 differ** |

**Rotate determinism:**
- The differing frames fall in one render chunk (about frames 466–764).
- Mean absolute difference is 1–2.4 levels, max 39. One run is slightly sharper there; both are
  close to the reference warp.
- `will-change: transform` did not fix it, and was reverted.

## Limits of 2D stills (what cannot be done here)

1. **Moving through the room is not possible.** Walking in, dollying or orbiting needs depth:
   near and far objects must move differently. Every 2D move here is a zoom, a crop slide or a
   rotation in place. The depth/parallax route was tried and rejected, because it bent lines.
2. **The start can't be very tight.** The sources are 1920×1280. At 1080p the zoom is capped at
   about 1.33× (0.75 source px per output px), so even the tightest start shows about 56% of the
   frame area. Tighter needs larger originals. As a rough guide, 2× tighter needs about 3840 px
   wide sources.
3. **The tilt and pan are limited by the photo's edges.**
   - Crop mode can only travel about 20% of the width and 16% of the height at the tight zoom.
   - Rotate mode looks physically right but reaches only about 40% of that: aiming toward the
     edges pulls in the stretched wide-angle periphery and exits the photo sooner. It is also
     not deterministic in the current renderer.
4. **Moves look alike across shots.** All four tilt_reveal shots have nearly identical keyframes
   (mirrored for s12). The photos share one size, and the saliency peaks sit in the lower half,
   so K1's height clamps to 0.5 every time. Without room context (objects, walkways), saliency
   is the only aim signal.

## Files

**In this folder** (not committed):
- `v3_crop.mp4`, `v3_rotate.mp4`, `compare_grid.mp4`. The grid shows liked baseline / v2 / v3 crop
  / v3 rotate.
- `v3_crop.plan.json`, `v3_rotate.plan.json` and the `*_plan_notes.json` files.
- `*.framemd5`.

**Copied to the Downing folder:** `virtual_tour_v3_tilt_reveal.mp4` (crop) and
`virtual_tour_v3_compare_grid.mp4`.

**Reproduce:**

    python -m tools.tour_v2.planner --baseline fixtures/downing/baseline_plan.json \
        --saliency fixtures/downing/saliency.json --out-dir out/tour_v3 --v3 --projection crop
    python -m tools.tour_v2.render --plan out/tour_v3/v3_crop.plan.json --images-dir <photos> \
        --out out/tour_v3/v3_crop.mp4 --jpeg-quality default

**Tests:** `tests/test_camera_path.py` has 23 tests, all passing. They include rotate vs crop at
the centre, aim and level horizon, TypeScript parity, unknown projection rejected, and rotate
bounds checked.
