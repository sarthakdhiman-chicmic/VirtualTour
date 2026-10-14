# Tour v4: compound moves, speed cap, cut continuity

Branch `feat/tour-v2`, uncommitted, built on the checkpoint `checkpoint/v3-tilt-reveal` (`c306c44`).
The render is 1920×1080, 30 fps, 2250 frames, 75.05 s, with hard cuts.
Determinism: two renders, 0 / 2250 frames differ (framemd5).

## Review points, and what changed

1. **Speed was too fast at times.**
   - New camera type `compound` (addendum 1.3.2) re-times every path by visual distance (RMS
     motion of the frame corners).
   - Every shot enters and leaves at 1.6 px/frame and stays ≤ 5.0 px/frame. v3 peaked at 7.8.
   - Durations follow each move's distance. A move that would still be too fast is shrunk toward
     its end state (the `shrink` column).
2. **Abrupt zoom in/out at cuts.**
   - Each shot's opening motion is classified (zoom in/out, pan left/right, tilt up/down) and so is
     the previous shot's closing motion.
   - A zoom reversal across a cut is not allowed. Matching pan and tilt directions score higher.
   - Cut speed is the same on both sides, 1.6 px/frame. Short moves are the exception: they enter
     and leave at a constant speed below 1.6 rather than slowing mid-shot.
3. **Always the same direction.**
   - Every move can be mirrored left/right, and moves with a tilt can be flipped up/down.
   - The sequence search rewards alternating left/right, but it rewards carrying a pan through a
     cut more.
   - Tilt-down was available but never chosen: its opening frames (ceilings) score low on saliency.
4. **More effects.** 13 moves are available; 9 are used here.
   - **Ported, same shapes:** `tilt_reveal` (v3), `reveal_sweep` (v2), `tilt_reveal_rotate`
     (v3 rotate).
   - **New:** `swoop`, `tilt_sweep`, `traverse`, `push_pan`, `slide_push`, `pan_reveal`, `crane`,
     `settle_push`, `pull_back`, `rise_reveal`.

## Selection by coverage and room score

**Coverage** is the photo's saliency spread, relative to the tour:

| Category | Spread |
|---|---|
| wide | > median (0.402) |
| medium | > 25th percentile (0.369) |
| tight | below that |

Exterior, bathroom and final shots have their own tables (`SUIT` in `tools/tour_v3/planner.py`).

**Room score:** heroes get +0.75 on feature moves. There is no numeric room score in the worker
plan. The backend's room scores can replace this proxy when the planner is ported.

**Search:** a beam search (width 400) over (move, mirror, flip) per shot. It scores:
- fit for the category
- how salient the opening and closing frames are
- continuity at the cut
- variety: −5 for the previous move again, −2 for the move two back, −1.5 for three reveals or
  three pushes in a row
- at most 3 uses of any move

## Per shot

Speeds are output px/frame, measured on the plan's frames.

| Shot | Room | Category | Move | Direction | Seconds | Frames | Speed first / max / cut | Shrink | Zoom range | Cut continuity |
|---|---|---|---|---|---|---|---|---|---|---|
| s01 | exterior | exterior | crane | left | 4.91 | 0–146 | 1.62 / 3.25 / 1.62 | 1.0 | 1.04–1.16 | – |
| s02 | living_room | wide | reveal_sweep | right / up | 7.77 | 147–379 | 1.63 / 4.70 / 1.63 | 0.8 | 1.044–1.237 | 4.0 |
| s03 | living_room | wide | tilt_reveal | left / up | 7.77 | 380–613 | 1.63 / 4.97 / 1.63 | 1.0 | 1.012–1.32 | 1.5 |
| s04 | living_room | medium | slide_push | right | 4.58 | 614–750 | 1.62 / 3.26 / 1.62 | 1.0 | 1.10–1.26 | 1.5 |
| s05 | kitchen | medium | push_pan | right | 4.63 | 751–889 | 1.62 / 3.24 / 1.62 | 1.0 | 1.05–1.26 | 4.0 |
| s06 | dining_room | wide | swoop | left / up | 7.77 | 890–1122 | 1.63 / 4.85 / 1.63 | 0.65 | 1.012–1.191 | 4.5 |
| s07 | dining_room | tight | pull_back | right | 3.74 | 1123–1234 | 1.63 / 3.26 / 1.63 | 1.0 | 1.012–1.25 | 4.0 |
| s08 | basement | wide | tilt_reveal | left / up | 7.77 | 1235–1467 | 1.63 / 5.00 / 1.63 | 1.0 | 1.012–1.32 | 1.5 |
| s09 | primary_bedroom | tight | slide_push | right | 4.62 | 1468–1606 | 1.62 / 3.23 / 1.62 | 1.0 | 1.10–1.26 | 1.5 |
| s10 | primary_bathroom | bath | settle_push | right | 3.5 | 1607–1711 | 1.61 / 1.97 / 1.61 | 1.0 | 1.03–1.14 | 4.0 |
| s11 | bedroom | tight | push_pan | right | 5.16 | 1712–1866 | 1.62 / 3.24 / 1.62 | 1.0 | 1.05–1.26 | 4.0 |
| s12 | bedroom | wide | swoop | left / up | 7.77 | 1867–2099 | 1.63 / 4.85 / 1.63 | 0.65 | 1.012–1.191 | 4.5 |
| s13 | bedroom | final | pull_back | right | 5.0 | 2100–2249 | 1.61 / 2.33 / 1.61 | 1.0 | 1.012–1.25 | 4.5 |

**Constraints:**
- Inside-image margin ≥ 1.999 px on every frame.
- Pixel density ≥ 0.7576 (floor 0.75).
- No cut reverses the zoom.

## Open questions and limits

- **Total is 75 s, not 70.** Sizing durations by distance wanted 89.4 s, which was capped at the
  +5 s tolerance. Cutting to 70 would shrink the reveals further.
- **The two swoops are shrunk to 65%** to stay under 5 px/frame. A 6.0 cap, or 8 s shots, would
  keep them full size.
- **`tilt_reveal_rotate` was not chosen.** Continuity favoured other moves in the medium and tight
  rooms. It is also still not deterministic (see `out/tour_v3/report.md`), so it is best kept
  rare until that is fixed.
- **Shots s09–s11 all pan right.** The search values carrying a pan through a cut above
  alternating. One weight changes that (`continuity` lateral +1.5 vs the −0.6 same-direction
  penalty).
- **All 2D-still limits from v3 still apply:** no moving through the room, about 1.33× maximum
  zoom on 1920 px sources, and saliency as the only aim signal.

## Files

- `v4_compound.plan.json`, `v4_plan_notes.json` (every decision, entry and exit vectors, speeds),
  `v4_compound*.framemd5`.
- `v4_compound.mp4` and `compare_v3_v4.mp4` (not committed). They are copied to the Downing folder
  as `virtual_tour_v4_compound.mp4` and `virtual_tour_v4_vs_v3_compare.mp4`.
- **Code:**
  - `modules/camera_path.py`: `CompoundPath`, `validate_compound`
  - `remotion/src/math.ts`: `CompoundPath`, parity-tested
  - `modules/plan_io.py`, `modules/remotion_render.py`, `remotion/src/KenBurnsShot.tsx`
  - `tools/tour_v3/planner.py`
- **Tests:** 62 passed, 4 skipped (the full worker suite).

**Reproduce:**

    python -m tools.tour_v3.planner --baseline fixtures/downing/baseline_plan.json \
        --saliency fixtures/downing/saliency.json --images-dir <photos> --out-dir out/tour_v4
    python -m tools.tour_v2.render --plan out/tour_v4/v4_compound.plan.json --images-dir <photos> \
        --out out/tour_v4/v4_compound.mp4 --jpeg-quality default
