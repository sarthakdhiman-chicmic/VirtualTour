# Tour v7: v3 rotate reveals as the base, always ending wider, plus a spiral and a subtle crane

Uncommitted on `feat/tour-v2`. Planner: `tools/tour_v6/planner.py` (outputs v7).
Checkpoints: `checkpoint/v3-tilt-reveal`, `checkpoint/v4-compound`. The v5 work, including the
playback fix, is not committed yet.

The render is 1920×1080, 30 fps, 75.0 s, with hard cuts.
- Determinism: two renders are byte-identical.
- Playback (`modules/encode.py`): 9.6 Mbps mean, 13.9 Mbps peak in any second, buffer never below
  37.9%, a keyframe every ≤ 2 s, Level 4.1, faststart, 90 MB.

## Review points

- **The v3 rotate tilt reveal was best; the v4 effects were too much.**
  - The v4 compound moves (swoop, push_pan, slide_push, crane-push and so on) are gone.
  - The base is the v3 rotate tilt reveal: the v3 keyframe path and time profile, rotate
    projection, pre-warped.
- **Always zoom out.** Every shot starts on part of the room and ends on all of it (zoom
  1.10–1.21 → 1.012–1.035). There are no push-ins.
- **Not every shot a rotate reveal; a subtle crane in rooms with more coverage.**
  - `crane` is a crop: a gentle rise from low to eye level (16% of the frame height), zoom
    1.10 → 1.035, a slight drift, max 2.0 px/frame.
  - A sliding window reads as the camera rising, unlike a tilt, where the verticals lean.
  - It is allowed in wide rooms and exteriors only. It is used in s03 and s08.
- **A "circlish" motion.**
  - A true orbit needs depth: near and far things moving differently. That is the parallax route
    that bent lines, so it is not possible with a flat photo.
  - `arc` turns the camera in place along a spiral: the aim circles once (left → top → right →
    bottom) while the circle shrinks into the centre and the camera pulls back to the whole room.
  - It is a compound camera with overlapping phases and distance- and turn-aware timing. A keyframe
    path turned 8–11°/frame on a circle; this one turns at most 3.07°/frame.
  - Circle size is ±87 × ±131 source px (the rotate camera's reach). It is used in s02 and s05,
    circling opposite ways.
- **Variety, kept subtle.**
  - Travel direction alternates on every cut, and the same variant never plays twice in a row.
  - Repeating a variant in the same direction is penalised, and so is using any variant a fourth time.
  - Every shot stays ≤ 3.3 px/frame (about the v3 rotate pace). Short shots get a smaller reveal
    rather than a faster one.

## Per shot

| Shot | Room | Category | Variant | Projection | Travel | Seconds | Zoom | Max px/frame | Max turn °/frame | Reveal scale |
|---|---|---|---|---|---|---|---|---|---|---|
| s01 | exterior | exterior | tilt_up | rotate | right | 5.09 | 1.154→1.012 | 3.19 | 3.53 | 0.75 |
| s02 | living_room | wide | arc (spiral) | rotate | left | 7.41 | 1.160→1.012 | 2.52 | 3.07 | 1.0 |
| s03 | living_room | wide | crane | crop | right | 6.48 | 1.100→1.035 | 2.01 | 0.01 | 1.0 |
| s04 | living_room | medium | pan | rotate | left | 5.56 | 1.185→1.012 | 3.18 | 3.00 | 0.85 |
| s05 | kitchen | medium | arc (spiral) | rotate | right | 7.41 | 1.160→1.012 | 2.52 | 3.07 | 1.0 |
| s06 | dining_room | wide | tilt_up | rotate | left | 6.48 | 1.206→1.012 | 3.18 | 2.64 | 1.0 |
| s07 | dining_room | tight | pull_back | rotate | right | 4.63 | 1.175→1.020 | 1.74 | 0.04 | 1.0 |
| s08 | basement | wide | crane | crop | left | 6.48 | 1.100→1.035 | 2.01 | 0.01 | 1.0 |
| s09 | primary_bedroom | tight | tilt_down | rotate | right | 4.63 | 1.154→1.012 | 3.22 | 3.59 | 0.75 |
| s10 | primary_bathroom | bath | pull_back | rotate | left | 4.17 | 1.175→1.020 | 2.51 | 0.10 | 1.0 |
| s11 | bedroom | tight | pan | rotate | right | 4.63 | 1.152→1.012 | 3.07 | 3.52 | 0.7 |
| s12 | bedroom | wide | tilt_up | rotate | left | 6.48 | 1.206→1.012 | 3.18 | 2.64 | 1.0 |
| s13 | bedroom | final | pull_back | rotate | right | 5.55 | 1.175→1.020 | 2.17 | 0.07 | 1.0 |

## Open points

- **Every cut goes from a wide end to a tighter start** (scale jump about 1.01 → 1.10–1.21). That
  follows from "always zoom out" with hard cuts.
- **The spiral and the rotate reveals are gentle:** rotate reach is about 40% of a crop's on
  1920 px photos.
- **The tour is 75 s.** The baseline is 70 s; slower reveals need the time.

## Also in the Downing folder

- `virtual_tour_v6_all_rotate_reveals.mp4`: the intermediate version with every shot a rotate
  reveal and no arc or crane. Byte-identical across two renders.

**Reproduce:**

    python -m tools.tour_v6.planner --baseline fixtures/downing/baseline_plan.json \
        --saliency fixtures/downing/saliency.json --images-dir <photos> --out-dir out/tour_v7
    python -m tools.tour_v2.render --plan out/tour_v7/v7.plan.json --images-dir <photos> \
        --out out/tour_v7/v7.mp4 --jpeg-quality default
