# ADR 0003: Pool-floor fittings ripple with the water; a partly detected spa is held still

- **Status:** Proposed, 2026-11-26. A departure from client review item 13; needs the client's
  written approval (review item 23).
- **Date:** 2026-11-26
- **Scope:** photo tours, water motion (`motion/water.py`, `motion/loader.py`).

## Context

Item 13: tile, coping, walls and fittings fully still; the whole visible water surface moving as one.
With the current segmentation (ADR 0001) the water masks stop at the waterline: tile, coping, the
spa wall and the waterline band stay still, and the pool surface moves as one (water_guard holds
only a thin band along the edge). Two points remain:

1. **Pool-floor fittings.** Lights and drains lie inside the water mask, so they ripple with it.
   Three detectors were tried on the Turkey Trot pool (bright compact blobs, ellipse fits, circle
   finding on a perspective-corrected image); each either missed most fittings or caught the bright
   tree reflections and the mosaic floor.
2. **The spa.** Its water is only partly detected (about half the basin). Growing the mask over the
   rest spilled onto the tile wall and coping, which item 13 forbids.

## Decision

- Pool-floor fittings ripple with the water, as submerged fittings do through moving water.
- A water region too faint to read is held still as a whole (no boost for water), so a partly
  detected spa never moves along an irregular line.

## Consequences

- Tile, coping, walls and the waterline stay still; the visible pool surface moves as one.
- The fittings move with the ripple; the spa is still. A detector for fittings (a model that knows
  pool lights and drains) can replace this later.
