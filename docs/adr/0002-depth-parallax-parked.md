# ADR 0002: Depth parallax on photo tours is parked for this delivery

- **Status:** Proposed by the product owner, 2026-11-26. A departure from client review item 10;
  needs the client's written approval (review item 23).
- **Date:** 2026-11-26
- **Scope:** photo tours (`tour_worker/parallax/`, `planner/config.py`, `jobs/render.py`).

## Context

Item 10 asks for depth on every photo-tour hero shot: Depth Anything V2 Small depth with the
existing segmentation, structure held as rigid planes, objects separated at depth edges, capped
travel with edge-extension fill. It was built that way (`tour_worker/parallax/`, 2026-11-23 to
2026-11-25) and rendered on two listings at 16, 24 and 32 px of travel.

On real listings the segmentation often splits one continuous surface into several labelled
regions, or labels a large region wrongly (a wall as "table", ceiling over the walls). Each region
becomes a layer at its own depth, and the layers tear apart along irregular boundaries as the camera
moves - on counters, rails, door frames and mirrors. Soft seams that bend instead of tearing,
upright object planes, edges snapped to the photo's colour edges and layers that follow the photo's
own depth reduced it a great deal (commits 2977d7f, cf74bf3), but not to a standard we would ship.

## Decision

Parallax stays in the worker, off: the planner plans no parallax (`parallax: False`) and the render
skips it - and its depth / segmentation prep - for any plan that asks for it, unless the worker runs
with `PARALLAX_ENABLED=1`. Photo tours render as before (one camera transform per frame) until the
tearing is fixed.

## Alternatives considered

- **Ship at a low travel (8-16 px).** The tears are smaller but still visible on review.
- **Parallax only on shots with clean segmentation.** No reliable automatic test for "clean" yet.

## Consequences

- Photo tours do not yet meet item 10; they stay at the §6.5 Fallback 1 level for depth.
- The work is kept (layers, seams, bent layers, tests) and resumes after this delivery; frames of
  the failures and of each fix go with the delivery.
