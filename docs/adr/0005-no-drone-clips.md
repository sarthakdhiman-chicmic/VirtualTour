# ADR 0005: No drone clips in this delivery

- **Status:** Proposed by the product owner, 2026-11-26. A departure from client review item 17
  (drone clips at exterior shots); needs the client's written approval (review item 23).
- **Date:** 2026-11-26
- **Scope:** 3D tours (`planner3d/plan.py` drone_clips, `render/clip.py`).

## Context

Item 17 places drone-style moves at the exterior approach, with drone clips at exterior shots and a
smooth transition to the interior. Drone clips are supported (`drone_clips` in `tour_plan`: entry
clips before the interior, rear clips after).

[REASON - to be written by the product owner.]

## Decision

No drone clip is rendered in this delivery's tours. The support stays: a request with `drone_clips`
still plays them.

## Consequences

3D tours open inside (the green-house scan has no exterior). Item 18's exterior opening is not
shown for 3D tours in this delivery.
