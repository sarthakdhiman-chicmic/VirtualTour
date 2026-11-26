# ADR 0005: No drone clips in this delivery

- **Status:** Proposed, 2026-11-26. Recorded against client review items 17 and 18 (drone clips at
  exterior shots, an exterior opening for 3D tours); it ends when drone footage is provided.
- **Date:** 2026-11-26
- **Scope:** 3D tours (`planner3d/plan.py` drone_clips, `render/clip.py`).

## Context

Item 17 places drone-style moves at the exterior approach, with drone clips at exterior shots and a
smooth transition to the interior. Drone clips are supported (`drone_clips` in `tour_plan`: entry
clips before the interior, rear clips after).

No drone footage has been provided for the listings in this delivery: the green-house and Home Scan
scans come without drone clips.

## Decision

No drone clip is rendered in this delivery's tours, because there is none for these properties. The
support stays: a request with `drone_clips` plays them (entry clips before the interior, rear clips
after); we will render a tour with them as soon as footage is provided.

## Consequences

3D tours open inside (the green-house scan has no exterior). Item 18's exterior opening is not
shown for 3D tours in this delivery.
