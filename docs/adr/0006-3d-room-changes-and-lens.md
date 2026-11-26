# ADR 0006: 3D room changes fly through doors; cuts at tight doors and small rooms

- **Status:** Proposed, 2026-11-26. Recorded against client review items 5, 6 and 17 (review item 23).
- **Date:** 2026-11-26
- **Scope:** 3D tours (`planner3d/plan.py`, `planner3d/moves.py`).

## Context

Item 6 asks that each room's last shot end moving toward the next doorway and the next start inside
the new room on the same heading and speed. Item 5 holds direction within 45 degrees at every cut;
item 17 keeps anything near the lens to a sliver.

## Decision

- **Room changes are one continuous move where the camera can pass.** The planner first plans a
  single flight from inside the room being left, through every door on the walking route, into the
  next room - narrow doors included. Only where that flight fails its checks does it cut at the door,
  matched: the shot before ends heading for the door, the next starts just past it on the same heading.
- **Door frames near the lens.** A flight through a door may pass the frame at the edge of the
  picture (up to 25% of the frame within 0.45 m of the lens); room shots keep the general limit
  (25% of the frame within 1 m anywhere, 20% of its middle within the move's near distance). 15%
  was tried and made the Home Scan skip three rooms.
- **Small rooms.** A bathroom or hallway often has only a back-out with room to breathe; a back-out
  cannot be played in reverse (it would end facing a wall), so a cut into or out of one may reverse
  direction. Such cuts are listed in the continuity report.

## Consequences

- Room changes read as one move through the home; the matched cut is the fallback.
- A few 3D cuts stay outside item 5's limits (green house: 9-11 of 13 within, local plans) - in very
  small rooms, reported per cut.
- The near-lens limit is not a "sliver" for flights through doors.
