# ADR 0004: Lit fires are held still

- **Status:** Proposed, 2026-11-26 (item 15's own fallback; recorded for review item 23).
- **Date:** 2026-11-26
- **Scope:** photo tours and 3D tours (fire motion).

## Context

Item 15: a lit fire must read as real, or be held still. The fire effect moved little more than a
glow patch. The current segmentation (ADR 0001) has no fireplace class, and the lit-fire detection
only runs inside fireplace masks.

## Decision

No fire is animated: photo tours plan no fire region, 3D scenes vote no flames. The effect stays in
the code (`motion/fire.py`, `planner/config.py`: `fire` is not mapped).

## Consequences

Fires show as in the photo / scan. Animating fire needs a fireplace source (a class or a detector)
and an effect that reads as real.
