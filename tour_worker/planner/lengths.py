"""Shot lengths by shot type and pacing - one table for photo tours and 3D tours (client standards,
2026-11-13). Viewers grasp what a room is almost at once, but judging its layout, size and light takes
a few seconds; a detail (a fixture, a finish) or a mood reads in under two.

  type     what it is                                         Normal     Fast        Slow
  room     a room shot, its layout visible                    3.5-5 s    2.5-4 s     4.5-6.5 s
  hero     a hero room's shots (best living room, kitchen,    5-8 s      4-6.5 s     6.5-9 s
           primary bedroom), and the opening / closing shots
  detail   a close-up: fixtures, hardware, finishes, on the   1.5-2.5 s  1.2-1.8 s   2-3 s
           music's drops

Where in its range a shot sits comes from what it shows (richness 0-1: the 2D planner's scene
context for a photo, the labelled objects in view for a 3D shot). The beat sync then EXTENDS room
and hero shots to the next beat - never shortens them; detail shots are the flexible ones
(SYNC_BOUNDS)."""
from __future__ import annotations

TYPES = ("room", "hero", "detail")

LENGTHS = {
    "normal": {"room": (3.5, 5.0), "hero": (5.0, 8.0), "detail": (1.5, 2.5)},
    "fast":   {"room": (2.5, 4.0), "hero": (4.0, 6.5), "detail": (1.2, 1.8)},
    "slow":   {"room": (4.5, 6.5), "hero": (6.5, 9.0), "detail": (2.0, 3.0)},
}
# a room shot is never shorter than this (the client's floor: at least 3 s, ~2.5 s fast)
FLOOR = {"normal": {"room": 3.0, "hero": 5.0, "detail": 1.2},
         "fast": {"room": 2.5, "hero": 4.0, "detail": 1.0},
         "slow": {"room": 4.5, "hero": 6.5, "detail": 1.8}}
# the beat sync's window, as a share of the planned length: room and hero shots only grow (to the
# next beat, or the one after when a strong moment is there); details flex both ways
SYNC_BOUNDS = {"room": (1.0, 1.25), "hero": (1.0, 1.2), "detail": (0.7, 1.6)}
WIDE_BOUNDS = {"room": (1.0, 1.4), "hero": (1.0, 1.35), "detail": (0.6, 1.8)}   # the hooks fallback
STRETCH = 1.25                         # a shot stretched for a short tour stops at range max x this


def pace_of(pacing: str | None) -> str:
    return pacing if pacing in LENGTHS else "normal"


def length_range(kind: str, pacing: str | None) -> tuple[float, float]:
    return LENGTHS[pace_of(pacing)][kind if kind in TYPES else "room"]


def shot_length(kind: str, pacing: str | None, richness: float = 0.5) -> float:
    """A shot's planned length: richer content sits higher in its type's range."""
    lo, hi = length_range(kind, pacing)
    r = min(1.0, max(0.0, float(richness)))
    return round(lo + r * (hi - lo), 2)


def floor_s(kind: str, pacing: str | None) -> float:
    return FLOOR[pace_of(pacing)][kind if kind in TYPES else "room"]


def ceiling_s(kind: str, pacing: str | None) -> float:
    return round(length_range(kind, pacing)[1] * STRETCH, 2)


def sync_kind(role: str | None) -> str:
    """The sync's type for a role as plans and older callers name it: "hero" stays, "detail"
    from the photo planner's past (a room's other photo) is a room shot unless marked "close"."""
    if role in ("hero", "room"):
        return role
    if role in ("detail_close", "close"):
        return "detail"
    return "room"
