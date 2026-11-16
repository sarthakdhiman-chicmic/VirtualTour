"""Rooms found in a splat scene with no input (user 2026-11-16: "give no inputs to the planner"):
where the rooms are from the floor plan alone, and what each is from the labels voted onto it.

  where   room centres are the most open spots of the free floor (clearance maxima, at least
          SEED_CLEAR from anything and SEED_GAP apart); Space.set_rooms grows the rooms from them
          (walking cost dear through narrow passages, so rooms meet at doorways). Two neighbours
          whose opening is nearly as wide as their open floor (MERGE_SADDLE) are one room - one
          big room gets one centre, two rooms joined by a door keep two. Fragments under MIN_ROOM_M2
          are dropped.
  what    from the labelled objects in each room (scene/semantics.py): a toilet or bath is a
          bathroom, a hob, sink or fridge a kitchen, a bed a bedroom (the biggest bed the primary
          bedroom), a sofa or television a living room (more than one: family rooms), a table with
          chairs a dining room, a desk an office; a long narrow room with none of these a hallway;
          else "other_room".
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

SEED_CLEAR = 0.6         # m: a room centre is at least this far from anything
SEED_GAP = 2.0           # m between room centres
MERGE_SADDLE = 0.7       # neighbours merge when their widest shared opening is this share of their open floor
MIN_ROOM_M2 = 3.0


def room_seeds(space) -> list[np.ndarray]:
    """Room centres: clearance maxima (planner xy), the most open first."""
    from tour_worker.planner3d.space import CELL
    dt = space.dt
    peak = ndi.maximum_filter(dt, size=max(3, int(1.6 / CELL))) == dt
    ys, xs = np.nonzero(peak & (dt >= SEED_CLEAR) & space.free)
    out: list[np.ndarray] = []
    for k in np.argsort(-dt[ys, xs], kind="stable"):
        p = np.array([space.lo[0] + (xs[k] + .5) * CELL, space.lo[1] + (ys[k] + .5) * CELL])
        if all(np.linalg.norm(p - q) >= SEED_GAP for q in out):
            out.append(p)
    return out


def _saddles(space, n: int) -> dict[tuple[int, int], float]:
    """The widest opening (clearance, m) on the boundary between each pair of neighbouring rooms."""
    lab, dt = space.room_of, space.dt
    out: dict[tuple[int, int], float] = {}
    for a, b, da, db in ((lab[:, :-1], lab[:, 1:], dt[:, :-1], dt[:, 1:]), (lab[:-1], lab[1:], dt[:-1], dt[1:])):
        m = (a >= 0) & (b >= 0) & (a != b)
        for i, j, d in zip(a[m], b[m], np.minimum(da[m], db[m])):
            key = (int(min(i, j)), int(max(i, j)))
            if d > out.get(key, 0.0):
                out[key] = float(d)
    return out


def discover_rooms(space) -> dict[str, np.ndarray]:
    """{room key: centre (planner xy)} with the rooms set on space (Space.set_rooms)."""
    from tour_worker.planner3d.space import CELL
    seeds = room_seeds(space)
    if not seeds:
        return {}
    while True:
        pts = {f"room_{k + 1}": p for k, p in enumerate(seeds)}
        space.set_rooms(pts)
        keys = space.room_keys
        clear = {i: float(space.dt[space.cell(pts[k])]) for i, k in enumerate(keys)}
        area = {i: float((space.room_of == i).sum()) * CELL * CELL for i in range(len(keys))}
        sad = _saddles(space, len(keys))
        merge = [(d / min(clear[i], clear[j]), i, j) for (i, j), d in sad.items()
                 if d >= MERGE_SADDLE * min(clear[i], clear[j])]
        small = [i for i in range(len(keys)) if area[i] < MIN_ROOM_M2]
        if merge:                          # the most open opening first: drop the less open centre
            _, i, j = max(merge)
            drop = i if clear[i] < clear[j] else j
        elif small:
            drop = min(small, key=lambda i: area[i])
        else:
            break
        seeds = [pts[k] for n, k in enumerate(keys) if n != drop]
        if not seeds:
            return {}
    # stable keys, most open room first
    return {k: pts[k] for k in keys}


KITCHEN = ("stove", "oven", "sink", "refrigerator", "dishwasher", "kitchen island", "countertop", "hood", "microwave")
BATH = ("toilet", "bathtub", "shower")
LIVING = ("sofa", "television receiver", "fireplace")
OFFICE = ("desk", "computer", "monitor", "crt screen", "screen")
BED_MIN_M2 = 1.2         # this much bed (all its pieces) makes a bedroom - a stray bath fixture does not
DINING_TABLE_M2 = 1.2    # a dining table at least this big (a desk is smaller), with chairs
CAPTURE_MIN = 0.75       # rooms whose survey views the scan covers less than this are not toured


def _area(o) -> float:
    f = o.get("footprint") or {}
    return float(f.get("long", 0.0)) * float(f.get("short", 0.0))


def _evidence(held: dict[str, list[dict]]) -> dict[str, float]:
    a = lambda names: sum(_area(o) for n in names for o in held.get(n, []))      # noqa: E731
    return {"bed": a(("bed",)), "bath": float(sum(len(held.get(n, [])) for n in BATH)),
            "kitchen": a(KITCHEN), "kitchen_kinds": float(sum(n in held for n in KITCHEN)),
            "living": a(("sofa",)) + 0.6 * sum(n in held for n in ("television receiver", "fireplace")),
            "table": max((_area(o) for o in held.get("table", [])), default=0.0),
            "chairs": float(len(held.get("chair", []))), "office": float(sum(n in held for n in OFFICE))}


def _type_of(space, key: str, ev: dict[str, float]) -> str:
    """One room's type from its evidence (module docstring): a bed dominates; then bath fixtures;
    then kitchen vs living by the larger area of their own furniture; a dining table; an office."""
    from tour_worker.planner3d.space import CELL
    if ev["bed"] >= BED_MIN_M2:
        return "bedroom"
    if ev["bath"] >= 1 and ev["bed"] < 0.5:
        return "bathroom"
    kitchen = ev["kitchen"] if ev["kitchen_kinds"] >= 2 else 0.0
    living = ev["living"] if ev["living"] >= 0.8 else 0.0
    if kitchen or living:
        return "kitchen" if kitchen >= living else "living_room"
    if ev["table"] >= DINING_TABLE_M2 and ev["chairs"] >= 3:
        return "dining_room"
    if ev["office"]:                       # a desk, a computer or a monitor
        return "home_office"
    # a small table with a chair (the labels often call a desk a table) is an office only in a room
    # of its own: in an open space it is a corner of its neighbour (the Home Scan's family room end,
    # 2026-11-16) - "other_room", which merges into it
    if ev["table"] > 0 and ev["chairs"] >= 1 and space.spaces.get(key, {key}) == {key}:
        return "home_office"
    mask = space.room_of == space.room_keys.index(key)
    if not mask.any():
        return "other_room"
    ys, xs = np.nonzero(mask)
    narrow = float(space.dt[mask].max())
    long_side = max(np.ptp(ys), np.ptp(xs)) * CELL
    return "hallway" if narrow <= 0.75 and long_side >= 3.0 else "other_room"


def _held(objects, keys) -> dict[str, dict[str, list[dict]]]:
    held: dict[str, dict[str, list[dict]]] = {k: {} for k in keys}
    for o in objects:
        if o.get("room") in held:
            held[o["room"]].setdefault(o["class"], []).append(o)
    return held


def room_types(space, objects: list[dict], points: dict[str, np.ndarray]) -> dict[str, str]:
    """{room key: room type} from the labelled objects in each room (module docstring): the primary
    bedroom is the largest bedroom by floor and bed; more than one living room: the largest is the
    living room, the others family rooms."""
    from tour_worker.planner3d.space import CELL
    held = _held(objects, points)
    ev = {k: _evidence(h) for k, h in held.items()}
    out = {k: _type_of(space, k, ev[k]) for k in points}
    area = {k: float((space.room_of == space.room_keys.index(k)).sum()) * CELL * CELL for k in points}
    beds = [k for k in out if out[k] == "bedroom"]
    if beds:
        out[max(beds, key=lambda k: area[k] + 2.0 * ev[k]["bed"])] = "primary_bedroom"
    living = [k for k in out if out[k] == "living_room"]
    if len(living) > 1:
        best = max(living, key=lambda k: area[k])
        for k in living:
            if k != best:
                out[k] = "family_room"
    return out


MERGE_SAME = {"living_room", "family_room", "dining_room", "home_office", "bedroom", "primary_bedroom",
              "kitchen", "hallway", "other_room", "bathroom"}


def finalize_rooms(space, points: dict[str, np.ndarray], objects: list[dict],
                   capture: dict[str, float] | None = None):
    """The rooms as tour_plan takes them, from the found rooms and the scene's labels:
      - neighbours in one open space merge when they are the same kind of room, or one is an
        "other_room" (one family room split in three is one room; a kitchen beside a living room
        stays two), the more open centre kept, then typed again;
      - a room the scan covers poorly (capture < CAPTURE_MIN of its survey views) is not toured.
    Returns ([(key, type, centre, toured)], warnings); objects' rooms are updated to the merges."""
    keys = list(points)
    merged: dict[str, str] = {}
    while True:
        space.set_rooms({k: points[k] for k in keys})
        for o in objects:
            if o.get("room") in merged:
                o["room"] = merged[o["room"]]
        types = room_types(space, objects, {k: points[k] for k in keys})
        sad = _saddles(space, len(keys))
        clear = {i: float(space.dt[space.cell(points[k])]) for i, k in enumerate(keys)}
        pairs = []
        for (i, j), d in sad.items():
            a, b = keys[i], keys[j]
            if b not in space.spaces.get(a, {a}):
                continue                                   # a doorway between them: two rooms
            ta, tb = types[a], types[b]
            same = ta == tb or {ta, tb} <= {"living_room", "family_room"}
            if same or "other_room" in (ta, tb):
                pairs.append((d, i, j))
        if not pairs:
            break
        _, i, j = max(pairs)
        keep, drop = (keys[i], keys[j]) if clear[i] >= clear[j] else (keys[j], keys[i])
        if types[drop] != "other_room" and types[keep] == "other_room":
            keep, drop = drop, keep
        merged = {**{k: (keep if v == drop else v) for k, v in merged.items()}, drop: keep}
        keys = [k for k in keys if k != drop]
    warnings = [f"rooms_merged:{d}->{k}" for d, k in sorted(merged.items())]
    out = []
    for k in keys:
        c = (capture or {}).get(k)
        if c is None:                                      # a merged room: its parts' worst coverage
            parts = [d for d, kk in merged.items() if kk == k] + [k]
            vals = [(capture or {}).get(p) for p in parts if (capture or {}).get(p) is not None]
            c = min(vals) if vals else None
        toured = c is None or c >= CAPTURE_MIN
        if types[k] == "other_room" and len(keys) > 1:
            toured = False                                 # nothing a viewer would recognise
        if not toured:
            warnings.append(f"room_not_toured:{k}:" + ("other_room" if types[k] == "other_room" else f"capture_{c:.2f}"))
        out.append((k, types[k], points[k], toured))
    return out, warnings
