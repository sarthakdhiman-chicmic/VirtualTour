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


KITCHEN = ("stove", "oven", "sink", "refrigerator", "dishwasher", "kitchen island", "countertop", "hood")
BATH = ("toilet", "bathtub", "shower")
LIVING = ("sofa", "television receiver", "fireplace")
OFFICE = ("desk",)


def room_types(space, objects: list[dict], points: dict[str, np.ndarray]) -> dict[str, str]:
    """{room key: room type} from the labelled objects in each room (module docstring)."""
    from tour_worker.planner3d.space import CELL
    held: dict[str, dict[str, list[dict]]] = {k: {} for k in points}
    for o in objects:
        r = o.get("room")
        if r in held:
            held[r].setdefault(o["class"], []).append(o)

    def size(o):
        f = o.get("footprint") or {}
        return float(f.get("long", 0.0)) * float(f.get("short", 0.0))

    out: dict[str, str] = {}
    beds = {k: max((size(o) for o in h.get("bed", [])), default=0.0) for k, h in held.items()}
    primary = max((k for k in beds if beds[k] >= 1.5), key=lambda k: beds[k], default=None)
    living = []
    for k, h in held.items():
        has = lambda names: any(n in h for n in names)          # noqa: E731
        if has(BATH):
            out[k] = "bathroom"
        elif sum(n in h for n in KITCHEN) >= 2 or "kitchen island" in h or "stove" in h:
            out[k] = "kitchen"
        elif beds[k] >= 1.5:
            out[k] = "primary_bedroom" if k == primary else "bedroom"
        elif has(LIVING):
            living.append(k)
            out[k] = "living_room"
        elif "table" in h and len(h.get("chair", [])) >= 3:
            out[k] = "dining_room"
        elif has(OFFICE):
            out[k] = "home_office"
        else:
            mask = space.room_of == space.room_keys.index(k)
            ys, xs = np.nonzero(mask)
            narrow = float(space.dt[mask].max()) if mask.any() else 0.0
            long_side = max(np.ptp(ys), np.ptp(xs)) * CELL if mask.any() else 0.0
            out[k] = "hallway" if narrow <= 0.75 and long_side >= 3.0 else "other_room"
    # one living room: the largest; the others are family rooms
    if len(living) > 1:
        areas = {k: float((space.room_of == space.room_keys.index(k)).sum()) for k in living}
        best = max(living, key=lambda k: areas[k])
        for k in living:
            if k != best:
                out[k] = "family_room"
    return out
