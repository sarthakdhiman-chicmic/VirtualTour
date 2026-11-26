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
          bedroom), a sofa or television a living room (never a family room), a table with
          chairs a dining room, a desk an office; a long narrow room with none of these a hallway;
          else "other_room".
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

SEED_CLEAR = 0.6         # m: a room centre is at least this far from anything
SEED_GAP = 2.0           # m between room centres
MERGE_SADDLE = 0.92      # neighbours merge when their widest opening is nearly as wide as their open floor (one room): an open plan's zones merge later, by type (finalize_rooms)
MIN_ROOM_M2 = 3.0


def room_seeds(space) -> list[np.ndarray]:
    """Room centres: clearance maxima (planner xy), the most open first."""
    from tour_worker.planner3d.space import CELL
    dt = _clear(space)
    peak = ndi.maximum_filter(dt, size=max(3, int(1.6 / CELL))) == dt
    # a centre on the room's floor (under its bed is fine): set_rooms snaps it to free floor
    near_free = ndi.binary_dilation(space.free, iterations=int(1.5 / CELL))
    ys, xs = np.nonzero(peak & (dt >= SEED_CLEAR) & near_free)
    out: list[np.ndarray] = []
    for k in np.argsort(-dt[ys, xs], kind="stable"):
        p = np.array([space.lo[0] + (xs[k] + .5) * CELL, space.lo[1] + (ys[k] + .5) * CELL])
        if all(np.linalg.norm(p - q) >= SEED_GAP for q in out):
            out.append(p)
    return out


def _clear(space) -> np.ndarray:
    """Clearance to the walls (Space.room_dt: furniture is not a wall), else to anything."""
    return np.maximum(getattr(space, "room_dt", space.dt), space.dt)


def _saddles(space, n: int) -> dict[tuple[int, int], float]:
    """The widest opening (clearance to the walls, m) on the boundary between each pair of
    neighbouring rooms."""
    lab, dt = space.room_of, _clear(space)
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
        clear = {i: float(_clear(space)[space.cell(pts[k])]) for i, k in enumerate(keys)}
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


KITCHEN = ("stove", "oven", "sink", "refrigerator", "dishwasher", "kitchen island", "countertop", "counter", "hood",
           "microwave", "toaster")
BATH = ("toilet", "bathtub", "shower")
LIVING = ("sofa", "television receiver", "fireplace")
# a desk, a computer or a monitor - not "screen" / "crt screen": the labels call a wall TV that
# (user 2026-11-16: a family room's fireplace corner became a "Home Office" from its TV)
OFFICE = ("desk", "computer", "monitor", "keyboard", "mouse")
# what the green house's labels showed in the rooms the user marked (2026-11-17): bathrooms with no
# toilet / bath / shower labelled - a sink, a towel, a mirror; a study with no desk - books, shelves,
# a lamp by a table and chairs; a bed labelled "sofa"; a hall's built-ins labelled "refrigerator"
BATH_SOFT = ("sink", "towel", "mirror")
STUDY = ("book", "bookcase", "shelf", "lamp")
COOKING = ("stove", "oven", "hood", "dishwasher", "microwave", "toaster", "kitchen island")   # (a countertop may be a vanity)
UTILITY_SIGNS = ("washer",)    # a washing machine: laundry, not toured (the green house's balcony nook)
LIVING_APART = 4.0       # m: two living rooms this far apart are two; nearer, one seen from two sides
BED_CLEAR = 1.15         # one bed this much bigger than the next makes its room the primary
BED_CLOSED_M2 = 0.8      # in a room behind a door, this much bed is enough (labels often get only part of it)
SMALL_ROOM_M2 = 7.0      # a closed room this small with a sink, towel or mirror is a bathroom
BED_MIN_M2 = 1.2         # this much bed (all its pieces) makes a bedroom - a stray bath fixture does not
DINING_TABLE_M2 = 1.2    # a dining table at least this big (a desk is smaller), with chairs
OPEN_WIDE = 1.3          # m: an opening this wide joins two rooms into one open plan (a door is ~0.8-1.0)
HALL_WIDE = 1.8          # m wall to wall: a long room this narrow (with nothing else in it) is a hallway
CAPTURE_MIN = 0.75       # rooms whose survey views the scan covers less than this are not toured


def _area(o) -> float:
    f = o.get("footprint") or {}
    return float(f.get("long", 0.0)) * float(f.get("short", 0.0))


def _evidence(held: dict[str, list[dict]]) -> dict[str, float]:
    a = lambda names: sum(_area(o) for n in names for o in held.get(n, []))      # noqa: E731
    return {"bed": a(("bed",)), "bath": float(sum(len(held.get(n, [])) for n in BATH)),
            "kitchen": a(KITCHEN), "kitchen_kinds": float(sum(n in held for n in KITCHEN)),
            "living": a(("sofa",)) + 0.5 * a(("armchair",)) + 0.6 * sum(n in held for n in ("television receiver", "fireplace")),
            "table": max((_area(o) for o in held.get("table", [])), default=0.0),
            "chairs": float(len(held.get("chair", []))), "office": float(sum(n in held for n in OFFICE)),
            "bath_soft": float(sum(len(held.get(n, [])) for n in BATH_SOFT)),
            "bath_soft_max": max((_area(o) for n in BATH_SOFT for o in held.get(n, [])), default=0.0),
            "study": float(sum(len(held.get(n, [])) for n in STUDY)),
            "cooking": float(sum(n in held for n in COOKING)),
            "sofa": a(("sofa",)), "tv": float(sum(n in held for n in ("television receiver", "fireplace"))),
            "utility": float(sum(len(held.get(n, [])) for n in UTILITY_SIGNS))}


def _type_of(space, key: str, ev: dict[str, float]) -> str:
    """One room's type from its evidence (module docstring): a bed dominates; then bath fixtures;
    then kitchen vs living by the larger area of their own furniture; a dining table; an office."""
    from tour_worker.planner3d.space import CELL
    mask_ = space.room_of == space.room_keys.index(key) if key in space.room_keys else None
    area_ = float(mask_.sum()) * CELL * CELL if mask_ is not None else 0.0
    closed = not _open_neighbours(space, key)
    # a "bed" among the seating of a room open onto the rest of the home is a sofa or daybed the
    # labels misnamed (user 2026-11-26, OneFormer COCO: the green house's living-room corner - five
    # chairs round it - read as a bedroom from a 5 m2 "bed"): the room is typed by its other
    # furniture. A bed on its own in an open room is still a bedroom
    if not closed and ev["bed"] and (ev["sofa"] > 0 or ev["chairs"] >= 2):
        ev = {**ev, "bed": 0.0}
    if ev["bed"] >= BED_MIN_M2:
        return "bedroom"
    if closed and ev["bed"] >= BED_CLOSED_M2:
        return "bedroom"
    if ev["utility"] and not ev["bed"] and ev["sofa"] < 0.8 and area_ < 10.0 and not ev["table"] \
            and ev["study"] < 2 and not ev["office"]:      # (the green house's study had a "washer")
        return "laundry_room"
    # a toilet / bath / shower: a bathroom - unless the room has a sofa (a stray label in the Home
    # Scan's family room)
    if ev["bath"] >= 1 and ev["bed"] < 0.5 and ev["sofa"] < 0.8:
        return "bathroom"
    # a small closed room with a sink, towel or mirror and no cooking fitting: a bathroom - when the
    # sign is real: two of them, or one of BATH_SOFT_MIN_M2 (user 2026-11-26, OneFormer COCO: the
    # Home Scan's hallway took a 0.01 m2 "sink" - a few splats seen through the bathroom door - and
    # was toured as a bathroom; the green house's bathroom basin is 0.04 m2)
    if closed and area_ < SMALL_ROOM_M2 and ev["bath_soft"] >= 1 and not ev["cooking"] and ev["bed"] < 0.5 \
            and (ev["bath_soft"] >= 2 or ev.get("bath_soft_max", 1.0) >= BATH_SOFT_MIN_M2):
        return "bathroom"
    # a kitchen cooks: two kinds of fitting, one of them for cooking or washing up (a fridge and a
    # sink alone are a hall's built-ins and a bathroom's basin)
    kitchen = ev["kitchen"] if ev["kitchen_kinds"] >= 2 and ev["cooking"] >= 1 else 0.0
    living = ev["living"] if ev["living"] >= 0.8 else 0.0
    # a sofa alone in a closed room, no TV or fireplace: a bed the labels called a sofa
    if closed and ev["sofa"] >= BED_MIN_M2 and not ev["tv"] and not kitchen and area_ < 16.0:
        return "bedroom"
    # a table and chairs with books / shelves / a lamp, off the open plan: a study
    if ev["table"] > 0 and ev["chairs"] >= 1 and ev["study"] >= 2 and ev["sofa"] < 0.8 \
            and not (_open_neighbours(space, key) & set(getattr(space, "zone_kinds", {}))):
        return "home_office"
    if kitchen or living:
        return "kitchen" if kitchen >= living else "living_room"
    if ev["table"] >= DINING_TABLE_M2 and ev["chairs"] >= 3:
        return "dining_room"
    if ev["office"]:                       # a desk, a computer or a monitor
        return "home_office"
    # a small table with a chair (the labels often call a desk a table) is an office only in a room
    # of its own: in an open space it is a corner of its neighbour (the Home Scan's family room end,
    # 2026-11-16) - "other_room", which merges into it
    mates = _open_neighbours(space, key)
    if ev["table"] > 0 and ev["chairs"] >= 1 and not (mates & set(getattr(space, "zone_kinds", {}))) \
            and (not mates or getattr(space, "zone_kinds", None)):
        return "home_office"
    mask = space.room_of == space.room_keys.index(key)
    if not mask.any():
        return "other_room"
    ys, xs = np.nonzero(mask)
    narrow = float(_clear(space)[mask].max())
    long_side = max(np.ptp(ys), np.ptp(xs)) * CELL
    # clearance to the walls (built-in cupboards along a corridor are walls): a corridor up to
    # HALL_WIDE wall to wall (the green house's hall, 2026-11-17, ~1.6 m)
    return "hallway" if narrow <= HALL_WIDE / 2 and long_side >= 3.0 else "other_room"


def _open_neighbours(space, key: str) -> set[str]:
    """The rooms `key` opens straight onto (an opening wider than a door - not through a third room)."""
    if key not in space.room_keys:
        return set()
    me = space.room_keys.index(key)
    out = set()
    for (i, j), d in _saddles(space, len(space.room_keys)).items():
        if me in (i, j) and 2 * d >= OPEN_WIDE:
            out.add(space.room_keys[j if i == me else i])
    return out


def _mattress_m2(space, key: str) -> float:
    """The room's biggest surface at mattress height (0.35-0.9 m), floor-plan m2 - its bed."""
    from tour_worker.planner3d.space import CELL
    top = getattr(space, "_top", None)
    if top is None or key not in space.room_keys:
        return 0.0
    m = (space.room_fill == space.room_keys.index(key)) & (top > 0.35) & (top < 0.9)
    lab, n = ndi.label(ndi.binary_closing(m, iterations=2))
    if n == 0:
        return 0.0
    return float(np.bincount(lab.ravel())[1:].max()) * CELL * CELL


def _held(objects, keys) -> dict[str, dict[str, list[dict]]]:
    held: dict[str, dict[str, list[dict]]] = {k: {} for k in keys}
    for o in objects:
        if o.get("room") in held:
            held[o["room"]].setdefault(o["class"], []).append(o)
    return held


def room_types(space, objects: list[dict], points: dict[str, np.ndarray]) -> dict[str, str]:
    """{room key: room type} from the labelled objects in each room (module docstring): the primary
    bedroom is the largest bedroom by floor and bed. There is no family room: every sofa area is a
    living room."""
    from tour_worker.planner3d.space import CELL
    held = _held(objects, points)
    ev = {k: _evidence(h) for k, h in held.items()}
    out = {k: _type_of(space, k, ev[k]) for k in points}
    area = {k: float((space.room_of == space.room_keys.index(k)).sum()) * CELL * CELL for k in points}
    beds = [k for k in out if out[k] == "bedroom"]
    if beds:
        # the primary bedroom has the biggest bed (a double over a single - the Home Scan's kids'
        # room had more floor), then the most floor
        # (user 2026-11-17: "we are marking the smaller bedrooms as primary") - the room's whole
        # footprint, under the furniture too: a big bed takes free floor away from the big room
        foot = {k: float((space.room_fill == space.room_keys.index(k)).sum()) * CELL * CELL
                if k in space.room_keys else 0.0 for k in beds}
        # the bed measured off the scan (the biggest surface at mattress height): one clearly bigger
        # (BED_CLEAR) is the primary's - a queen over a twin (the Home Scan's kids' room had more
        # footprint); otherwise the largest room (the green house's beds are near alike)
        # the bed its labelled splats make (Space.labelled_piece) when the labels are there - the
        # mattress-height surface merges a kids' bed with its desk and shelves (Home Scan v15)
        lp = {k: (space.labelled_piece(k, "bed") if getattr(space, "labels", None) is not None else None) for k in beds}
        mat = {k: (lp[k].area if lp[k] is not None else _mattress_m2(space, k)) for k in beds}
        top2 = sorted(mat.values(), reverse=True)
        if len(beds) > 1 and top2[1] > 0 and top2[0] >= BED_CLEAR * top2[1]:
            out[max(beds, key=lambda k: mat[k])] = "primary_bedroom"
        else:
            out[max(beds, key=lambda k: (round(foot[k], 0), mat[k]))] = "primary_bedroom"
    # no "family room": a second sofa area is a living room too - merged into the first when they
    # share an open space (finalize_rooms), else a living room of its own (user 2026-11-18: the
    # Home Scan's "family room" took the living room's place and the living room was skipped)
    return out


# ---- an open plan's rooms, from where its furniture stands --------------------------------------
# (user 2026-11-17: "the room discovery needs to get a better idea of the room boundary when the
# layout is a bit open using the objects and where they end up on floor plans") - walls cannot split
# an open plan; its furniture does: a sofa group is the living room, a table with chairs round it the
# dining room, the sink / hob / fridge run the kitchen
ZONES = {"living": ("sofa", "armchair", "coffee table", "television receiver", "fireplace"),
         "kitchen": KITCHEN, "bed": ("bed",), "office": OFFICE}      # (a bathroom is never part of an open plan)
ZONE_GAP = 1.5           # m: objects of one purpose this close are one zone
ZONE_MIN = {"living": 0.8, "kitchen": 1.0, "bed": 1.2, "office": 0.3, "bath": 0.1, "dining": 1.0}   # m2 of their furniture
DINING_REACH = 1.2       # m from a table's edge its chairs stand (pulled out round a round table)
WALL_END = 0.4           # m at each end of a line between two objects where a wall does not part them
SEED_ROOM = 0.45         # m clear round where a zone's room starts
SEED_APART = 1.5         # m between two zones' starts
ZONE_NEAR = 2.0          # m: a room's centre this near a zone makes it that zone's kind of room
ZONE_APART = 2.0         # m between zone centres for them to be two rooms
ZONE_FLOOR_M2 = 3.0      # a zone's room needs this much free floor, else it joins its neighbour
# a kitchen's appliances (stove, hood, dishwasher - COOKING) make it one however little open floor
# the split leaves it: a galley kitchen beside a dining table (green house 2026-11-18: 2.3 m2 once
# the dining room's start moved 0.5 m, and the kitchen was folded into the dining room)
ZONE_FLOOR_MIN = {"kitchen": 1.5}
ZONE_TYPE = {"living": "living_room", "dining": "dining_room", "kitchen": "kitchen", "office": "home_office",
             "bath": "bathroom"}        # a zone's room is what its zone is (beds: room_types picks the primary)


def _box(o):
    return np.asarray(o["min"][:2], float), np.asarray(o["max"][:2], float)


def _gap(a, b) -> float:
    (la, ha), (lb, hb) = a, b
    return float(np.maximum(0.0, np.maximum(la - hb, lb - ha)).max())


def _through_wall(space, a, b) -> bool:
    """A wall (Space.walls) on the straight line between two floor points."""
    if space is None or getattr(space, "walls", None) is None:
        return False
    from tour_worker.planner3d.space import CELL
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    n = max(2, int(L / CELL))
    for q in np.linspace(a, b, n)[1:-1]:
        # a fitting stands against (or in) its wall - a hood, tall units: only a wall well between
        # the two counts, not within WALL_END of either
        if min(np.linalg.norm(q - a), np.linalg.norm(q - b)) < WALL_END:
            continue
        i, j = space.cell(q)
        if 0 <= i < space.walls.shape[0] and 0 <= j < space.walls.shape[1] and space.walls[i, j]:
            return True
    return False


def zones(objects: list[dict], space=None) -> list[dict]:
    """[{kind, centre (xy), weight, objects}] - the purposes the furniture shows, where it stands.
    Objects of one purpose join only on the same side of every wall (space given): the green house's
    hall built-ins and bathroom basin chained into its kitchen through the walls."""
    out = []
    # a dining table: at table height (a coffee table is lower), 3+ chairs pulled up round it
    # (a "table" topping out over DINING_TOP_MAX is a counter: a kitchen peninsula with its stools is
    # no dining room - the Home Scan's, OneFormer COCO, user 2026-11-26)
    tables = [o for o in objects if o["class"] == "table" and _area(o) >= 0.4
              and 0.6 <= float(o.get("top", 0.75)) <= DINING_TOP_MAX]
    chairs = [o for o in objects if o["class"] == "chair"]
    books = [o for o in objects if o["class"] in ("book", "bookcase")]
    # a table with books beside it is a desk (the green house's study), not dining (shelves and
    # lamps stand by dining tables too)
    dining = [t for t in tables if sum(_gap(_box(t), _box(c)) <= DINING_REACH for c in chairs) >= 3
              and sum(_gap(_box(t), _box(o)) <= 1.5 for o in books) < 2]
    groups = [("dining", o) for o in dining]
    for kind, classes in ZONES.items():
        groups += [(kind, o) for o in objects if o["class"] in classes]
    for kind in set(k for k, _ in groups):
        items = [o for k, o in groups if k == kind]
        clusters: list[list[dict]] = []
        for o in items:
            near = [c for c in clusters if any(_gap(_box(o), _box(x)) <= ZONE_GAP
                                               and not _through_wall(space, o["centre"], x["centre"]) for x in c)]
            merged = [o] + [x for c in near for x in c]
            clusters = [c for c in clusters if c not in near] + [merged]
        for c in clusters:
            w = np.array([max(_area(o), 0.05) for o in c])
            if w.sum() < ZONE_MIN[kind] or (kind == "kitchen" and (len({o["class"] for o in c}) < 2
                                                                     or not any(o["class"] in COOKING for o in c))):
                continue
            if kind == "living" and not any(o["class"] in LIVING for o in c):
                continue                          # armchairs alone are a reading corner, not a living room
            # a kitchen stands where its fittings are (a run of tall white built-ins labelled
            # "refrigerator" along a hall drew the green house's kitchen into its hall)
            core = [n for n, o in enumerate(c) if not (kind == "kitchen" and o["class"] == "refrigerator")] or list(range(len(c)))
            cw = w[core]
            centre = (np.array([c[n]["centre"][:2] for n in core], float) * cw[:, None]).sum(0) / cw.sum()
            out.append({"kind": kind, "centre": centre, "weight": float(w.sum()), "objects": c})
    return sorted(out, key=lambda z: -z["weight"])


def _zone_seed(space, c: np.ndarray, radius: float = 2.0, avoid=(), others=()) -> np.ndarray:
    """Where a zone's room starts: beside the zone's own furniture - the nearest free floor with
    some room round it (SEED_ROOM m clear) that is nearer this zone than any other and SEED_APART
    from the zones placed already; failing that, the most open floor within radius."""
    from tour_worker.planner3d.space import CELL
    i, j = space.cell(c)
    k = int(radius / CELL)
    H, W = space.shape
    a0, a1, b0, b1 = max(0, i - k), min(H, i + k + 1), max(0, j - k), min(W, j + k + 1)
    yy, xx = np.mgrid[a0:a1, b0:b1]
    pts = np.stack([space.lo[0] + (xx + .5) * CELL, space.lo[1] + (yy + .5) * CELL], -1)
    d_own = np.hypot(yy - i, xx - j) * CELL
    ok = space.free[a0:a1, b0:b1] & (d_own <= radius)
    for o in others:                                  # nearer this zone than the others
        ok &= d_own < np.linalg.norm(pts - np.asarray(o)[:2], axis=-1)
    for s in avoid:                                   # apart from the rooms started already
        ok &= np.linalg.norm(pts - np.asarray(s)[:2], axis=-1) >= SEED_APART
    good = ok & (space.dt[a0:a1, b0:b1] >= SEED_ROOM)
    if good.any():
        y, x = np.unravel_index(int(np.argmin(np.where(good, d_own, np.inf))), good.shape)
        return space.centre_of(a0 + y, b0 + x)
    if ok.any():
        y, x = np.unravel_index(int(np.argmax(np.where(ok, space.dt[a0:a1, b0:b1], -1.0))), ok.shape)
        return space.centre_of(a0 + y, b0 + x)
    return c


def split_by_zones(space, points: dict[str, np.ndarray], objects: list[dict]):
    """An open space (rooms joined by wide openings, or one big room) holding furniture of two or
    more purposes, ZONE_APART apart, becomes one room per zone: the zones' centres replace its room
    centres and the floor goes to the nearest by walking (Space.set_rooms). A zone whose room has
    under ZONE_FLOOR_M2 of floor joins its neighbour. Returns (points, warnings)."""
    from tour_worker.planner3d.space import CELL
    space.set_rooms(points)
    for o in objects:
        o["room"] = space.room_of_object(np.asarray(o["centre"][:2]))
    groups, seen = [], set()
    for k in space.room_keys:
        if k not in seen:
            g = space.spaces.get(k, {k})
            seen |= g
            groups.append(sorted(g))
    out, warnings, kinds, owner = dict(points), [], {}, {}
    for g in groups:
        zs = zones([o for o in objects if o.get("room") in g], space)
        if any(z["kind"] == "bed" for z in zs):                 # a bedroom's desk is not a room
            zs = [z for z in zs if z["kind"] not in ("office",)]
        keep: list[dict] = []
        for z in zs:
            if all(np.linalg.norm(z["centre"] - y["centre"]) >= ZONE_APART for y in keep):
                keep.append(z)
        if len({z["kind"] for z in keep}) < 2:
            continue
        while len(keep) >= 2:
            # the group's rooms that hold no zone (a hall, a study open to the living room) keep
            # their own centres; the rooms the zones stand in give theirs to the zones
            space.set_rooms(out)
            zone_rooms = {space.room_at(z["centre"]) or space.room_near(z["centre"], 1.5) for z in keep}
            trial = {k: v for k, v in out.items() if k not in g or k not in zone_rooms}
            # a zone's room keeps the key of the room it stands in (its capture, its object list),
            # the first zone there; another zone in the same room gets <room>_<kind>
            names, used = [], set(trial)
            for z in keep:
                home = space.room_at(z["centre"]) or space.room_near(z["centre"], 1.5) or g[0]
                nm = home if home not in used else f"{home}_{z['kind']}"
                n = 1
                while nm in used:
                    nm, n = f"{home}_{z['kind']}{n}", n + 1
                names.append(nm)
                used.add(nm)
            seeds: list[np.ndarray] = []
            for nm, z in zip(names, keep):
                # (v11: the living and dining rooms both started in the same open aisle, 0.7 m apart)
                trial[nm] = _zone_seed(space, z["centre"], avoid=seeds,
                                       others=[y["centre"] for y in keep if y is not z])
                seeds.append(trial[nm])
            space.set_rooms(trial)
            area = {nm: float((space.room_of == space.room_keys.index(nm)).sum()) * CELL * CELL
                    if nm in space.room_keys else 0.0 for nm in names}
            small = [n for n, nm in enumerate(names) if area[nm] < ZONE_FLOOR_MIN.get(keep[n]["kind"], ZONE_FLOOR_M2)]
            if not small:
                out = trial
                kinds.update({nm: z["kind"] for nm, z in zip(names, keep)})
                # a zone's furniture is its room's - not a fitting behind a wall from the zone's
                # centre (user 2026-11-26: the green house's kitchen zone took the sink, fridge and
                # counter of the bathroom behind the hall built-ins, which then read as a hallway)
                owner.update({id(o): nm for nm, z in zip(names, keep) for o in z["objects"]
                              if not _through_wall(space, o["centre"], z["centre"])})
                warnings.append("open_plan_split:" + "+".join(sorted(zone_rooms - {None})) + "->" + ",".join(names))
                break
            keep.pop(min(small, key=lambda n: keep[n]["weight"]))       # the lightest small zone joins its neighbour
    space.set_rooms(out)
    for o in objects:                    # a zone's furniture is its room's, wherever the floor is nearest
        o["room"] = owner.get(id(o)) or space.room_of_object(np.asarray(o["centre"][:2]))
    space.zone_kinds = kinds
    return out, warnings


def zone_types(space, objects, points: dict, types: dict) -> dict:
    """A room is what the furniture at its centre is for: an open plan's zone rooms are their zones;
    any other room takes the zone nearest its centre (within ZONE_NEAR) - the Home Scan's dining room
    voted "living" from a sofa at its edge. Beds stay room_types' (it picks the primary)."""
    from tour_worker.planner3d.space import CELL
    out = dict(types)
    zk = getattr(space, "zone_kinds", {})
    every = [z for z in zones(objects, space) if z["kind"] in ZONE_TYPE]     # the whole home's: chairs round a
    for k, p in points.items():                                        # table may stand in the next room
        kind = zk.get(k)
        if kind is None:
            zs = [z for z in every if (space.room_at(z["centre"]) or space.room_near(z["centre"], 1.0)) == k]
            near = min(zs, key=lambda z: float(np.linalg.norm(z["centre"] - p)), default=None)
            if near is not None and np.linalg.norm(near["centre"] - p) <= ZONE_NEAR \
                    and out[k] not in ("bedroom", "primary_bedroom", "hallway", "bathroom", "home_office"):
                kind = near["kind"]
        if kind in ZONE_TYPE:
            out[k] = ZONE_TYPE[kind]
    return out


MERGE_SAME = {"living_room", "family_room", "dining_room", "home_office", "bedroom", "primary_bedroom",
              "kitchen", "hallway", "other_room", "bathroom"}


DINING_TOP_MAX = 1.0         # m: a table higher than this is a counter or bar
SEATING_GROUP_M2 = 2.0       # m2 of sofa: a living area's own seating group (two such, apart, are two rooms)
BATH_SOFT_MIN_M2 = 0.02      # m2: a lone sink / towel / mirror smaller than this is noise, not a bathroom
ANNEX_M2 = 4.0               # m2 of free floor: a room under this, entered only through a bedroom, is part of it
BEDROOMS = ("bedroom", "primary_bedroom", "guest_bedroom", "childrens_bedroom")


def _leaves_into(space, key: str, keys: list[str]) -> set[str]:
    """The rooms one first walks into leaving `key` for every other room (Space.route): a single
    one means `key` is reached only through it."""
    out = set()
    for other in keys:
        if other == key:
            continue
        r = space.route(key, other)
        if r and len(r[1]) > 1:
            out.add(r[1][1])
    return out


def finalize_rooms(space, points: dict[str, np.ndarray], objects: list[dict],
                   capture: dict[str, float] | None = None):
    """The rooms as tour_plan takes them, from the found rooms and the scene's labels:
      - neighbours in one open space merge when they are the same kind of room, or one is an
        "other_room" (one family room split in three is one room; a kitchen beside a living room
        stays two), the more open centre kept, then typed again;
      - a room the scan covers poorly (capture < CAPTURE_MIN of its survey views) is not toured.
    Returns ([(key, type, centre, toured)], warnings); objects' rooms are updated to the merges.
    First, an open plan's rooms come from its furniture (split_by_zones)."""
    points, zone_w = split_by_zones(space, dict(points), objects)
    keys = list(points)
    merged: dict[str, str] = {}
    while True:
        space.set_rooms({k: points[k] for k in keys})
        for o in objects:
            if o.get("room") in merged:
                o["room"] = merged[o["room"]]
        types = room_types(space, objects, {k: points[k] for k in keys})
        types = zone_types(space, objects, {k: points[k] for k in keys}, types)
        sad = _saddles(space, len(keys))
        clear = {i: float(_clear(space)[space.cell(points[k])]) for i, k in enumerate(keys)}
        pairs = []
        for (i, j), d in sad.items():
            a, b = keys[i], keys[j]
            ta, tb = types[a], types[b]
            living_pair = {ta, tb} <= {"living_room", "family_room"}
            # (an unrecognisable corner opening onto its neighbour is part of it too - the green
            # house's living-room corner, once COCO's "bed" there no longer makes it a bedroom)
            other_open = "other_room" in (ta, tb) and b in _open_neighbours(space, a)
            if b not in space.spaces.get(a, {a}) and not ((living_pair or other_open) and b in _open_neighbours(space, a)):
                continue                                   # a doorway between them: two rooms
            zk = getattr(space, "zone_kinds", {})
            # (a non-zone room may fold INTO a zone room - the green house's living end became a
            # "family room" of its own - never the other way)
            same = ta == tb or {ta, tb} <= {"living_room", "family_room"}
            if same and {ta, tb} <= {"living_room", "family_room"}:
                # each with its own sofa group and a TV or fireplace: two living rooms, not one (the
                # Home Scan's wood-stove living room was folded into its family room)
                ea, eb = (_evidence(_held(objects, [x])[x]) for x in (a, b))
                # ... far apart: two rooms. Near (one open area seen from two sides - the green house's
                # living room was captioned "Family Room" from its other end, user 2026-11-17): one
                # (or each with a seating group of its own, TV or not - OneFormer COCO misses TVs;
                # the Home Scan's family room, user 2026-11-26: "it's a different room - the planner
                # should know when it's a second room")
                apart = float(np.linalg.norm(points[a] - points[b])) >= LIVING_APART
                if apart and (min(ea["sofa"], eb["sofa"]) >= SEATING_GROUP_M2
                              or (min(ea["sofa"], eb["sofa"]) >= 0.8 and min(ea["tv"], eb["tv"]) >= 1)):
                    continue
            if same or "other_room" in (ta, tb):
                pairs.append((d, i, j))
        if not pairs:
            break
        _, i, j = max(pairs)
        keep, drop = (keys[i], keys[j]) if clear[i] >= clear[j] else (keys[j], keys[i])
        zk = getattr(space, "zone_kinds", {})
        if drop in zk and keep not in zk:
            keep, drop = drop, keep
        if types[drop] != "other_room" and types[keep] == "other_room":
            keep, drop = drop, keep
        merged = {**{k: (keep if v == drop else v) for k, v in merged.items()}, drop: keep}
        keys = [k for k in keys if k != drop]
    types = zone_types(space, objects, {k: points[k] for k in keys}, room_types(space, objects, {k: points[k] for k in keys}))
    # a small room entered only through a bedroom, with no bed of its own, is part of that bedroom
    # (its en-suite or dressing area - user 2026-11-26: the green house's wardrobe-and-shower nook
    # off the primary bedroom was toured as a "Bedroom" of its own, typed by the edge of the
    # primary's bed over the boundary): it folds into the bedroom, whose shots show it
    while True:
        annex = None
        for k in keys:
            if space.room_area(k) >= ANNEX_M2 or _evidence(_held(objects, [k])[k])["bed"] >= BED_MIN_M2:
                continue
            out_to = _leaves_into(space, k, keys)
            if len(out_to) == 1 and types[next(iter(out_to))] in BEDROOMS:
                annex = (k, next(iter(out_to)))
                break
        if annex is None:
            break
        drop, keep = annex
        merged = {**{k: (keep if v == drop else v) for k, v in merged.items()}, drop: keep}
        keys = [k for k in keys if k != drop]
        space.set_rooms({k: points[k] for k in keys})
        for o in objects:
            if o.get("room") in merged:
                o["room"] = merged[o["room"]]
        types = zone_types(space, objects, {k: points[k] for k in keys}, room_types(space, objects, {k: points[k] for k in keys}))
    warnings = zone_w + [f"rooms_merged:{d}->{k}" for d, k in sorted(merged.items())]
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
        if types[k] in ("laundry_room", "utility_room"):
            toured = False                                 # a washing machine is not a room to tour
        if not toured:
            warnings.append(f"room_not_toured:{k}:" + ("other_room" if types[k] == "other_room" else f"capture_{c:.2f}"))
        out.append((k, types[k], points[k], toured))
    return out, warnings
