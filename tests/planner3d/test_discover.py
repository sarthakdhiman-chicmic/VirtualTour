"""Rooms with no input (planner3d/discover.py) and the 3D tour's walk order (planner3d/plan.py),
from the Home Scan review (user 2026-11-16): one open space split into three "family rooms", an
office typed as a dining room, the primary bedroom typed as a bathroom, a poorly captured room
toured, the tour jumping across the house and closing in the "bathroom"."""
import types

import numpy as np

from tests.planner3d.test_planner3d import home
from tour_worker.planner3d.discover import finalize_rooms, room_types
from tour_worker.planner3d.space import Space


def _o(cls, room, x0, y0, x1, y1, top=0.8):
    c = [(x0 + x1) / 2, (y0 + y1) / 2, top / 2]
    return {"class": cls, "room": room, "centre": c, "min": [x0, y0, 0.0], "max": [x1, y1, top], "top": top,
            "footprint": {"centre": c[:2], "long": max(x1 - x0, y1 - y0), "short": min(x1 - x0, y1 - y0),
                          "axis_deg": 0.0}}


def _space(door=1.0):
    sp = Space(home(door=door, bed=False))
    pts = {"a": np.array([3.0, 3.0]), "b": np.array([8.0, 2.5])}
    sp.set_rooms(pts)
    return sp, pts


def test_a_room_is_what_dominates_it():
    sp, pts = _space()
    # bed pieces outweigh a stray sink and toilet (an en-suite's fixtures); a big sofa outweighs
    # fridge pieces (a kitchen's things seen across an open plan)
    objs = [_o("bed", "b", 7.6, 3.0, 8.6, 4.6), _o("bed", "b", 8.6, 3.0, 9.6, 4.6),
            _o("sink", "b", 9.5, 0.2, 9.9, 0.6), _o("toilet", "b", 9.4, 1.0, 9.9, 1.5),
            _o("sofa", "a", 1.0, 0.4, 3.2, 1.3), _o("refrigerator", "a", 5.5, 4.4, 5.9, 4.9),
            _o("sink", "a", 5.4, 3.8, 5.9, 4.2)]
    assert room_types(sp, objs, pts) == {"a": "living_room", "b": "primary_bedroom"}
    # a desk with its chair is an office, not a dining room; a toilet without a bed a bathroom
    objs = [_o("table", "a", 1.0, 1.0, 2.0, 1.6), _o("chair", "a", 1.2, 1.7, 1.6, 2.1),
            _o("chair", "a", 1.7, 1.7, 2.0, 2.0), _o("chair", "a", 1.0, 1.7, 1.2, 2.0),
            _o("toilet", "b", 9.4, 1.0, 9.9, 1.5)]
    assert room_types(sp, objs, pts) == {"a": "home_office", "b": "bathroom"}


def test_the_primary_bedroom_is_the_largest_bedroom():
    sp, pts = _space()
    objs = [_o("bed", "a", 1.0, 1.0, 2.6, 3.0), _o("bed", "b", 7.6, 3.0, 9.6, 4.6)]
    t = room_types(sp, objs, pts)
    assert t["a"] == "primary_bedroom" and t["b"] == "bedroom"         # the 6 x 5 m room, not the bigger bed


def test_one_open_space_is_one_room_and_a_poorly_captured_room_is_not_toured():
    sp, pts = _space(door=2.4)                                           # one open space
    sofas = [_o("sofa", "a", 1.0, 0.4, 3.2, 1.3), _o("sofa", "b", 7.6, 3.0, 9.6, 4.0)]
    rooms, warn = finalize_rooms(sp, dict(pts), [dict(o) for o in sofas], {"a": 0.95, "b": 0.9})
    assert [(t, toured) for _, t, _, toured in rooms] == [("living_room", True)]      # the more open centre kept
    assert any(w.startswith("rooms_merged:") for w in warn)
    beds = [sofas[0], _o("bed", "b", 7.6, 3.0, 9.6, 4.6)]               # a bed: a different room, kept
    rooms, warn = finalize_rooms(sp, dict(pts), [dict(o) for o in beds], {"a": 0.95, "b": 0.5})
    assert {k: (t, toured) for k, t, _, toured in rooms} == {"a": ("living_room", True),
                                                            "b": ("primary_bedroom", False)}
    assert "room_not_toured:b:capture_0.50" in warn


def test_the_tour_follows_the_walk_and_never_closes_in_a_bathroom():
    from tour_worker.planner3d.plan import _walk_order
    pos = {"living": 0.0, "far_family": 20.0, "kitchen": 2.0, "bed": 30.0, "bath": 31.0}
    space = types.SimpleNamespace(route=lambda a, b: (np.array([[pos[a], 0.0], [pos[b], 0.0]]), [a, b]),
                                  room_points={k: np.array([v, 0.0]) for k, v in pos.items()})
    sel = [types.SimpleNamespace(room_key=k, final_duration=4.0) for k in ("living", "far_family", "kitchen", "bed", "bath")]
    sel[-1].final_duration = 8.0                                         # the closing length
    stage = {"living": "living", "far_family": "living", "kitchen": "living", "bed": "private", "bath": "private"}
    rtype = {"living": "living_room", "far_family": "family_room", "kitchen": "kitchen", "bed": "bedroom", "bath": "bathroom"}
    cfg = types.SimpleNamespace(pacing="normal")
    out = _walk_order(space, sel, stage, rtype, cfg)
    assert [s.room_key for s in out] == ["living", "kitchen", "far_family", "bath", "bed"]
    assert out[-1].final_duration == 8.0                                 # the closing length moved with the close
