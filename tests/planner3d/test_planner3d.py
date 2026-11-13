"""The 3D planner (tour_worker/planner3d): from a splat scene and its rooms (type + one point) alone,
a plan that follows the tour rules. A synthetic two-room home: a living room with a sofa, a doorway,
a bedroom with a bed, walls 2.5 m high."""
import math

import numpy as np
import pytest

from tour_worker.planner.config import PlannerConfig
from tour_worker.planner3d.moves import (HFOV, JUMP_ANGLE, JUMP_DIST, MAX_MOVE_MPS, MAX_TURN_DPS, PITCH_LADDER,
                                        speeds_of)
from tour_worker.planner3d.plan import plan_tour_3d, room_keys
from tour_worker.planner3d.space import Space
from tour_worker.scene.splat_io import SH_C0, SplatModel

S = 0.025                      # splat spacing: 4 a floor-plan cell per surface


def _grid(x0, x1, y0, y1):
    xs, ys = np.arange(x0 + S / 2, x1, S), np.arange(y0 + S / 2, y1, S)
    a, b = np.meshgrid(xs, ys, indexing="ij")
    return a.ravel(), b.ravel()


def _wall_x(x, y0, y1, z1=2.5):          # a wall along y at x
    a, b = _grid(y0, y1, 0, z1)
    return np.c_[np.full_like(a, x), a, b]


def _wall_y(y, x0, x1, z1=2.5):
    a, b = _grid(x0, x1, 0, z1)
    return np.c_[a, np.full_like(a, y), b]


def _box(x0, x1, y0, y1, top):           # a low piece: its body (layers, as a scan has volume) and sides
    a, b = _grid(x0, x1, y0, y1)
    parts = [np.c_[a, b, np.full_like(a, z)] for z in (0.35, (0.35 + top) / 2, top)]
    for z in np.arange(0.2, top, 0.1):
        for x in (x0, x1):
            yy = np.arange(y0, y1, S)
            parts.append(np.c_[np.full_like(yy, x), yy, np.full_like(yy, z)])
        for y in (y0, y1):
            xx = np.arange(x0, x1, S)
            parts.append(np.c_[xx, np.full_like(xx, y), np.full_like(xx, z)])
    return np.concatenate(parts)


def home(door=1.0, bed=True, hall=False, split=False) -> SplatModel:
    """Living room x 0-6, bedroom x 6-10, y 0-5; a door `door` m wide in the wall at x = 6.
    hall: the living room ends at x = 4.5 and a 1.5 m hallway (x 4.5-6) joins it to the bedroom,
    a door at each end, in line. split: the east side is a kitchen (y 0-3, its door from the living
    room at y 0.9-1.9) and a bedroom behind it (y 3-5, its door from the kitchen at x 6.3-7.3)."""
    a, b = _grid(0, 10, 0, 5)
    parts = [np.c_[a, b, np.zeros_like(a)], np.c_[a, b, np.full_like(a, 2.6)]]          # floor, ceiling
    parts += [_wall_x(0, 0, 5), _wall_x(10, 0, 5), _wall_y(0, 0, 10), _wall_y(5, 0, 10)]
    d0 = 2.5 - door / 2
    if split:
        parts += [_wall_x(6, 0, 0.9), _wall_x(6, 1.9, 5), _wall_y(3.0, 6, 6.3), _wall_y(3.0, 7.3, 10)]
    else:
        for x in ((4.5, 6) if hall else (6,)):
            parts += [_wall_x(x, 0, d0), _wall_x(x, d0 + door, 5)]
    parts += [_box(1.0, 3.2, 0.4, 1.3, 0.8)]          # sofa against the living room's south wall
    if bed:                                           # bed in the bedroom's north-east corner
        parts += [_box(7.6, 9.6, 3.3, 4.9, 0.6) if split else _box(7.6, 9.6, 3.0, 4.6, 0.6)]
    means = np.concatenate(parts).astype(np.float32)
    n = len(means)
    sh = np.zeros((n, 1, 3), np.float32)
    sh[:, 0] = (0.6 - 0.5) / SH_C0
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1
    return SplatModel(means, np.full((n, 3), 0.02, np.float32), quats, np.full(n, 0.95, np.float32), sh, 0,
                      "ply", {})


ROOMS = [{"room_key": "bed", "room_type": "Bedroom", "point": [8.0, 1.5, 1.0]},     # listed first on purpose
         {"room_key": "living", "room_type": "Living Room", "point": [3.0, 3.0, 1.0]}]
SCENE = {"uri": "s3://b/home.ply", "sha256": "ab" * 32, "format": "ply"}


@pytest.fixture(scope="module")
def model():
    return home()


@pytest.fixture(scope="module")
def planned(model):
    return plan_tour_3d(model, SCENE, ROOMS, "test-home", PlannerConfig(), check_occlusion=False)


def test_the_space_finds_the_floor_rooms_and_furniture(model):
    sp = Space(model)
    assert abs(sp.up_scene @ np.array([0, 0, 1.0])) > 0.999 and abs(sp.room_height - 2.6) < 0.1
    assert sp.set_rooms({"living": np.array([3.0, 3.0]), "bed": np.array([8.0, 1.5])}) == []
    assert sp.room_at((1.0, 4.0)) == "living" and sp.room_at((9.0, 1.0)) == "bed"
    assert sp.spaces["living"] == {"living"}                       # a 1 m door: two rooms, not an open plan
    pieces = {sp.piece_room(p): p for p in sp.main_pieces("living") + sp.main_pieces("bed")}
    assert abs(pieces["living"].centre - (2.1, 0.85)).max() < 0.15 and abs(pieces["living"].top - 0.8) < 0.1
    assert abs(pieces["bed"].centre - (8.6, 3.8)).max() < 0.15
    assert not sp.tall[sp.cell((2.1, 0.85))] and sp.tall[sp.cell((6.0, 0.5))]   # a sofa does not block a view; a wall does


def test_a_wide_opening_makes_one_open_plan_space():
    sp = Space(home(door=2.4))
    sp.set_rooms({"living": np.array([3.0, 3.0]), "bed": np.array([8.0, 1.5])})
    assert sp.spaces["living"] == {"living", "bed"}


def test_the_plan_follows_the_photo_planners_order_and_screen_time(planned):
    plan, rep, _ = planned
    keys = [s.room_key for s in plan.shots]
    assert keys[0] == "living" and keys[-1] == "bed"               # narrative order, not request order
    assert keys.count("living") == 2 and keys.count("bed") == 1    # the living room is a hero room: two angles
    assert plan.mode == "cinematic_3d" and plan.plan_version == "1.6"
    from tour_worker.planner.lengths import ceiling_s
    assert all(3.0 - 1e-6 <= s.duration_s <= ceiling_s("hero", "normal") * 1.5 + 1e-6 for s in plan.shots)
    assert plan.shots[0].duration_s >= 5.0 - 1e-6                   # the opening shot holds
    assert [s.caption for s in plan.shots] == [None, None, None]   # no room labels asked for
    assert rep["sequence"] == ["living", "bed"]


def _poses(shot):
    from tour_worker.scene.camera import BezierCameraPath
    path = BezierCameraPath([x.model_dump() for x in shot.camera.segments], shot.camera.hfov_deg)
    return path.pose(0.0), path.pose(1.0)


def test_every_shot_keeps_the_rules(planned):
    plan, rep, _ = planned
    for s in plan.shots:
        cam = s.camera
        assert cam.type == "bezier_path" and cam.hfov_deg == [HFOV, HFOV]       # no zoom
        assert cam.time_warp is None and cam.punches is None                    # no ramps or pulses
        turn, move = speeds_of([x.model_dump() for x in cam.segments], s.duration_s)
        assert turn <= MAX_TURN_DPS + 0.05 and move <= MAX_MOVE_MPS + 0.005
        for eye, tgt, _ in _poses(s):
            d = np.asarray(tgt) - np.asarray(eye)
            pitch = math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1])))
            assert pitch <= PITCH_LADDER[-1] + 0.5                                 # the horizon stays up
            assert abs(eye[2] - 1.5) < 0.03                                        # eye height (floor found to ~1 cm)
        assert s.transition_in.type in ("cut", "fade")
    for a, b in zip(plan.shots, plan.shots[1:]):                                   # no jump cuts
        (_, (e0, t0, _)), ((e1, t1, _), _) = _poses(a), _poses(b)
        u, v = np.asarray(t0) - e0, np.asarray(t1) - e1
        ang = math.degrees(math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
        assert ang >= JUMP_ANGLE or np.linalg.norm(np.asarray(e0[:2]) - e1[:2]) >= JUMP_DIST
    # the v3 standard: a room with a piece is filmed by walkways composed on it - a hero room's
    # second angle on the same piece from elsewhere; never a pan inside a room
    living = [r for r in rep["shots"] if r["room_key"] == "living"]
    assert [r["move"] for r in living] == ["walkway", "walkway"] and living[0]["piece"]["area_m2"] > 1.5
    assert all(r["move"] != "pan" for r in rep["shots"][:-1])


def test_a_y_down_scene_plans_the_same_tour_in_its_own_axes(model, planned):
    """The same home written with COLMAP's axes (up = -y): the planner finds up, plans the same
    shots, and writes them in the scene's axes with up_axis set."""
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)       # z-up -> y-down
    m = SplatModel(model.means @ R.T, model.scales, model.quats, model.opacities, model.sh, 0, "ply", {})
    rooms = [{**r, "point": (np.asarray(r["point"]) @ R.T).tolist()} for r in ROOMS]
    plan, _, _ = plan_tour_3d(m, SCENE, rooms, "test-home", PlannerConfig(), check_occlusion=False)
    ref = planned[0]
    assert [s.room_key for s in plan.shots] == [s.room_key for s in ref.shots]
    assert np.allclose(plan.scene.up_axis, [0, -1, 0], atol=0.02)
    e_ref = np.asarray(ref.shots[0].camera.segments[0].eye[0])
    e = np.asarray(plan.shots[0].camera.segments[0].eye[0])
    assert abs(-e[1] - 1.5) < 0.05 and np.linalg.norm(e @ R - e_ref) < 0.3        # eye 1.5 m up, same spot


def test_room_keys_and_points_off_the_floor():
    assert room_keys([{"room_type": "Bedroom"}, {"room_type": "Bedroom"}, {"room_type": "Kitchen"}]) == \
        ["bedroom_1", "bedroom_2", "kitchen_1"]
    with pytest.raises(ValueError):
        room_keys([{"room_key": "a", "room_type": "x"}, {"room_key": "a", "room_type": "y"}])


def test_occlusion_is_measured_in_the_middle_of_the_frame(model):
    """A wall right beside the lens is part of the room; a post right in front of it is in the way."""
    from tour_worker.planner3d.occlusion import Occlusion
    sp = Space(model)
    occ = Occlusion(model, sp)
    assert occ.cover([0.4, 2.5, 1.5], [3.0, 2.5, 1.2], HFOV) < 0.05             # wall at x=0 behind / beside
    post = np.c_[np.full(400, 1.0), np.repeat(np.linspace(2.3, 2.7, 20), 20), np.tile(np.linspace(0.3, 2.3, 20), 20)]
    m2 = SplatModel(np.concatenate([model.means, post.astype(np.float32)]),
                    np.concatenate([model.scales, np.full((400, 3), 0.05, np.float32)]),
                    np.concatenate([model.quats, np.tile([[1, 0, 0, 0]], (400, 1)).astype(np.float32)]),
                    np.concatenate([model.opacities, np.full(400, 0.95, np.float32)]),
                    np.concatenate([model.sh, np.zeros((400, 1, 3), np.float32)]), 0, "ply", {})
    sp2 = Space(m2)
    assert Occlusion(m2, sp2).cover([0.4, 2.5, 1.5], [3.0, 2.5, 1.2], HFOV) > 0.2


# ---- with the scene's labels (scene/semantics.py) ---------------------------------------------------
def _labels(model):
    from tour_worker.scene.semantics import INDEX, SceneLabels
    P = model.means
    lab = np.zeros(len(model), np.uint8)
    sofa = (P[:, 0] > 0.95) & (P[:, 0] < 3.25) & (P[:, 1] > 0.35) & (P[:, 1] < 1.35) & (P[:, 2] > 0.15) & (P[:, 2] < 0.85)
    bed = (P[:, 0] > 7.55) & (P[:, 0] < 9.65) & (P[:, 1] > 2.95) & (P[:, 1] < 4.65) & (P[:, 2] > 0.15) & (P[:, 2] < 0.65)
    lab[sofa], lab[bed] = INDEX["sofa"], INDEX["bed"]
    vase = {"class": "vase", "centre": [5.0, 4.4, 0.9], "min": [4.85, 4.25, 0.7], "max": [5.15, 4.55, 1.1],
            "footprint": {"centre": [5.0, 4.4], "long": 0.3, "short": 0.3, "axis_deg": 0.0}, "top": 1.1,
            "splats": 400, "room": "living"}
    return SceneLabels(lab, "ab" * 32, len(model), [vase])


def test_labels_name_the_pieces_and_a_hero_room_with_a_piece_gets_a_second_angle_not_a_close_up(model):
    plan, rep, _ = plan_tour_3d(model, SCENE, ROOMS, "test-home", PlannerConfig(), check_occlusion=False,
                                labels=_labels(model))
    rooms = {r["room_key"]: r for r in rep["rooms"]}
    assert rooms["living"]["main_pieces"][0]["class"] == "sofa" and rooms["bed"]["main_pieces"][0]["class"] == "bed"
    assert [s.camera.move for s in plan.shots if s.room_key == "living"] == ["walkway", "walkway"]
    assert not [s for s in plan.shots if s.role == "close"]        # at most two shots a room


def _obj(cls, lo, hi):
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    return {"class": cls, "centre": c, "min": list(lo), "max": list(hi), "top": hi[2], "splats": 400, "room": None,
            "footprint": {"centre": c[:2], "long": max(hi[0] - lo[0], hi[1] - lo[1]),
                          "short": min(hi[0] - lo[0], hi[1] - lo[1]), "axis_deg": 0.0 if hi[0] - lo[0] >= hi[1] - lo[1] else 90.0}}


def test_a_kitchen_with_no_piece_pushes_toward_its_counter_and_gets_the_close_up():
    """A hero room with nothing to compose on (a kitchen of units) gets one push toward its own
    working surfaces - never an aim at its upper cabinets - and its close-up; never more than two
    shots a room."""
    from tour_worker.planner3d.moves import AIM_Z_MAX
    from tour_worker.scene.semantics import INDEX, SceneLabels
    m = home(split=True)
    P = m.means
    lab = np.zeros(len(m), np.uint8)
    lab[(P[:, 0] > 0.95) & (P[:, 0] < 3.25) & (P[:, 1] > 0.35) & (P[:, 1] < 1.35) & (P[:, 2] > 0.15) & (P[:, 2] < 0.85)] = INDEX["sofa"]
    objs = [_obj("cabinet", (7.0, 0.05, 0.0), (9.9, 0.65, 0.9)),      # the counter run along the south wall
            _obj("sink", (8.0, 0.1, 0.75), (8.6, 0.6, 0.9)),
            _obj("cabinet", (7.0, 0.05, 1.7), (9.9, 0.4, 2.4)),       # upper cabinets: never aimed at
            _obj("vase", (9.55, 2.5, 0.7), (9.85, 2.8, 1.1))]
    rooms = [{"room_key": "kit", "room_type": "Kitchen", "point": [8.0, 1.5, 1.0]},
             {"room_key": "bed", "room_type": "Bedroom", "point": [7.0, 4.0, 1.0]},
             {"room_key": "living", "room_type": "Living Room", "point": [3.0, 3.0, 1.0]}]
    plan, rep, _ = plan_tour_3d(m, SCENE, rooms, "test-home", PlannerConfig(), check_occlusion=False,
                                labels=SceneLabels(lab, "ab" * 32, len(m), objs))
    kit = [(s, r) for s, r in zip(plan.shots, rep["shots"]) if s.room_key == "kit"]
    assert [s.camera.move for s, _ in kit] in (["push", "detail"], ["pull", "detail"]) and kit[1][0].role == "close"
    assert kit[1][1]["object"] == "vase"                              # not the sink the push ends on
    anchor = kit[0][1]["anchor"]
    assert anchor[1] <= 0.7 and anchor[2] < 1.0                        # the counter, not the upper cabinets
    for s, _ in kit[:1]:
        assert all(seg.target[k][2] <= AIM_Z_MAX + 0.02 for seg in s.camera.segments for k in range(4))


def test_a_hallway_is_a_link_between_two_rooms_not_a_stop():
    """Living room -> hallway -> bedroom: the tour opens in the living room (not the hallway), pans
    from it onto the hallway toward the bedroom, then walks into the bedroom from the hallway - cuts
    all along, one walk through the home."""
    m = home(hall=True)
    rooms = [{"room_key": "hall", "room_type": "Hallway", "point": [5.25, 2.5, 1.0]},
             {"room_key": "living", "room_type": "Living Room", "point": [2.5, 3.0, 1.0]},
             {"room_key": "bed", "room_type": "Bedroom", "point": [8.0, 1.5, 1.0]}]
    plan, rep, _ = plan_tour_3d(m, SCENE, rooms, "test-home", PlannerConfig(), check_occlusion=False)
    assert [s.room_key for s in plan.shots] == ["living", "living", "hall", "bed"]
    assert [s.camera.move for s in plan.shots] == ["walkway", "walkway", "link", "enter"]
    link = rep["shots"][2]
    assert link["from"] == "living" and link["toward"] == "bed" and link["hall_share"] >= 0.3   # through it, the bedroom
    seg = plan.shots[2].camera.segments
    assert seg[0].eye[0][0] < 4.5 and seg[-1].target[-1][0] > 5.5     # stands in the living room, looks down the hallway
    enter = plan.shots[3].camera.segments
    assert enter[0].eye[0][0] < 6.0 < enter[-1].eye[-1][0]            # from the hallway, through the door
    assert all(s.transition_in.type == "cut" for s in plan.shots[2:])


def test_a_foyer_opening_reveals_the_next_room():
    """A tour that opens in a room with nothing to compose on (a foyer) moves toward the next room
    and ends with it filling the frame."""
    m = home(door=2.4, bed=False)
    rooms = [{"room_key": "foyer", "room_type": "Foyer", "point": [8.0, 2.5, 1.0]},
             {"room_key": "living", "room_type": "Living Room", "point": [3.0, 3.0, 1.0]}]
    plan, rep, _ = plan_tour_3d(m, SCENE, rooms, "test-home", PlannerConfig(), check_occlusion=False)
    assert plan.shots[0].room_key == "foyer" and plan.shots[0].camera.move == "reveal"
    assert rep["shots"][0]["reveals"] == "living" and rep["shots"][0]["share_next"] >= 0.35
    assert plan.shots[0].camera.segments[-1].target[-1][0] < 6.0       # it ends looking into the living room


def test_without_labels_there_are_no_close_ups(planned):
    plan, _, _ = planned
    assert not [s for s in plan.shots if s.role == "close"]
