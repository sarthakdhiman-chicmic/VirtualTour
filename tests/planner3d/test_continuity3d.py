"""3D continuous cuts (client review 2026-11-20, items 4-6): the speed band trims a flight's start
(planner/speed_band.trim_start) and the 3D chooser reads each move's direction as the render
report does (planner3d/moves.move_motion)."""
import numpy as np

from tour_worker.planner.schemas import BezierSegment, CameraParams
from tour_worker.planner.speed_band import _scene_ends, _split, trim_start
from tour_worker.planner3d.moves import Move, _unit, move_motion
from tour_worker.scene.camera import BezierCameraPath, bezier


def _cam(path_s=6.0):
    eye = [[0, 0, 1.5], [1, 0, 1.5], [2, 0, 1.5], [3, 0, 1.5]]
    tgt = [[0, 5, 1.4], [1, 5, 1.4], [2, 5, 1.4], [3, 5, 1.4]]
    return CameraParams(type="bezier_path", easing="linear", move="truck", hfov_deg=[78.0, 78.0],
                        segments=[BezierSegment(eye=eye, target=tgt)], path_s=path_s)


def test_split_keeps_the_curve_exactly():
    p = np.array([[0, 0, 0], [1, 2, 0], [3, 2, 1], [4, 0, 1]], float)
    right = _split(p, 0.3)
    for v in (0.0, 0.5, 1.0):
        assert np.allclose(bezier(right, v)[0], bezier(p, 0.3 + 0.7 * v)[0])


def test_trimming_the_start_keeps_the_end_and_slows_the_move():
    cam = _cam()
    half = trim_start(cam, 0.5)
    end = BezierCameraPath([s.model_dump() for s in cam.segments], cam.hfov_deg).pose_at_fraction(1.0)
    end2 = BezierCameraPath([s.model_dump() for s in half.segments], half.hfov_deg).pose_at_fraction(1.0)
    assert np.allclose(end[0], end2[0]) and np.allclose(end[1], end2[1])
    a, b = _scene_ends(cam, 5.0, 30, (0, 0, 1)), _scene_ends(half, 5.0, 30, (0, 0, 1))
    assert abs(b[0] / a[0] - 0.5) < 0.05


def test_a_truck_reads_as_sideways_and_a_walk_in_as_forward():
    n = 50
    xs = np.linspace(0, 2, n)
    truck = Move("truck", np.c_[xs, np.zeros(n)], np.c_[xs, np.full(n, 5.0), np.full(n, 1.4)], 1.0)
    assert _unit(move_motion(truck, end=False)).direction() in ("right", "left")
    ys = np.linspace(0, 2, n)
    walk = Move("walkway", np.c_[np.zeros(n), ys], np.c_[np.zeros(n), np.full(n, 6.0), np.full(n, 1.4)], 1.0)
    assert _unit(move_motion(walk, end=True)).direction().startswith("forward")
