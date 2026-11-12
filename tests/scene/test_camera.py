"""Camera maths: OpenCV look_at, intrinsics, Bezier paths at an even speed."""
import numpy as np

from tour_worker.scene.camera import BezierCameraPath, bezier, intrinsics, look_at


def _project(V, K, w):
    c = V[:3, :3] @ np.asarray(w, float) + V[:3, 3]
    return (K @ c / c[2])[:2], c[2]


def test_look_at_uses_opencv_axes_with_world_up_on_top():
    V, K = look_at([0, 0, 1], [3, 0, 1]), intrinsics(70, 640, 360)
    (u, v), z = _project(V, K, [3, 0, 1])
    assert abs(u - 320) < 1e-6 and abs(v - 180) < 1e-6 and z > 0          # the target is centred, in front
    (u, v), _ = _project(V, K, [3, 0, 2])
    assert v < 180                                                        # higher in the world: higher on screen
    (u, v), _ = _project(V, K, [3, 1, 1])
    assert u < 320                                                        # +y is left when looking along +x (z up)
    assert np.allclose(V[:3, :3] @ V[:3, :3].T, np.eye(3)) and np.isclose(np.linalg.det(V[:3, :3]), 1.0)


def test_look_at_survives_looking_straight_down():
    V = look_at([0, 0, 2], [0, 0, 0])
    assert np.allclose(V[:3, :3] @ V[:3, :3].T, np.eye(3))


def test_intrinsics_put_the_hfov_edge_on_the_image_edge():
    K = intrinsics(90, 800, 450)
    u = K[0, 0] * np.tan(np.radians(45)) + K[0, 2]
    assert abs(u - 800) < 1e-9


def _seg(e, t):
    return {"eye": e, "target": t}


def test_the_camera_moves_at_an_even_speed_whatever_the_control_points():
    # control points bunched at the start: a naive u would crawl, then rush
    eye = [[0, 0, 1], [0.05, 0, 1], [0.1, 0, 1], [4, 0, 1]]
    p = BezierCameraPath([_seg(eye, [[0, 5, 1]] * 4)], [70, 70], easing="linear")
    pos = np.array([p.pose(t)[0] for t in np.linspace(0, 1, 41)])
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    assert steps.max() / steps.min() < 1.05
    assert np.allclose(pos[0], eye[0]) and np.allclose(pos[-1], eye[3])


def test_a_pan_in_place_is_paced_by_the_target():
    p = BezierCameraPath([_seg([[0, 0, 1]] * 4, [[3, -2, 1], [3, -1.9, 1], [3, -1.8, 1], [3, 2, 1]])], [70, 70], easing="linear")
    tg = np.array([p.pose(t)[1] for t in np.linspace(0, 1, 21)])
    steps = np.linalg.norm(np.diff(tg, axis=0), axis=1)
    assert steps.max() / steps.min() < 1.05


def test_segments_join_hfov_interpolates_and_easing_and_time_warp_apply():
    a = _seg([[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1]], [[0, 5, 1]] * 4)
    b = _seg([[3, 0, 1], [4, 0, 1], [5, 0, 1], [6, 0, 1]], [[0, 5, 1]] * 4)
    p = BezierCameraPath([a, b], [60, 80], easing="linear")
    assert np.allclose(p.pose(0.5)[0], [3, 0, 1], atol=1e-3) and abs(p.pose(0.5)[2] - 70) < 1e-6
    eased = BezierCameraPath([a, b], [60, 80], easing="ease_in_out")
    assert eased.pose(0.1)[0][0] < p.pose(0.1)[0][0]                       # slow start
    warped = BezierCameraPath([a, b], [60, 80], easing="linear", time_warp=[[0, 0], [0.5, 0.8], [1, 1]])
    assert abs(warped.path_fraction(0.5) - 0.8) < 1e-6
    for t in (0, 1):
        assert np.allclose(bezier(np.array(a["eye"], float), t)[0], a["eye"][3 * t])


def test_a_still_camera_is_allowed():
    p = BezierCameraPath([_seg([[1, 1, 1]] * 4, [[2, 1, 1]] * 4)], [70, 70])
    assert np.allclose(p.pose(0.7)[0], [1, 1, 1])
