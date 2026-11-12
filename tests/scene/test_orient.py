"""Up, floor and ceiling from the geometry: any axis, either sign, tilted; an explicit up wins."""
import numpy as np
import pytest

from tests.scene_fixtures import box_room
from tour_worker.scene.orient import estimate_frame


def _furnished(sh_degree=0):
    """The box room plus furniture standing on the floor (a table, a sofa block)."""
    m = box_room(sh_degree=sh_degree)
    rng = np.random.default_rng(1)
    n = 12000
    furn = np.concatenate([rng.uniform([-1, -1, 0.0], [0.5, 0.5, 0.75], (n // 2, 3)),
                           rng.uniform([0.8, -1.8, 0.0], [1.8, 1.8, 0.9], (n // 2, 3))])
    m.means = np.concatenate([m.means, furn]).astype(np.float32)
    m.scales = np.concatenate([m.scales, np.full((n, 3), 0.03)]).astype(np.float32)
    m.quats = np.concatenate([m.quats, np.tile([1, 0, 0, 0], (n, 1))]).astype(np.float32)
    m.opacities = np.concatenate([m.opacities, np.full(n, 0.9)]).astype(np.float32)
    m.sh = np.concatenate([m.sh, np.zeros((n,) + m.sh.shape[1:])]).astype(np.float32)
    return m


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


@pytest.mark.parametrize("R", [np.eye(3), _rot([1, 0, 0], 180), _rot([1, 0, 0], -90), _rot([1, 0, 0], 90),
                               _rot([0.6, 0.8, 0], 20)])
def test_up_floor_and_ceiling_are_found_in_any_orientation(R):
    m = _furnished()
    m.means = (m.means @ R.T).astype(np.float32)
    f = estimate_frame(m)
    true_up = R @ [0, 0, 1.0]
    assert np.degrees(np.arccos(np.clip(f.up @ true_up, -1, 1))) < 3.0
    assert f.source == "geometry" and f.confident
    assert abs(f.floor - 0.0) < 0.1 and abs(f.ceiling - 2.5) < 0.1


def test_an_explicit_up_wins():
    m = _furnished()
    f = estimate_frame(m, up_hint=[0, 0, -2])
    assert np.allclose(f.up, [0, 0, -1]) and f.source == "given"
    assert abs(f.floor - (-2.5)) < 0.1 and abs(f.ceiling - 0.0) < 0.1      # heights along the given up


def test_an_empty_box_falls_back_to_the_file_convention_and_says_so():
    f = estimate_frame(box_room())                     # no furniture: no sign cue
    assert np.allclose(f.up, [0, 0, 1], atol=1e-6) and not f.confident and "scene_up_uncertain" in f.warnings
    m = box_room()
    m.means = (m.means @ _rot([1, 0, 0], 90).T).astype(np.float32)   # z up -> -y up (COLMAP-like)
    f = estimate_frame(m)
    assert np.allclose(f.up, [0, -1, 0], atol=1e-6) and f.source == "convention"
