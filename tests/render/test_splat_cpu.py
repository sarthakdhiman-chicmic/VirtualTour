"""The CPU reference rasteriser on a synthetic box room (CPU only, small images)."""
import numpy as np
import pytest

import tour_worker.render.splat_cpu as sc
from tests.scene_fixtures import FACE_RGB, box_room
from tour_worker.render.splat_cpu import CpuSplatRenderer
from tour_worker.scene.camera import intrinsics, look_at
from tour_worker.scene.splat_io import SH_C0

W, H = 160, 90


@pytest.fixture(scope="module")
def room():
    return CpuSplatRenderer(box_room())


def _view(r, eye, target, hfov=70):
    return r.render(look_at(eye, target), intrinsics(hfov, W, H), W, H)


def test_each_wall_shows_its_colour_and_up_is_up(room):
    for target, face in (([3, 0, 1.25], "+x"), ([-3, 0, 1.25], "-x"), ([0, 3, 1.25], "+y"), ([0, -3, 1.25], "-y")):
        img = _view(room, [0, 0, 1.25], target)
        np.testing.assert_allclose(img[H // 2, W // 2], FACE_RGB[face], atol=0.03)
    img = _view(room, [0, 0, 1.25], [3, 0, 1.25], hfov=110)
    np.testing.assert_allclose(img[1, W // 2], FACE_RGB["ceiling"], atol=0.05)    # top row: the ceiling
    np.testing.assert_allclose(img[-2, W // 2], FACE_RGB["floor"], atol=0.05)     # bottom row: the floor


def test_nearer_splats_hide_farther_ones(room):
    m = box_room()
    m.means = np.concatenate([m.means, [[1.0, 0, 1.25]]]).astype(np.float32)      # one big opaque blob
    m.scales = np.concatenate([m.scales, [[0.3, 0.3, 0.3]]]).astype(np.float32)
    m.quats = np.concatenate([m.quats, [[1, 0, 0, 0]]]).astype(np.float32)
    m.opacities = np.concatenate([m.opacities, [0.99]]).astype(np.float32)
    blob = np.zeros((1, 1, 3), np.float32)                                          # black
    blob[0, 0] = (0.0 - 0.5) / SH_C0
    m.sh = np.concatenate([m.sh, blob]).astype(np.float32)
    img = _view(CpuSplatRenderer(m), [0, 0, 1.25], [3, 0, 1.25])
    assert img[H // 2, W // 2].max() < 0.05


def test_renders_are_deterministic_and_chunking_does_not_change_them(room, monkeypatch):
    a = _view(room, [0.3, -0.4, 1.1], [2, 1, 1.4])
    b = _view(room, [0.3, -0.4, 1.1], [2, 1, 1.4])
    assert a.tobytes() == b.tobytes()
    monkeypatch.setattr(sc, "PAIR_BUDGET", 5000)
    monkeypatch.setattr(sc, "PROJECT_CHUNK", 7000)
    c = _view(room, [0.3, -0.4, 1.1], [2, 1, 1.4])
    np.testing.assert_allclose(a, c, atol=1e-5)


def test_outside_looking_away_is_background():
    r = CpuSplatRenderer(box_room(), background=(0.2, 0.4, 0.6))
    img = r.render(look_at([10, 0, 1], [20, 0, 1]), intrinsics(70, W, H), W, H)
    np.testing.assert_allclose(img, np.broadcast_to([0.2, 0.4, 0.6], img.shape), atol=1e-6)


def test_view_dependent_colour_changes_with_the_viewing_direction():
    m = box_room(sh_degree=1)
    m.sh[:, 3] = 1.0                       # an x-dependent term: brighter seen from one side
    r = CpuSplatRenderer(m)
    a = _view(r, [-1.5, 0, 1.25], [3, 0.0, 1.25])     # the +x wall seen looking +x
    b = _view(r, [-1.5, 0, 1.25], [3, 0.0, 1.25])
    assert a.tobytes() == b.tobytes()
    m2 = box_room(sh_degree=1)
    base = _view(CpuSplatRenderer(m2), [-1.5, 0, 1.25], [3, 0.0, 1.25])
    assert abs(float(a[H // 2, W // 2].mean() - base[H // 2, W // 2].mean())) > 0.1
