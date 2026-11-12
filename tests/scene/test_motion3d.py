"""3D motion (plan 1.6 scene.motion_uri): the motion file, its runtime on a renderer, the labeller."""
import math

import numpy as np
import pytest

from tests.scene_fixtures import box_room
from tour_worker.scene.camera import intrinsics, look_at
from tour_worker.scene.motion3d import (TERMS, MotionField, MotionInvalid, check_binding, load_motion)

torch = pytest.importorskip("torch")


def _field(n=10, splats=100, sha="ab" * 32):
    rng = np.random.default_rng(0)
    d = rng.normal(size=(n, TERMS, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return MotionField(np.arange(0, 2 * n, 2, dtype=np.int64), d, np.full((n, TERMS), 0.02, np.float32),
                       np.full((n, TERMS), 0.3, np.float32), rng.uniform(0, 6, (n, TERMS)).astype(np.float32),
                       sha, splats, {"curtain": n})


def test_the_motion_file_round_trips_and_is_checked(tmp_path):
    f = _field()
    p = str(tmp_path / "m.npz")
    f.save(p)
    g = load_motion(p)
    assert np.array_equal(g.idx, f.idx) and np.allclose(g.offsets(1.7), f.offsets(1.7)) and g.classes == {"curtain": 10}
    assert check_binding(g, "ab" * 32, 100) is None
    assert "another scene" in check_binding(g, "cd" * 32, 100)
    assert "Gaussians" in check_binding(g, "ab" * 32, 99)
    bad = _field(); bad.amp[0, 0] = 0.5                       # over the bound
    bad.save(p)
    with pytest.raises(MotionInvalid):
        load_motion(p)
    bad = _field(splats=5)                                    # indices past the scene
    bad.save(p)
    with pytest.raises(MotionInvalid):
        load_motion(p)


def test_offsets_are_the_documented_sum_of_sines():
    f = _field(n=3)
    t = 2.25
    want = sum(f.dirs[:, k] * (f.amp[:, k] * np.sin(2 * math.pi * f.freq[:, k] * t + f.phase[:, k]))[:, None]
               for k in range(TERMS))
    assert np.allclose(f.offsets(t), want, atol=1e-6)


def test_a_renderer_applies_the_motion_in_place_and_restores_it():
    from tour_worker.render.splat_cpu import CpuSplatRenderer
    m = box_room()
    r = CpuSplatRenderer(m)
    before = r.means.clone()
    f = _field(n=50, splats=len(m))
    r.attach_motion(f, "sha-1")
    assert r.motion_sha256 == "sha-1"
    r.set_time(3.0)
    moved = r.means[torch.from_numpy(f.idx)] - before[torch.from_numpy(f.idx)]
    assert np.allclose(moved.numpy(), f.offsets(3.0), atol=1e-6)
    r.set_time(3.0)                                           # same time, same positions: order does not matter
    assert np.allclose((r.means[torch.from_numpy(f.idx)] - before[torch.from_numpy(f.idx)]).numpy(), f.offsets(3.0), atol=1e-6)
    still = np.setdiff1d(np.arange(len(m)), f.idx)
    assert torch.equal(r.means[still], before[still])
    r.attach_motion(None)
    assert torch.equal(r.means, before) and r.motion_sha256 is None


def test_the_labeller_marks_the_masked_wall_and_anchors_its_top():
    """Box room, curtain masks over the +x wall seen from three places: that wall's Gaussians are
    labelled (and nothing else), still at the top, moving most at the bottom."""
    from tour_worker.scene.motion_label import CLASSES, View, build_field, label
    m = box_room()
    W, H = 160, 90
    views = []
    for eye in ([-1.0, -0.5, 1.2], [-1.0, 0.5, 1.4], [-0.5, 0.0, 1.0]):
        V, K = look_at(eye, [2.0, 0.0, 1.2]), intrinsics(80, W, H)
        # the +x wall fills the middle of these views: mask where a +x wall point projects
        pts = m.means[m.means[:, 0] > 1.95].astype(np.float64)
        cam = pts @ V[:3, :3].T + V[:3, 3]
        u = (K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int); v = (K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
        mask = np.zeros((H, W), bool)
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        mask[v[ok], u[ok]] = True
        views.append(View(V, K, W, H, {"curtain": mask}))
    lab = label(m, views)
    wall = m.means[:, 0] > 1.95
    assert (lab[m.means[:, 0] < 1.9] == 0).all()             # nothing beyond the wall's corner row (4 cm)
    assert (lab[wall] == CLASSES["curtain"]).mean() > 0.5
    f = build_field(m, lab, (0, 0, 1), "ab" * 32)
    z = m.means[f.idx, 2]
    amp = f.amp[:, 0]
    assert amp[z > 2.3].max() < 0.2 * amp.max()              # the rod barely moves
    assert amp[z < 0.3].min() > 0.8 * amp.max()               # the hem moves most
    assert f.classes == {"curtain": len(f)}
