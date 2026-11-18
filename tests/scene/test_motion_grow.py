"""Curtain labels grown into the Gaussians no view saw (scene/motion_label.grow): the back of the
curtain takes them, the wall a few cm behind it does not."""
import numpy as np

from tour_worker.scene.motion_label import CLASSES, grow
from tour_worker.scene.splat_io import SplatModel


def _model(pts):
    n = len(pts)
    return SplatModel(np.asarray(pts, np.float32), np.full((n, 3), 0.01, np.float32),
                      np.tile(np.array([[1, 0, 0, 0]], np.float32), (n, 1)), np.ones(n, np.float32),
                      np.zeros((n, 1, 3), np.float32), 0, "ply", {})


def test_growth_takes_the_curtains_back_not_the_wall():
    xs, ys = np.meshgrid(np.linspace(0, 0.3, 7), np.linspace(1.0, 1.3, 7))
    plane = lambda z: np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, z)]     # noqa: E731
    curtain, back, by_wall, wall = plane(2.0), plane(2.015), plane(2.035), plane(2.05)
    pts = np.vstack([curtain, back, by_wall, wall])
    n = len(curtain)
    lab = np.zeros(len(pts), np.int8)
    lab[:n] = CLASSES["curtain"]
    seen = np.r_[np.ones(n), np.zeros(2 * n), np.ones(n)].astype(np.int32)
    rigid = np.zeros(len(pts), bool)
    rigid[3 * n:] = True
    old = grow(_model(pts), lab, seen)
    new = grow(_model(pts), lab, seen, rigid=rigid)
    assert (old[n:3 * n] == CLASSES["curtain"]).all()           # before: the wall's side too
    assert (new[n:2 * n] == CLASSES["curtain"]).all()           # the curtain's back
    assert (new[2 * n:3 * n] == 0).all()                        # nearer the wall: still


def test_flutter_and_flame_noise_is_smooth_and_deterministic():
    """Neighbouring Gaussians get nearly the same phase (a value per 4 cm cell made them jump out
    of step); far apart they differ; the same points give the same values."""
    import math
    from tour_worker.scene.motion_label import FLUTTER_SCALE, _smooth01, _smooth_angle
    rng = np.random.default_rng(2)
    p = rng.uniform(-1, 1, size=(4000, 3))
    q = p + rng.normal(scale=0.005, size=p.shape)                 # 5 mm away
    a, b = _smooth_angle(p, FLUTTER_SCALE, 13.0), _smooth_angle(q, FLUTTER_SCALE, 13.0)
    near = np.abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    assert np.percentile(near, 99) < 0.5                         # (old 4 cm cells: 27% of these
    #                                                              pairs had unrelated phases)
    far = np.abs((a - np.roll(a, 1) + math.pi) % (2 * math.pi) - math.pi)
    assert np.median(far) > 0.8                                    # it still varies across the plant
    assert np.array_equal(_smooth01(p, 0.15, 1.0), _smooth01(p.copy(), 0.15, 1.0))
