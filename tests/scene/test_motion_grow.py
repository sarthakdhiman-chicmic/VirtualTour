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
