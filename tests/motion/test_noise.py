"""tour_worker.motion.noise: seeded gradient noise."""

import numpy as np

from tour_worker.motion.noise import GradientNoise3D


def test_noise_is_seeded_bounded_and_smooth():
    a, b, c = GradientNoise3D(1), GradientNoise3D(1), GradientNoise3D(2)
    xs = np.linspace(0, 8, 400)
    assert np.array_equal(a(xs, xs * 0.3, 0.5), b(xs, xs * 0.3, 0.5))
    assert not np.array_equal(a(xs, xs * 0.3, 0.5), c(xs, xs * 0.3, 0.5))
    v = a(xs, xs * 0.7, xs * 0.2)
    assert np.abs(v).max() <= 1.0
    assert np.abs(np.diff(a(0.3, 0.4, np.linspace(0, 2, 2001)))).max() < 0.01   # continuous in time
