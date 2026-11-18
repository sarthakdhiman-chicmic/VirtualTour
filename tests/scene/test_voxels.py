"""Connected voxels without a dense grid (scene/voxels.py): the same pieces and the same grown cells
as the dense 3x3x3 grid it replaced - and a stray cell far away costs nothing."""
import numpy as np
from scipy import ndimage as ndi

from tour_worker.scene.voxels import components, near


def _dense_components(cells):
    lo = cells.min(0)
    grid = np.zeros(tuple(cells.max(0) - lo + 1), bool)
    grid[tuple((cells - lo).T)] = True
    comp, _ = ndi.label(grid, structure=np.ones((3, 3, 3), bool))
    return comp[tuple((cells - lo).T)]


def _same_partition(a, b):
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_components_match_the_dense_grid():
    rng = np.random.default_rng(3)
    for n, span in ((200, 12), (2000, 30), (5000, 25)):
        cells = rng.integers(0, span, size=(n, 3))
        assert _same_partition(components(cells), _dense_components(cells))


def test_near_matches_a_dense_dilation():
    rng = np.random.default_rng(5)
    src = rng.integers(0, 20, size=(150, 3))
    q = rng.integers(-3, 23, size=(3000, 3))
    lo = np.minimum(src.min(0), q.min(0)) - 1
    dims = np.maximum(src.max(0), q.max(0)) - lo + 2
    grid = np.zeros(tuple(dims), bool)
    grid[tuple((src - lo).T)] = True
    grid = ndi.binary_dilation(grid, structure=np.ones((3, 3, 3), bool))
    assert np.array_equal(near(q, src), grid[tuple((q - lo).T)])


def test_a_far_stray_cell_is_its_own_piece_and_cheap():
    # 0.05 m cells: a chair at the origin and one stray "chair" splat 70 m away and 20 m up - the
    # dense grid was 1400 x 1400 x 400 cells
    chair = np.array([[x, y, z] for x in range(10) for y in range(10) for z in range(8)])
    cells = np.vstack([chair, [[1400, 1400, 400]]])
    lab = components(cells)
    assert len(set(lab[:-1].tolist())) == 1 and lab[-1] != lab[0]
