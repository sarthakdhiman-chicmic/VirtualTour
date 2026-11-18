"""Connected voxels without a dense grid: the scene's labelled Gaussians grouped into objects
(scene/semantics.find_objects, scene/motion_label.objects / flame_field) and labels grown into
neighbouring cells (motion_label.grow).

A dense boolean grid over a class's bounding box was the old way: one stray Gaussian labelled
"chair" 70 m from the real chairs made a 1400 x 1400 x 400 grid (0.8 GB) and its int32 component
map (3 GB) - a worker out of memory. These work on the occupied cells only (memory ~ the number of
Gaussians), with the same rule: cells touching by a face, edge or corner (26-connectivity) are one
piece. The results equal the dense grid's, so labels and objects are unchanged."""
from __future__ import annotations

import numpy as np

# the 13 neighbour offsets "after" a cell; with their opposites, all 26
_HALF = np.array([o for o in np.ndindex(3, 3, 3) if o > (1, 1, 1)], np.int64) - 1


def _keys(cells: np.ndarray, lo: np.ndarray, dims: np.ndarray) -> np.ndarray:
    c = cells - lo
    return (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2]


def components(cells: np.ndarray) -> np.ndarray:
    """(n, 3) integer cells -> (n,) component ids 0..k-1 (26-connected occupied cells)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cells = np.asarray(cells, np.int64)
    if not len(cells):
        return np.zeros(0, np.int64)
    lo = cells.min(0) - 1
    dims = cells.max(0) - lo + 2
    uniq, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    keys = _keys(uniq, lo, dims)                      # sorted: np.unique sorts rows lexicographically
    rows, cols = [], []
    for off in _HALF:
        nk = _keys(uniq + off, lo, dims)
        j = np.searchsorted(keys, nk)
        j = np.minimum(j, len(keys) - 1)
        hit = keys[j] == nk
        rows.append(np.flatnonzero(hit))
        cols.append(j[hit])
    r, c = np.concatenate(rows), np.concatenate(cols)
    m = len(uniq)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(m, m))
    _, lab = connected_components(g, directed=False)
    return lab[inv].astype(np.int64)


def near(query: np.ndarray, source: np.ndarray) -> np.ndarray:
    """(nq,) bool: each query cell is a source cell or one of its 26 neighbours (a 3x3x3 dilation
    of the source cells, looked up instead of drawn)."""
    query, source = np.asarray(query, np.int64), np.asarray(source, np.int64)
    out = np.zeros(len(query), bool)
    if not len(query) or not len(source):
        return out
    lo = np.minimum(source.min(0), query.min(0)) - 2
    dims = np.maximum(source.max(0), query.max(0)) - lo + 3
    keys = np.unique(_keys(source, lo, dims))
    for off in np.array(list(np.ndindex(3, 3, 3)), np.int64) - 1:
        nk = _keys(query + off, lo, dims)
        j = np.minimum(np.searchsorted(keys, nk), len(keys) - 1)
        out |= keys[j] == nk
    return out
