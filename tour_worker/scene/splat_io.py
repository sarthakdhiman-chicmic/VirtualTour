"""Gaussian splat scene files -> one SplatModel (plan 1.6 tours).

  .ply    the 3DGS training output (binary little-endian, one float32 property per column): x y z,
          f_dc_0..2 and f_rest_* (spherical-harmonic colour, degree 0-3), opacity (logit),
          scale_0..2 (log), rot_0..3 (quaternion w x y z). Other columns (nx ny nz) are skipped.
  .splat  the web-viewer format, 32 bytes a splat: position 3 x float32, scale 3 x float32
          (linear), RGBA 4 x uint8, rotation w x y z 4 x uint8 as (b - 128) / 128. Colour only,
          no view-dependent part.

Everything comes out activated (linear scales, opacity 0-1, unit quaternions) in the file's own
units and axes. Splats with non-finite values or a zero quaternion are dropped and counted; a file
with more than MAX_BAD_FRACTION of them, a wrong header, a truncated body or too many splats
raises SceneInvalid. Splats too faint to ever show (opacity < 1/255) are dropped too.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from tour_worker.cpus import usable_cpus
from tour_worker.scene import SceneInvalid

SH_C0 = 0.28209479177387814          # the degree-0 spherical-harmonic constant
MAX_SPLATS = 20_000_000              # ~5 GB of float32 at degree 3; a house scan is 1-8 M
MAX_BAD_FRACTION = 0.01
MIN_OPACITY = 1.0 / 255.0
CHUNK_ROWS = 400_000
LOAD_THREADS = 8                     # PLY chunks parsed at once
SPLAT_RECORD = 32
_SH_REST = {0: 0, 9: 1, 24: 2, 45: 3}      # f_rest count -> SH degree (3 colour channels)


@dataclass
class SplatModel:
    means: np.ndarray          # (N, 3) float32, scene units
    scales: np.ndarray         # (N, 3) float32, linear, scene units
    quats: np.ndarray          # (N, 4) float32, unit, w x y z
    opacities: np.ndarray      # (N,) float32, 0-1
    sh: np.ndarray             # (N, K, 3) float32, K = (sh_degree + 1) ** 2; sh[:, 0] is the base colour
    sh_degree: int
    source_format: str         # "ply" | "splat"
    stats: dict = field(default_factory=dict)   # counts read / dropped (provenance, warnings)

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @property
    def base_rgb(self) -> np.ndarray:
        """(N, 3) colour without the view-dependent part, 0-1."""
        return np.clip(0.5 + SH_C0 * self.sh[:, 0], 0.0, 1.0)


def load_scene(path: str, fmt: str | None = None, max_sh_degree: int = 3, max_splats: int = MAX_SPLATS) -> SplatModel:
    """The scene file at path. fmt: "ply" | "splat" (default: from the extension).
    max_sh_degree: keep the colour up to this degree (a preview needs 0-1; memory)."""
    fmt = fmt or os.path.splitext(path)[1].lstrip(".").lower()
    if fmt == "ply":
        return load_ply(path, max_sh_degree, max_splats)
    if fmt == "splat":
        return load_splat(path, max_splats)
    raise SceneInvalid(f"unsupported scene format {fmt!r} (ply or splat)")


def _read_ply_header(f) -> tuple[int, list[tuple[str, str]], int]:
    if f.readline() != b"ply\n":
        raise SceneInvalid("not a PLY file")
    count, props, in_vertex, size = None, [], False, 4
    while True:
        line = f.readline()
        size += len(line)
        if not line or size > 65536:
            raise SceneInvalid("PLY header has no end_header")
        parts = line.decode("ascii", "replace").split()
        if not parts:
            continue
        if parts[0] == "format" and parts[1:2] != ["binary_little_endian"]:
            raise SceneInvalid(f"PLY format {' '.join(parts[1:])!r}: only binary_little_endian is supported")
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
            elif count is not None and int(parts[2]) > 0:
                raise SceneInvalid(f"PLY has an extra element {parts[1]!r}")
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise SceneInvalid("PLY vertex list properties are not supported")
            props.append((parts[2], parts[1]))
        elif parts[0] == "end_header":
            break
    if count is None:
        raise SceneInvalid("PLY has no vertex element")
    return count, props, size


def load_ply(path: str, max_sh_degree: int = 3, max_splats: int = MAX_SPLATS) -> SplatModel:
    with open(path, "rb") as f:
        count, props, header = _read_ply_header(f)
        names = [n for n, _ in props]
        if any(t not in ("float", "float32") for _, t in props):
            raise SceneInvalid("PLY properties must all be float32")
        need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        missing = [n for n in need if n not in names]
        if missing:
            raise SceneInvalid(f"PLY is not a Gaussian splat file (missing {', '.join(missing)})")
        n_rest = sum(1 for n in names if n.startswith("f_rest_"))
        if n_rest not in _SH_REST:
            raise SceneInvalid(f"PLY has {n_rest} f_rest columns (expected 0, 9, 24 or 45)")
        if not 0 < count <= max_splats:
            raise SceneInvalid(f"PLY has {count} splats (1-{max_splats})")
        stride = len(names)
        if os.path.getsize(path) < header + count * stride * 4:
            raise SceneInvalid("PLY body is truncated")
        file_deg = _SH_REST[n_rest]
        deg = max(0, min(file_deg, max_sh_degree))
        k = (deg + 1) ** 2
        col = {n: i for i, n in enumerate(names)}
        # f_rest is stored channel-major: 15 R coefficients, then 15 G, then 15 B (degree 3)
        per = n_rest // 3
        rest_idx = [[col[f"f_rest_{c * per + j}"] for c in range(3)] for j in range(k - 1)]

        means = np.empty((count, 3), np.float32)
        scales = np.empty((count, 3), np.float32)
        quats = np.empty((count, 4), np.float32)
        opac = np.empty(count, np.float32)
        sh = np.empty((count, k, 3), np.float32)

    def chunk(a: int) -> None:
        """Rows a .. a+CHUNK_ROWS into their own slices (threads: numpy releases the GIL here)."""
        n = min(CHUNK_ROWS, count - a)
        rows = np.fromfile(path, dtype="<f4", count=n * stride, offset=header + a * stride * 4).reshape(n, stride)
        means[a:a + n] = rows[:, [col["x"], col["y"], col["z"]]]
        scales[a:a + n] = np.exp(rows[:, [col["scale_0"], col["scale_1"], col["scale_2"]]])
        quats[a:a + n] = rows[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]]
        opac[a:a + n] = 1.0 / (1.0 + np.exp(-rows[:, col["opacity"]]))
        sh[a:a + n, 0] = rows[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]]
        for j, idx in enumerate(rest_idx):
            sh[a:a + n, j + 1] = rows[:, idx]

    # each chunk writes only its own rows, so the result is the same however many threads read
    starts = range(0, count, CHUNK_ROWS)
    with ThreadPoolExecutor(max_workers=max(1, min(LOAD_THREADS, usable_cpus(), len(starts)))) as pool:
        list(pool.map(chunk, starts))
    return _finish(means, scales, quats, opac, sh, deg, "ply", {"file_sh_degree": file_deg})


def load_splat(path: str, max_splats: int = MAX_SPLATS) -> SplatModel:
    size = os.path.getsize(path)
    if size == 0 or size % SPLAT_RECORD:
        raise SceneInvalid(f".splat size {size} is not a multiple of {SPLAT_RECORD} bytes")
    count = size // SPLAT_RECORD
    if count > max_splats:
        raise SceneInvalid(f".splat has {count} splats (at most {max_splats})")
    rec = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    data = np.fromfile(path, dtype=rec)
    rgba = data["rgba"].astype(np.float32) / 255.0
    quats = (data["rot"].astype(np.float32) - 128.0) / 128.0
    sh = ((rgba[:, :3] - 0.5) / SH_C0)[:, None, :].astype(np.float32)
    return _finish(data["pos"].astype(np.float32), data["scale"].astype(np.float32), quats,
                   rgba[:, 3].copy(), sh, 0, "splat", {"file_sh_degree": 0})


def _finish(means, scales, quats, opac, sh, deg, fmt, stats) -> SplatModel:
    n = means.shape[0]
    qn = np.linalg.norm(quats, axis=1)
    ok = (np.isfinite(means).all(1) & np.isfinite(scales).all(1) & (scales > 0).all(1)
          & np.isfinite(quats).all(1) & (qn > 1e-8) & np.isfinite(opac) & np.isfinite(sh).all((1, 2)))
    bad = int(n - ok.sum())
    if bad > MAX_BAD_FRACTION * n:
        raise SceneInvalid(f"{bad} of {n} splats have non-finite or degenerate values")
    keep = ok & (opac >= MIN_OPACITY)
    if not keep.any():
        raise SceneInvalid("the scene has no visible splats")
    if not keep.all():
        means, scales, quats, opac, sh, qn = means[keep], scales[keep], quats[keep], opac[keep], sh[keep], qn[keep]
    quats = (quats / qn[:, None]).astype(np.float32)
    stats = dict(stats, splats_read=n, splats_bad=bad, splats_faint=int(ok.sum() - keep.sum()), splats=int(keep.sum()))
    return SplatModel(np.ascontiguousarray(means), np.ascontiguousarray(scales), quats,
                      np.ascontiguousarray(opac), np.ascontiguousarray(sh), deg, fmt, stats)
