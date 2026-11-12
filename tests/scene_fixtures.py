"""Synthetic splat scenes for tests: a box room (each face its own colour) and tiny files."""
import numpy as np

from tour_worker.scene.splat_io import SH_C0, SplatModel

# BGR-free plain RGB per face
FACE_RGB = {"+x": (0.9, 0.1, 0.1), "-x": (0.1, 0.9, 0.1), "+y": (0.1, 0.1, 0.9), "-y": (0.9, 0.9, 0.1),
            "floor": (0.5, 0.3, 0.1), "ceiling": (0.95, 0.95, 0.95)}


def box_room(half=2.0, height=2.5, spacing=0.04, scale=0.03, opacity=0.95, sh_degree=0) -> SplatModel:
    """A closed box room, z up, floor at z=0: about 45k splats at the defaults."""
    g = np.arange(-half + spacing / 2, half, spacing)
    h = np.arange(spacing / 2, height, spacing)
    pts, cols = [], []

    def add(p, face):
        pts.append(p)
        cols.append(np.tile(FACE_RGB[face], (len(p), 1)))
    a, b = np.meshgrid(g, h, indexing="ij")
    a, b = a.ravel(), b.ravel()
    add(np.stack([np.full_like(a, half), a, b], 1), "+x")
    add(np.stack([np.full_like(a, -half), a, b], 1), "-x")
    add(np.stack([a, np.full_like(a, half), b], 1), "+y")
    add(np.stack([a, np.full_like(a, -half), b], 1), "-y")
    a, b = np.meshgrid(g, g, indexing="ij")
    a, b = a.ravel(), b.ravel()
    add(np.stack([a, b, np.zeros_like(a)], 1), "floor")
    add(np.stack([a, b, np.full_like(a, height)], 1), "ceiling")
    means = np.concatenate(pts).astype(np.float32)
    rgb = np.concatenate(cols).astype(np.float32)
    n = len(means)
    k = (sh_degree + 1) ** 2
    sh = np.zeros((n, k, 3), np.float32)
    sh[:, 0] = (rgb - 0.5) / SH_C0
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1.0
    return SplatModel(means, np.full((n, 3), scale, np.float32), quats, np.full(n, opacity, np.float32),
                      sh, sh_degree, "ply", {"splats": n})


def write_ply(path, means, f_dc, f_rest, opacity_logit, log_scales, quats, extra_normals=True):
    """A 3DGS-style binary PLY (f_rest channel-major, as the trainer writes it)."""
    n = len(means)
    cols = [("x", means[:, 0]), ("y", means[:, 1]), ("z", means[:, 2])]
    if extra_normals:
        cols += [(f"n{c}", np.zeros(n)) for c in "xyz"]
    cols += [(f"f_dc_{i}", f_dc[:, i]) for i in range(3)]
    cols += [(f"f_rest_{i}", f_rest[:, i]) for i in range(f_rest.shape[1])]
    cols += [("opacity", opacity_logit)] + [(f"scale_{i}", log_scales[:, i]) for i in range(3)]
    cols += [(f"rot_{i}", quats[:, i]) for i in range(4)]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {name}\n" for name, _ in cols) + "end_header\n"
    body = np.stack([np.asarray(v, np.float32) for _, v in cols], 1).astype("<f4")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(body.tobytes())


def write_splat(path, model: SplatModel):
    """The model as a .splat file (32 bytes a splat; colour only)."""
    rec = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    a = np.zeros(len(model), rec)
    a["pos"] = model.means
    a["scale"] = model.scales
    rgb = np.clip(0.5 + SH_C0 * model.sh[:, 0], 0, 1)
    a["rgba"][:, :3] = np.round(rgb * 255)
    a["rgba"][:, 3] = np.round(np.clip(model.opacities, 0, 1) * 255)
    a["rot"] = np.clip(np.round(model.quats * 128 + 128), 0, 255)
    a.tofile(path)
