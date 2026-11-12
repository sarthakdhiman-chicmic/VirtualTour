"""Scene files: .ply (3DGS training output) and .splat load activated and validated."""
import numpy as np
import pytest

from tests.scene_fixtures import write_ply
from tour_worker.scene import SceneInvalid
from tour_worker.scene.splat_io import SH_C0, load_scene


def _ply(tmp_path, n=4, rest=45, **kw):
    rng = np.random.default_rng(0)
    data = dict(means=rng.normal(size=(n, 3)), f_dc=rng.normal(size=(n, 3)),
                f_rest=np.arange(n * rest, dtype=float).reshape(n, rest) if rest else np.zeros((n, 0)),
                opacity_logit=np.full(n, 2.0), log_scales=np.full((n, 3), np.log(0.02)),
                quats=np.tile([2.0, 0.0, 0.0, 0.0], (n, 1)))
    data.update(kw)
    p = tmp_path / "s.ply"
    write_ply(p, **data)
    return p, data


def test_ply_values_are_activated(tmp_path):
    p, d = _ply(tmp_path)
    m = load_scene(str(p))
    assert len(m) == 4 and m.sh_degree == 3 and m.sh.shape == (4, 16, 3)
    np.testing.assert_allclose(m.scales, 0.02, rtol=1e-5)
    np.testing.assert_allclose(m.opacities, 1 / (1 + np.exp(-2.0)), rtol=1e-6)
    np.testing.assert_allclose(m.quats, np.tile([1, 0, 0, 0], (4, 1)))          # normalised
    np.testing.assert_allclose(m.sh[:, 0], d["f_dc"], rtol=1e-6)
    np.testing.assert_allclose(m.base_rgb, np.clip(0.5 + SH_C0 * d["f_dc"], 0, 1), rtol=1e-6)


def test_f_rest_is_read_channel_major(tmp_path):
    p, d = _ply(tmp_path)
    m = load_scene(str(p))
    r = d["f_rest"]
    # coefficient j of channel c is f_rest[c * 15 + j]
    for j in (0, 7, 14):
        np.testing.assert_allclose(m.sh[:, j + 1], r[:, [j, 15 + j, 30 + j]], rtol=1e-6)


def test_sh_degree_can_be_capped_and_degree_0_files_load(tmp_path):
    p, _ = _ply(tmp_path)
    assert load_scene(str(p), max_sh_degree=1).sh.shape == (4, 4, 3)
    p0, _ = _ply(tmp_path, rest=0)
    m = load_scene(str(p0))
    assert m.sh_degree == 0 and m.sh.shape == (4, 1, 3)


def test_splat_files_load(tmp_path):
    rec = np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    a = np.zeros(3, rec)
    a["pos"] = [[0, 0, 0], [1, 2, 3], [-1, 0, 1]]
    a["scale"] = 0.05
    a["rgba"] = [255, 0, 128, 200]
    a["rot"] = [255, 128, 128, 128]
    p = tmp_path / "s.splat"
    a.tofile(p)
    m = load_scene(str(p))
    assert len(m) == 3 and m.sh_degree == 0
    np.testing.assert_allclose(m.base_rgb[0], [1.0, 0.0, 128 / 255], atol=1e-6)
    np.testing.assert_allclose(m.opacities, 200 / 255, rtol=1e-6)
    np.testing.assert_allclose(m.quats[0], [1, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(m.means[1], [1, 2, 3])


def test_bad_files_raise_scene_invalid(tmp_path):
    p, _ = _ply(tmp_path)
    raw = p.read_bytes()
    bad = tmp_path / "bad.ply"
    for content in (b"not a ply", raw.replace(b"binary_little_endian", b"ascii"),
                    raw.replace(b"property float opacity", b"property float opacityx"), raw[:-20]):
        bad.write_bytes(content)
        with pytest.raises(SceneInvalid):
            load_scene(str(bad))
    with pytest.raises(SceneInvalid):
        load_scene(str(p), max_splats=2)
    with pytest.raises(SceneInvalid):
        load_scene(str(tmp_path / "x.obj"))
    odd = tmp_path / "odd.splat"
    odd.write_bytes(b"\0" * 33)
    with pytest.raises(SceneInvalid):
        load_scene(str(odd))


def test_a_few_broken_or_faint_splats_are_dropped_and_many_are_fatal(tmp_path):
    n = 300
    means = np.zeros((n, 3))
    means[0] = np.nan                                             # 1 of 300: dropped
    op = np.full(n, 2.0)
    op[1] = -10.0                                                 # far below 1/255: faint
    p, _ = _ply(tmp_path, n=n, rest=0, means=means, opacity_logit=op)
    m = load_scene(str(p))
    assert len(m) == n - 2 and m.stats["splats_bad"] == 1 and m.stats["splats_faint"] == 1
    means[:10] = np.nan                                           # 10 of 300: over 1 %
    p, _ = _ply(tmp_path, n=n, rest=0, means=means)
    with pytest.raises(SceneInvalid):
        load_scene(str(p))
