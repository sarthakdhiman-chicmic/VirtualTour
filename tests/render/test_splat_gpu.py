"""gsplat on CUDA (run on the GPU host; skipped without CUDA): deterministic frames, and the same
picture as the CPU reference rasteriser (PSNR >= 45 dB on the box room)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from tests.scene_fixtures import box_room  # noqa: E402
from tour_worker.render.splat_cpu import CpuSplatRenderer  # noqa: E402
from tour_worker.scene.camera import intrinsics, look_at  # noqa: E402

W, H = 320, 180
VIEWS = [([0.3, -0.4, 1.1], [2, 1, 1.4]), ([-1.2, 1.0, 1.5], [1.5, -1.0, 0.8]), ([0, 0, 1.25], [0, 3, 1.25])]


@pytest.fixture(scope="module")
def renderers():
    gsplat = pytest.importorskip("gsplat")  # noqa: F841
    from tour_worker.render.splat_shot import GsplatRenderer
    m = box_room(sh_degree=1)
    m.sh[:, 3] = 0.3
    return GsplatRenderer(m), CpuSplatRenderer(m)


def _psnr(a, b):
    mse = float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def test_gsplat_frames_are_byte_identical_run_to_run(renderers):
    g, _ = renderers
    for eye, target in VIEWS:
        V, K = look_at(eye, target), intrinsics(70, W, H)
        assert g.render(V, K, W, H).tobytes() == g.render(V, K, W, H).tobytes()


def test_gsplat_matches_the_cpu_reference(renderers):
    g, c = renderers
    for eye, target in VIEWS:
        V, K = look_at(eye, target), intrinsics(70, W, H)
        assert _psnr(g.render(V, K, W, H), c.render(V, K, W, H)) >= 45.0


def test_the_renderer_records_its_provenance(renderers):
    g, _ = renderers
    assert g.info["backend"] == "gsplat" and g.info["gpu"] and g.info["gsplat_version"]
