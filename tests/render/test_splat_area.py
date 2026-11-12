"""The GPU path's area reduction (render/splat_shot.area_reduce_bgr8) is OpenCV's INTER_AREA:
checked on CPU torch, so it runs without a GPU."""
import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tour_worker.render.splat_shot import area_reduce_bgr8, area_weights  # noqa: E402


def _cpu_path(img, ow, oh):
    if img.shape[:2] != (oh, ow):
        img = cv2.resize(np.ascontiguousarray(img, np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
    return np.clip(img[:, :, ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
def test_area_reduction_matches_opencv_inter_area(scale):
    ow, oh = 160, 90
    w, h = int(np.ceil(ow * scale)), int(np.ceil(oh * scale))
    rng = np.random.default_rng(int(scale * 100))
    img = rng.random((h, w, 3), dtype=np.float32)
    img[: h // 2, : w // 3] = np.linspace(0, 1, w // 3, dtype=np.float32)[None, :, None]   # smooth ramps too
    ours = area_reduce_bgr8(torch.from_numpy(img), ow, oh).numpy()
    ref = _cpu_path(img, ow, oh)
    diff = np.abs(ours.astype(int) - ref.astype(int))
    assert ours.shape == (oh, ow, 3) and ours.dtype == np.uint8
    assert diff.max() <= 1 and (diff > 0).mean() < 0.01      # float summation order only


def test_area_weights_rows_sum_to_one_and_cover_the_input():
    for n_in, n_out in ((135, 90), (2880, 1920), (100, 100)):
        w = area_weights(n_in, n_out)
        assert np.allclose(w.sum(1), 1.0) and np.allclose(w.sum(0) * n_in / n_out, 1.0)
