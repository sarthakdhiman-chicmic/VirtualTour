import os
import shutil

import cv2
import numpy as np
import pytest

# feat/tour-v2: depth parallax is not in the render path (render() takes no depth maps); the
# parallax work and these tests live on park/m3-and-retune-2026-10-12
pytestmark = pytest.mark.skip(reason="parallax is out of the render path on feat/tour-v2")

from modules import depth
from modules.remotion_render import render
from tools.motion_lab import create_one_shot_plan, check_black_edges
from tests.test_motion import _generate_textured_image

needs_model = pytest.mark.skipif(
    not os.path.exists(depth.model_path()), reason="depth model not downloaded (modules.depth.ensure_model)"
)


def test_prepare_depth_map_shape_and_range():
    raw = np.zeros((100, 150), np.float32)
    raw[40:60, 60:90] = 1.0
    d = depth.prepare_depth_map(raw, 300, 200)
    assert d.shape == (200, 300)
    assert 0.0 <= d.min() and d.max() <= 1.0
    # dilation grows the near object, so its centre stays at full depth
    assert d[100, 150] == pytest.approx(1.0, abs=1e-3)


@needs_model
def test_build_depth_maps_writes_png_and_focus(tmp_path):
    img = str(tmp_path / "room.jpg")
    _generate_textured_image(1500, 1000, img)
    maps, warnings = depth.build_depth_maps(str(tmp_path), ["room.jpg", "room.jpg", "missing.jpg"])
    assert set(maps) == {"room.jpg"}
    assert any(w.startswith("depth_unavailable:missing.jpg") for w in warnings)
    out = cv2.imread(str(tmp_path / maps["room.jpg"]["image"]), cv2.IMREAD_GRAYSCALE)
    assert out is not None and max(out.shape) <= depth.DEPTH_MAP_LONG_EDGE
    assert 0.0 <= maps["room.jpg"]["focus"] <= 1.0


@pytest.mark.parametrize("preset", ["slide", "push_in_strong"])
def test_parallax_render_has_no_black_edges(tmp_path, preset):
    # a synthetic depth ramp is enough to exercise the WebGL path without the model
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img = str(images_dir / "room.jpg")
    _generate_textured_image(1500, 1000, img)
    ramp = np.tile(np.linspace(0, 255, 1500, dtype=np.float32), (1000, 1))
    ramp[300:700, 600:900] = 255  # a "near object"
    cv2.imwrite(str(images_dir / "room.depth.png"), ramp.astype(np.uint8))

    plan, _, _ = create_one_shot_plan(img, preset, 1500, 1000, duration_s=2.0)
    out = render(plan, str(tmp_path), scale=0.25,
                 depth_maps={"room.jpg": {"image": "room.depth.png", "focus": 0.5}})
    passed, min_lum = check_black_edges(out)
    assert passed, f"black edge in parallax render (min luminance {min_lum})"
    shutil.rmtree(images_dir)
