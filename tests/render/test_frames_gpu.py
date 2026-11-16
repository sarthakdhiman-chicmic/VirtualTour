"""Photo frames drawn with torch (render/frames_gpu.py) - the path the A100 runs - checked here on
torch's CPU device: the same frames as OpenCV's ShotWarp, for every effect, and a render through it
that is valid and byte-identical however many encoders run."""
import hashlib
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.perceptual_fixtures import _plan, _region, _shot, _workdir  # noqa: E402
from tour_worker.contract.plan import parse_plan  # noqa: E402
from tour_worker.render.frames_gpu import GpuShot, photo_device  # noqa: E402

CPU = torch.device("cpu")
EFFECTS = [("fabric_sway", "top", 5.0, 0.3, False), ("foliage_sway", "bottom", 4.0, 0.4, False),
           ("canopy_flutter", "bottom", 6.0, 1.1, True), ("calm_water", "none", 4.0, 0.2, False),
           ("water_ripple", "none", 4.0, 0.2, False)]


@pytest.mark.parametrize("effect,anchor,amp,freq,leafy", EFFECTS)
def test_gpu_frames_match_the_opencv_frames(effect, anchor, amp, freq, leafy):
    plan = parse_plan(_plan([_region(effect=effect, amp=amp, anchor=anchor, freq=freq)]))
    with tempfile.TemporaryDirectory() as td:
        _workdir(td, True, leafy=leafy)
        sw = _shot(plan, td)
        assert sw.regions, "the region must survive (visible motion)"
        g = GpuShot(sw, CPU)
        for f in (0, sw.n // 2, sw.n - 1):
            a, b = sw.frame(f).astype(np.int16), g.frame(f).astype(np.int16)
            d = np.abs(a - b)
            # bilinear sampling with exact weights vs OpenCV's 1/32 px fixed point: a level or two
            assert d.mean() < 0.6 and np.percentile(d, 99.9) <= 4 and d.max() <= 12, (effect, f, d.mean(), d.max())


def test_a_render_drawn_on_the_device_is_valid_and_byte_identical_for_any_encoder_count():
    from tour_worker.render.direct import render_direct
    from tour_worker.render.probe import verify
    from tests.render.test_direct import _plan as two_shots, _workdir as two_shot_images
    plan = two_shots()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        two_shot_images(a), two_shot_images(b)
        out_a = render_direct(plan, a, workers=3, photo_device=CPU)
        out_b = render_direct(plan, b, workers=1, photo_device=CPU)
        assert hashlib.md5(open(out_a, "rb").read()).hexdigest() == hashlib.md5(open(out_b, "rb").read()).hexdigest()
        verify(out_a, plan)


def test_the_cpu_setting_keeps_the_opencv_path():
    assert photo_device("cpu") is None
    if not torch.cuda.is_available():
        assert photo_device("auto") is None and photo_device("gpu") is None


def test_labels_and_the_mls_copy_through_the_device_path_match_the_cpu_render():
    """The frame stays on the device through composition (a fade) and the overlays (room labels):
    the master and the MLS copy decode to the CPU path's frames."""
    import subprocess
    from tour_worker.contract.plan import compute_plan_hash
    from tour_worker.render.direct import render_direct, total_frames
    from tests.render.test_direct import _plan as two_shots, _workdir as two_shot_images
    d = two_shots(fade=0.5).model_dump(mode="json")
    d["output"]["room_labels"] = {"font": "sacramento", "scale": 0.8, "placement": "bottom_left"}
    d["output"]["mls_copy"] = True
    d["shots"][0]["caption"], d["shots"][1]["caption"] = "Living Room", "Bedroom"
    d["provenance"]["plan_hash"] = compute_plan_hash(d)
    plan = parse_plan(d)
    n = total_frames(plan)

    def frames(path):
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(n, 360, 640, 3).astype(float)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        two_shot_images(a), two_shot_images(b)
        cpu = render_direct(plan, a, workers=1)
        gpu = render_direct(plan, b, workers=2, photo_device=CPU)
        for name in ("out.mp4", "out_mls.mp4"):
            x, y = frames(os.path.join(a, name)), frames(os.path.join(b, name))
            psnr = 10 * np.log10(255 ** 2 / max(1e-9, ((x - y) ** 2).mean()))
            assert psnr > 40, (name, psnr)
        assert os.path.exists(gpu)
