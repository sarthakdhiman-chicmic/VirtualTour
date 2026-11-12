"""3D shots (plan 1.6): a bezier_path camera flying through a Gaussian splat scene.

SplatShot has ShotWarp's frame interface (frame(f) -> BGR uint8 at the output size), so
render/direct.py composes 3D shots exactly like photo shots: fades, whip, push, overlays,
chunked encode, audio and join are unchanged.

Backends (SPLAT_BACKEND):
  gsplat  gsplat.rasterization on CUDA (Apache-2.0; LICENSES.md) - production
  cpu     the reference rasteriser (render/splat_cpu.py) - checking a scene without a GPU;
          seconds a frame, so only when asked for explicitly
  auto    gsplat when CUDA is available, else refuse (a CPU render of a whole tour would take hours)

Supersampling (SPLAT_SUPERSAMPLE, default 1.5): each frame is rasterised at 1.5x the output size and
reduced with INTER_AREA, which calms the shimmer of tiny splats as the camera moves.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from tour_worker.errors import WorkerError
from tour_worker.scene.camera import BezierCameraPath, intrinsics, look_at
from tour_worker.scene.splat_io import SplatModel

DEFAULT_UP = (0.0, 0.0, 1.0)


class GsplatRenderer:
    """gsplat on CUDA. Deterministic for a given GPU, driver and gsplat version (tested on the GPU host)."""

    def __init__(self, model: SplatModel, sh_degree: int | None = None, background=(0.0, 0.0, 0.0)):
        import torch
        import gsplat                                  # noqa: F401  (import errors surface here)
        from gsplat import rasterization
        self._raster = rasterization
        self.torch = torch
        dev = torch.device("cuda")
        self.degree = model.sh_degree if sh_degree is None else min(sh_degree, model.sh_degree)
        k = (self.degree + 1) ** 2
        self.means = torch.from_numpy(model.means).to(dev)
        self.quats = torch.from_numpy(model.quats).to(dev)
        self.scales = torch.from_numpy(model.scales).to(dev)
        self.opac = torch.from_numpy(model.opacities).to(dev)
        self.sh = torch.from_numpy(np.ascontiguousarray(model.sh[:, :k])).to(dev)
        self.bg = torch.tensor([background], dtype=torch.float32, device=dev)
        self.info = {"backend": "gsplat", "gsplat_version": getattr(gsplat, "__version__", "?"),
                     "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                     "driver": _nvidia_driver()}

    def render(self, viewmat: np.ndarray, K: np.ndarray, width: int, height: int) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            v = t.from_numpy(np.asarray(viewmat, np.float32))[None].cuda()
            k = t.from_numpy(np.asarray(K, np.float32))[None].cuda()
            img, _, _ = self._raster(self.means, self.quats, self.scales, self.opac, self.sh, v, k, width, height,
                                     sh_degree=self.degree, near_plane=0.05, far_plane=1e4,
                                     backgrounds=self.bg, packed=False, rasterize_mode="classic")
        return img[0].float().cpu().numpy()


def _nvidia_driver() -> str | None:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def make_renderer(model: SplatModel, backend: str = "auto", sh_degree: int | None = None):
    """The scene's renderer for this host. Raises WorkerError("scene_render_failed") when the
    asked-for backend can't run here."""
    if backend in ("auto", "gsplat"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            return GsplatRenderer(model, sh_degree)
        except Exception as e:
            raise WorkerError("scene_render_failed",
                              f"3D rendering needs gsplat on a CUDA GPU ({type(e).__name__}: {e}); "
                              f"SPLAT_BACKEND=cpu renders on the CPU, slowly") from None
    if backend == "cpu":
        from tour_worker.render.splat_cpu import CpuSplatRenderer
        r = CpuSplatRenderer(model, sh_degree)
        r.info = {"backend": "cpu"}
        return r
    raise WorkerError("scene_render_failed", f"unknown SPLAT_BACKEND {backend!r}")


class SplatShot:
    """Everything needed to draw any frame of one 3D shot."""

    def __init__(self, plan, shot, n: int, renderer, supersample: float = 1.5, stats: dict | None = None):
        cam = shot.camera
        self.shot, self.n = shot, n
        self.ow, self.oh = plan.output.width, plan.output.height
        self.ss = max(1.0, float(supersample))
        self.rw, self.rh = int(math.ceil(self.ow * self.ss)), int(math.ceil(self.oh * self.ss))
        self.renderer = renderer
        self.stats = stats if stats is not None else {}
        up = getattr(plan.scene, "up_axis", None) if getattr(plan, "scene", None) is not None else None
        self.up = tuple(up) if up else DEFAULT_UP
        self.path = BezierCameraPath([s.model_dump() for s in cam.segments], cam.hfov_deg,
                                     easing=getattr(cam.easing, "value", cam.easing), time_warp=cam.time_warp)

    def view(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        """(view matrix, K at the rasterised size) for frame f."""
        eye, target, hfov = self.path.pose(f / max(1, self.n - 1))
        return look_at(eye, target, self.up), intrinsics(hfov, self.rw, self.rh)

    def frame(self, f: int, step: int | None = None) -> np.ndarray:
        V, K = self.view(f)
        t0 = time.perf_counter()
        img = self.renderer.render(V, K, self.rw, self.rh)
        self.stats["rasterise_s"] = self.stats.get("rasterise_s", 0.0) + time.perf_counter() - t0
        self.stats["frames"] = self.stats.get("frames", 0) + 1
        if (self.rw, self.rh) != (self.ow, self.oh):         # reduce in float, round once
            img = cv2.resize(np.ascontiguousarray(img, np.float32), (self.ow, self.oh), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(np.clip(img[:, :, ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8))
