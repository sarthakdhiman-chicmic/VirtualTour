"""Near-lens occlusion by rendering (review 2026-11-12: "nothing near the lens hiding the view").

The splats within `near` m of the camera - minus the floor and the framed piece - are drawn white on
black with the CPU reference renderer at 96 x 54. A pixel's brightness is how much that near stuff
hides of the view; cover = the share of pixels over 0.5 in the middle of the frame (CENTRE_X / Y).
Faint, smeared splats (a badly captured glass door) count with their real size and opacity, which a
floor-plan or opaque-only test misses."""
from __future__ import annotations

import logging
import os

import numpy as np

from tour_worker.scene.camera import intrinsics, look_at
from tour_worker.scene.splat_io import SH_C0, SplatModel

W, H = 96, 54
_WHITE = (1.0 - 0.5) / SH_C0


# what counts: the middle of the frame, where the subject is - a wall at the frame's edge or the ceiling
# above is part of a room, not something in the way
CENTRE_X, CENTRE_Y = (0.2, 0.8), (0.15, 0.85)
CEILING_GAP = 0.25
TOP_ROWS = 0.2                # the top of the frame a ceiling's black band would fill
VOID_SPLATS = 400_000      # the scene, thinned, for the void check
log = logging.getLogger(__name__)


class _GpuWhite:
    """The checks' white-on-black renders with gsplat on the GPU host (milliseconds each; the CPU
    reference renderer took seconds on a dense scan - Home Scan planning 31 min, 2026-11-18). The
    arrays go to the GPU once; a render draws the rows idx of them (None: all)."""

    def __init__(self, means, scales, quats, opacities):
        import torch
        from gsplat import rasterization
        self.torch, self._raster = torch, rasterization
        dev = torch.device("cuda")
        self.means = torch.from_numpy(np.ascontiguousarray(means, np.float32)).to(dev)
        self.scales = torch.from_numpy(np.ascontiguousarray(scales, np.float32)).to(dev)
        self.quats = torch.from_numpy(np.ascontiguousarray(quats, np.float32)).to(dev)
        self.opac = torch.from_numpy(np.ascontiguousarray(opacities, np.float32)).to(dev)
        self.white = torch.full((1, 1, 3), float(_WHITE), dtype=torch.float32, device=dev)

    def render(self, idx, viewmat, K, w: int, h: int) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            if idx is None:
                mean, scale, quat, op = self.means, self.scales, self.quats, self.opac
            else:
                i = t.from_numpy(np.asarray(idx, np.int64)).to(self.means.device)
                mean, scale, quat, op = self.means[i], self.scales[i], self.quats[i], self.opac[i]
            sh = self.white.expand(len(mean), 1, 3)
            v = t.from_numpy(np.asarray(viewmat, np.float32))[None].to(mean.device)
            k = t.from_numpy(np.asarray(K, np.float32))[None].to(mean.device)
            img, _, _ = self._raster(mean, quat, scale, op, sh, v, k, w, h, sh_degree=0, near_plane=0.05,
                                     far_plane=1e4, backgrounds=None, packed=False, rasterize_mode="classic")
            return img[0].float().clamp(0, 1).cpu().numpy()


def _use_gpu() -> bool:
    """gsplat on CUDA unless PLANNER_CHECK_RENDER=cpu (local runs on a GPU host that should match a
    CPU host's plan)."""
    if os.environ.get("PLANNER_CHECK_RENDER", "auto").lower() == "cpu":
        return False
    try:
        import torch
        import gsplat  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class Occlusion:
    def __init__(self, model: SplatModel, space):
        self.model, self.space = model, space
        self._gpu = None
        if _use_gpu():
            try:
                self._gpu = _GpuWhite(model.means, model.scales, model.quats, model.opacities)
            except Exception as e:                       # noqa: BLE001 - the CPU renderer still works
                log.warning("occlusion checks on the CPU: GPU setup failed (%s: %s)", type(e).__name__, e)
        self.backend = "gpu" if self._gpu is not None else "cpu"
        L = space.local
        self._keep = np.nonzero((L[:, 2] > 0.08) & (L[:, 2] < space.room_height - CEILING_GAP))[0]
        self._xy = L[self._keep, :2]
        self._area: tuple | None = None

    def focus(self, centre, radius: float) -> None:
        """Work on the splats within radius (m, on the floor plan) of centre - one room's worth."""
        sel = np.nonzero(np.linalg.norm(self._xy - np.asarray(centre, np.float32)[:2], axis=1) < radius)[0]
        self._area = (np.asarray(centre, np.float32)[:2], radius, self._keep[sel])

    def _candidates(self, eye, near):
        if self._area is not None:
            c, r, idx = self._area
            if np.linalg.norm(np.asarray(eye)[:2] - c) + near <= r:
                return idx
        return self._keep

    def cover(self, eye, target, hfov: float, piece=None, near: float = 2.0, region: str = "centre") -> float:
        """eye, target: planner points (z above the floor). region: "centre" (the subject's part of
        the frame) or "full" (a door frame at the edge counts too)."""
        from tour_worker.render.splat_cpu import CpuSplatRenderer
        idx = self._candidates(eye, near)
        L = self.space.local[idx]
        d = np.linalg.norm(L - np.asarray(eye, np.float32), axis=1)
        sel = d < near
        if piece is not None:                          # the piece itself is what we look at
            sel &= ~(piece.contains(L, 0.15) & (L[:, 2] < piece.top + 0.25))
        if not sel.any():
            return 0.0
        sel = idx[sel]
        e, t = self.space.to_scene(eye), self.space.to_scene(target)
        img = None
        if self._gpu is not None:
            try:
                img = self._gpu.render(sel, look_at(e, t, up=self.space.up_scene), intrinsics(hfov, W, H), W, H)
            except Exception as ex:                      # noqa: BLE001
                log.warning("occlusion checks on the CPU from now: GPU render failed (%s: %s)", type(ex).__name__, ex)
                self._gpu, self.backend = None, "cpu"
        if img is not None:
            if region == "full":
                return float((img.mean(2) > 0.5).mean())
            mid = img[int(CENTRE_Y[0] * H):int(CENTRE_Y[1] * H), int(CENTRE_X[0] * W):int(CENTRE_X[1] * W)]
            return float((mid.mean(2) > 0.5).mean())
        m = self.model
        sh = np.zeros((len(sel), 1, 3), np.float32)
        sh[:, 0] = _WHITE
        sub = SplatModel(m.means[sel], m.scales[sel], m.quats[sel], m.opacities[sel], sh, 0, m.source_format, {})
        e, t = self.space.to_scene(eye), self.space.to_scene(target)
        img = CpuSplatRenderer(sub).render(look_at(e, t, up=self.space.up_scene), intrinsics(hfov, W, H), W, H)
        if region == "full":
            return float((img.mean(2) > 0.5).mean())
        mid = img[int(CENTRE_Y[0] * H):int(CENTRE_Y[1] * H), int(CENTRE_X[0] * W):int(CENTRE_X[1] * W)]
        return float((mid.mean(2) > 0.5).mean())

    def void(self, eye, target, hfov: float, top: bool = False):
        """Share of the view the scan never captured (it renders as the empty black background): an
        uncaptured window or glass door, a missing ceiling. The whole scene, thinned to VOID_SPLATS
        with each kept splat grown to cover for the ones left out, drawn white at 96 x 54."""
        from tour_worker.render.splat_cpu import CpuSplatRenderer
        if getattr(self, "_void", None) is None:
            from tour_worker.scene.orient import opaque_cut
            m = self.model
            idx = np.nonzero(m.opacities > opaque_cut(m, 0.35))[0]
            k = max(1, int(np.ceil(len(idx) / VOID_SPLATS)))
            idx = idx[::k]
            sh = np.zeros((len(idx), 1, 3), np.float32)
            sh[:, 0] = _WHITE
            grow = np.float32(np.sqrt(k))
            sub = SplatModel(m.means[idx], m.scales[idx] * grow, m.quats[idx],
                             np.minimum(1.0, m.opacities[idx] * 1.5).astype(np.float32), sh, 0, m.source_format, {})
            self._void = CpuSplatRenderer(sub)
            self._void_gpu = None
            if self._gpu is not None:
                try:
                    self._void_gpu = _GpuWhite(sub.means, sub.scales, sub.quats, sub.opacities)
                except Exception as e:                   # noqa: BLE001
                    log.warning("void checks on the CPU: GPU setup failed (%s: %s)", type(e).__name__, e)
        e, t = self.space.to_scene(eye), self.space.to_scene(target)
        img = None
        if getattr(self, "_void_gpu", None) is not None:
            try:
                img = self._void_gpu.render(None, look_at(e, t, up=self.space.up_scene), intrinsics(hfov, W, H), W, H)
            except Exception as ex:                      # noqa: BLE001
                log.warning("void checks on the CPU from now: GPU render failed (%s: %s)", type(ex).__name__, ex)
                self._void_gpu = None
        if img is None:
            img = self._void.render(look_at(e, t, up=self.space.up_scene), intrinsics(hfov, W, H), W, H)
        dark = img.mean(2) < 0.35
        if top:                       # (whole view, its top TOP_ROWS: a missing ceiling's black band)
            return float(dark.mean()), float(dark[:max(1, int(round(TOP_ROWS * H)))].mean())
        return float(dark.mean())
