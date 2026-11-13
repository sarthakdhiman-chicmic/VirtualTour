"""Near-lens occlusion by rendering (review 2026-11-12: "nothing near the lens hiding the view").

The splats within `near` m of the camera - minus the floor and the framed piece - are drawn white on
black with the CPU reference renderer at 96 x 54. A pixel's brightness is how much that near stuff
hides of the view; cover = the share of pixels over 0.5 in the middle of the frame (CENTRE_X / Y).
Faint, smeared splats (a badly captured glass door) count with their real size and opacity, which a
floor-plan or opaque-only test misses."""
from __future__ import annotations

import numpy as np

from tour_worker.scene.camera import intrinsics, look_at
from tour_worker.scene.splat_io import SH_C0, SplatModel

W, H = 96, 54
_WHITE = (1.0 - 0.5) / SH_C0


# what counts: the middle of the frame, where the subject is - a wall at the frame's edge or the ceiling
# above is part of a room, not something in the way
CENTRE_X, CENTRE_Y = (0.2, 0.8), (0.15, 0.85)
CEILING_GAP = 0.25


class Occlusion:
    def __init__(self, model: SplatModel, space):
        self.model, self.space = model, space
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

    def cover(self, eye, target, hfov: float, piece=None, near: float = 2.0) -> float:
        """eye, target: planner points (z above the floor)."""
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
        m = self.model
        sh = np.zeros((len(sel), 1, 3), np.float32)
        sh[:, 0] = _WHITE
        sub = SplatModel(m.means[sel], m.scales[sel], m.quats[sel], m.opacities[sel], sh, 0, m.source_format, {})
        e, t = self.space.to_scene(eye), self.space.to_scene(target)
        img = CpuSplatRenderer(sub).render(look_at(e, t, up=self.space.up_scene), intrinsics(hfov, W, H), W, H)
        mid = img[int(CENTRE_Y[0] * H):int(CENTRE_Y[1] * H), int(CENTRE_X[0] * W):int(CENTRE_X[1] * W)]
        return float((mid.mean(2) > 0.5).mean())
