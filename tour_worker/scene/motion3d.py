"""3D motion for splat scenes: curtains that sway, plants that stir (plan 1.6 scene.motion_uri).

A motion file (made once per scene by tools/scene/label_motion.py, tour_worker/scene/motion_label.py)
lists the Gaussians that move and how. Each moving Gaussian gets TERMS sine terms; its offset at
tour time t (seconds) is

    offset(t) = sum_k  dir_k * amp_k * sin(2 pi freq_k t + phase_k)

added to its centre before the frame is rasterised. Time is the tour's clock, so the motion runs on
through cuts and transitions; the same plan gives the same frames. Only centres move (no rotation or
scale), which at these amplitudes (a few cm at a curtain's hem, ~1 cm at a leaf tip) is invisible as
such and keeps the splats' shapes intact.

The file is an .npz bound to its scene: `scene_sha256` and `splats` (the count after loading,
tour_worker/scene/splat_io) must match the scene it is used with, or the motion is skipped with a
warning - indices into a different scene would move the wrong Gaussians.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TERMS = 2
FORMAT_VERSION = 1
MAX_AMP_M = 0.10                 # a sanity bound: no term moves a Gaussian more than 10 cm


class MotionInvalid(ValueError):
    """The motion file is malformed (not: it belongs to another scene - that is a mismatch)."""


@dataclass
class MotionField:
    idx: np.ndarray        # (M,) int64 indices into the scene's Gaussians
    dirs: np.ndarray       # (M, TERMS, 3) float32 unit directions
    amp: np.ndarray        # (M, TERMS) float32 metres (scene units)
    freq: np.ndarray       # (M, TERMS) float32 Hz
    phase: np.ndarray      # (M, TERMS) float32 radians
    scene_sha256: str
    splats: int            # the scene's Gaussian count the indices refer to
    classes: dict          # class name -> count of moving Gaussians (provenance)

    def __len__(self) -> int:
        return int(self.idx.shape[0])

    def offsets(self, t: float) -> np.ndarray:
        """(M, 3) float32 offsets at tour time t (the CPU path; the GPU path is TorchMotion)."""
        s = np.sin(2.0 * math.pi * self.freq * np.float32(t) + self.phase).astype(np.float32)   # (M, T)
        return np.einsum("mk,mkc->mc", self.amp * s, self.dirs).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez_compressed(path, idx=self.idx.astype(np.int64), dirs=self.dirs.astype(np.float32),
                            amp=self.amp.astype(np.float32), freq=self.freq.astype(np.float32),
                            phase=self.phase.astype(np.float32),
                            meta=np.array([FORMAT_VERSION, self.splats], np.int64),
                            scene_sha256=np.array(self.scene_sha256),
                            class_names=np.array(list(self.classes)), class_counts=np.array(list(self.classes.values())))


def load_motion(path: str) -> MotionField:
    try:
        z = np.load(path, allow_pickle=False)
        idx, dirs, amp, freq, phase = (z[k] for k in ("idx", "dirs", "amp", "freq", "phase"))
        version, splats = (int(v) for v in z["meta"])
        sha = str(z["scene_sha256"])
        classes = {str(n): int(c) for n, c in zip(z["class_names"], z["class_counts"])}
    except Exception as e:                       # noqa: BLE001 - any read problem is a bad file
        raise MotionInvalid(f"motion file unreadable ({type(e).__name__}: {e})") from None
    m = idx.shape[0]
    if version != FORMAT_VERSION:
        raise MotionInvalid(f"motion file version {version} (expected {FORMAT_VERSION})")
    if (dirs.shape != (m, TERMS, 3) or amp.shape != (m, TERMS) or freq.shape != (m, TERMS)
            or phase.shape != (m, TERMS)):
        raise MotionInvalid("motion file arrays do not agree in shape")
    if m and (idx.min() < 0 or idx.max() >= splats or len(np.unique(idx)) != m):
        raise MotionInvalid("motion indices out of range or repeated")
    if not all(np.isfinite(a).all() for a in (dirs, amp, freq, phase)) or (np.abs(amp) > MAX_AMP_M).any():
        raise MotionInvalid("motion values not finite or amplitudes over the bound")
    return MotionField(idx.astype(np.int64), dirs.astype(np.float32), amp.astype(np.float32),
                       freq.astype(np.float32), phase.astype(np.float32), sha, splats, classes)


def check_binding(field: MotionField, scene_sha256: str, splats: int) -> str | None:
    """None when the field belongs to this scene, else why not (the job skips the motion)."""
    if field.scene_sha256 != scene_sha256:
        return "the motion file was made for another scene"
    if field.splats != splats:
        return f"the motion file expects {field.splats} Gaussians, the scene has {splats}"
    return None


class TorchMotion:
    """A MotionField on a torch device, applied in place to a means tensor: every call writes
    base + offset(t) for the moving Gaussians, so frames do not depend on the order they are drawn in."""

    def __init__(self, field: MotionField, means, sha256: str):
        import torch
        dev = means.device
        self.torch, self.sha256 = torch, sha256
        self.idx = torch.from_numpy(field.idx).to(dev)
        self.base = means[self.idx].clone()
        self.dirs = torch.from_numpy(field.dirs).to(dev)
        self.amp = torch.from_numpy(field.amp).to(dev)
        self.w = torch.from_numpy(2.0 * math.pi * field.freq).to(dev)
        self.phase = torch.from_numpy(field.phase).to(dev)

    def apply(self, means, t: float) -> None:
        s = self.torch.sin(self.w * float(t) + self.phase)                       # (M, T)
        off = self.torch.einsum("mk,mkc->mc", self.amp * s, self.dirs)
        means[self.idx] = self.base + off

    def restore(self, means) -> None:
        means[self.idx] = self.base


class MotionMixin:
    """attach_motion / set_time for a renderer that keeps its Gaussian centres in `self.means`
    (a torch tensor). Callers hold the renderer's lock (render/splat_shot.GPU_LOCK)."""
    _motion: "TorchMotion | None" = None

    @property
    def motion_sha256(self) -> str | None:
        return self._motion.sha256 if self._motion is not None else None

    def attach_motion(self, field: "MotionField | None", sha256: str | None = None) -> None:
        if self._motion is not None:
            self._motion.restore(self.means)
            self._motion = None
        if field is not None and len(field):
            self._motion = TorchMotion(field, self.means, sha256 or "")

    def set_time(self, t: float) -> None:
        if self._motion is not None:
            self._motion.apply(self.means, t)
