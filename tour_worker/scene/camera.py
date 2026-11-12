"""Camera maths for 3D shots (plan 1.6 bezier_path).

Conventions: world = the scene file's units and axes. Camera = OpenCV: x right, y down, z forward.
A view matrix maps world points to camera points (x_c = R x_w + t); K is the pinhole intrinsics.

A shot's camera is an eye curve and a target curve (cubic Beziers, 1-3 joined segments). Both are
evaluated at one shared path parameter that is reparameterised by arc length, so the camera moves
at an even speed before easing and the beat-sync time_warp are applied:
    output progress t -> time_warp -> easing -> path fraction s -> (eye, target, hfov)
The arc length is the eye's travel plus TARGET_WEIGHT x the target's travel, so a pan in place
(the eye still, the target sweeping) is paced evenly too.
"""
from __future__ import annotations

import math

import numpy as np

from tour_worker.render.camera_path import time_warp_at

ARC_SAMPLES = 256          # per segment
TARGET_WEIGHT = 0.5
EASINGS = {"linear": lambda t: t, "ease_in": lambda t: t * t, "ease_out": lambda t: 1 - (1 - t) * (1 - t),
           "ease_in_out": lambda t: t * t * (3 - 2 * t)}


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 world-to-camera matrix (OpenCV axes) for a camera at eye looking at target, with the
    image's up as close to `up` as the view allows (a view straight along up keeps a stable roll)."""
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    z = target - eye
    n = np.linalg.norm(z)
    if n < 1e-9:
        raise ValueError("look_at: eye and target coincide")
    z /= n
    x = np.cross(z, up)                       # camera x (right) = forward x up
    if np.linalg.norm(x) < 1e-6:              # looking straight up / down: pick any right
        x = np.cross(z, np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                        # camera y (down)
    R = np.stack([x, y, z])                   # rows: camera axes in world coordinates
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = -R @ eye
    return m


def intrinsics(hfov_deg: float, width: int, height: int) -> np.ndarray:
    """3x3 pinhole K: square pixels, principal point at the image centre."""
    if not 1.0 <= hfov_deg <= 170.0:
        raise ValueError(f"hfov_deg {hfov_deg} out of range")
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def bezier(p: np.ndarray, u) -> np.ndarray:
    """Cubic Bezier through control points p (4, 3) at u (scalar or (M,)) -> (M, 3)."""
    u = np.atleast_1d(np.asarray(u, np.float64))[:, None]
    a = 1.0 - u
    return a ** 3 * p[0] + 3 * a * a * u * p[1] + 3 * a * u * u * p[2] + u ** 3 * p[3]


class BezierCameraPath:
    """A plan 1.6 bezier_path camera: segments [{"eye": 4 points, "target": 4 points}],
    hfov_deg [start, end], easing, time_warp."""

    def __init__(self, segments: list[dict], hfov_deg, easing: str = "ease_in_out", time_warp=None):
        self.eye = [np.asarray(s["eye"], np.float64) for s in segments]
        self.target = [np.asarray(s["target"], np.float64) for s in segments]
        if not self.eye or any(e.shape != (4, 3) for e in self.eye + self.target):
            raise ValueError("bezier_path segments need 4 3D points per curve")
        self.hfov = (float(hfov_deg[0]), float(hfov_deg[1]))
        self.ease = EASINGS.get(easing or "ease_in_out", EASINGS["ease_in_out"])
        self.time_warp = time_warp
        # arc-length table over the whole path: cumulative length at (segment, u) samples
        us = np.linspace(0.0, 1.0, ARC_SAMPLES + 1)
        lengths = [0.0]
        self._table = []                       # (segment, u) per sample, matching lengths
        for k, (e, t) in enumerate(zip(self.eye, self.target)):
            pe, pt = bezier(e, us), bezier(t, us)
            step = np.linalg.norm(np.diff(pe, axis=0), axis=1) + TARGET_WEIGHT * np.linalg.norm(np.diff(pt, axis=0), axis=1)
            if k == 0:
                self._table.append((0, 0.0))
            for j in range(ARC_SAMPLES):
                lengths.append(lengths[-1] + float(step[j]))
                self._table.append((k, float(us[j + 1])))
        self._lengths = np.asarray(lengths)
        self.length = float(self._lengths[-1])

    def path_fraction(self, t: float) -> float:
        """Output progress t (0-1 over the shot) -> fraction of the path's arc length."""
        t = min(1.0, max(0.0, float(t)))
        return float(self.ease(time_warp_at(self.time_warp, t)))

    def _locate(self, s: float) -> tuple[int, float]:
        """Path fraction -> (segment, u)."""
        if self.length < 1e-12:                 # a still camera
            return 0, s
        d = min(1.0, max(0.0, s)) * self.length
        i = int(np.searchsorted(self._lengths, d, side="right")) - 1
        i = min(max(i, 0), len(self._lengths) - 2)
        (k0, u0), (k1, u1) = self._table[i], self._table[i + 1]
        span = self._lengths[i + 1] - self._lengths[i]
        w = 0.0 if span <= 0 else (d - self._lengths[i]) / span
        if k1 != k0:                            # the first sample of a new segment
            return k1, u1 * w
        return k0, u0 + (u1 - u0) * w

    def pose(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        """(eye, target, hfov_deg) at output progress t."""
        s = self.path_fraction(t)
        k, u = self._locate(s)
        eye = bezier(self.eye[k], u)[0]
        target = bezier(self.target[k], u)[0]
        hfov = self.hfov[0] + (self.hfov[1] - self.hfov[0]) * s
        return eye, target, hfov

    def view(self, t: float, width: int, height: int, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
        """(4x4 view matrix, 3x3 K) at output progress t."""
        eye, target, hfov = self.pose(t)
        return look_at(eye, target, up), intrinsics(hfov, width, height)
