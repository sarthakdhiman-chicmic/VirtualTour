"""Drone clips as shots (SOW CM-04 "drone video used as shots", IN-01 "drone video" input) - the
renderer's building block, started 2026-11-17: a ClipShot is a shot's frame source, like ShotWarp
(photo) and SplatShot (3D), for a stretch of a video file:

  frame(f) -> (H, W, 3) uint8 BGR   the clip's frame at the tour's frame f of the shot

The clip is read once (ffmpeg: seek to in_s, the shot's length at the tour's fps, scaled and centre
cropped to the output size - a drone clip is wider or taller than 16:9 often - frame-rate converted
by duplicating / dropping frames, never blended), held in memory as the shot plays. An optional
slow push (zoom_end > 1) crops in on the centre across the shot. Deterministic: the same file,
in / out points and size give the same frames.

Still to come (the plan side): a plan version with a "video_clip" camera (clip uri, in / out,
zoom), the planner placing a listing's drone clips (the exterior / aerial opening and the
closing), and the backend accepting drone video uploads."""
from __future__ import annotations

import subprocess

import cv2
import numpy as np


class ClipShot:
    def __init__(self, path: str, in_s: float, n: int, width: int, height: int, fps: int,
                 zoom_end: float = 1.0):
        self.n, self.w, self.h = int(n), int(width), int(height)
        self.zoom_end = float(zoom_end)
        dur = self.n / float(fps)
        vf = (f"fps={fps},scale={self.w}:{self.h}:force_original_aspect_ratio=increase:flags=lanczos,"
              f"crop={self.w}:{self.h}")
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, in_s):.3f}", "-i", path, "-t", f"{dur:.3f}",
                              "-vf", vf, "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                             capture_output=True, check=True).stdout
        size = self.w * self.h * 3
        got = len(raw) // size
        if got == 0:
            raise ValueError(f"clip {path}: no frames from {in_s:.2f} s")
        frames = np.frombuffer(raw[:got * size], np.uint8).reshape(got, self.h, self.w, 3)
        if got < self.n:                     # a clip shorter than the shot holds its last frame
            frames = np.concatenate([frames, np.repeat(frames[-1:], self.n - got, axis=0)])
        self.frames = frames[:self.n]

    def frame(self, f: int) -> np.ndarray:
        img = self.frames[min(max(int(f), 0), self.n - 1)]
        if self.zoom_end <= 1.0 + 1e-6:
            return img.copy()
        t = f / max(self.n - 1, 1)
        z = 1.0 + (self.zoom_end - 1.0) * (t * t * (3 - 2 * t))
        cw, ch = self.w / z, self.h / z
        M = np.float32([[z, 0, -(self.w - cw) / 2 * z], [0, z, -(self.h - ch) / 2 * z]])
        return cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
