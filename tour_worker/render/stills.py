"""Instagram feed stills (SOW EX-03: "Instagram Feed - JPEG still (hero images)"): one JPEG per room,
taken from the rendered tour - the frame late in the room's lead shot, once its move has settled -
cropped to the feed's 4:5 portrait and scaled to FEED_SIZE. From the MLS copy when the render made
one (no captions or branding on it), else from the master. Deterministic: the same video, the same
frames, the same bytes."""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np

FEED_SIZE = (1080, 1350)      # Instagram feed portrait, 4:5
AT = 0.8                      # how far into the lead shot the still is taken (the move settled)
QUALITY = 92


def still_times(plan) -> list[tuple[str, str | None, float]]:
    """[(room_key, room_type, seconds)]: each room once, at AT of its first shot."""
    out, seen, t = [], set(), 0.0
    for s in plan.shots:
        if s.room_key not in seen and s.role != "close":
            seen.add(s.room_key)
            out.append((s.room_key, s.room_type, round(t + AT * s.duration_s, 3)))
        t += s.duration_s
    return out


def _frame(video: str, t: float) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-ss", f"{t:.3f}", "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"no frame at {t:.3f} s")
    return img


def crop_feed(img: np.ndarray, size=FEED_SIZE) -> np.ndarray:
    """The frame's centre, cut to the feed's aspect, scaled to size."""
    h, w = img.shape[:2]
    tw, th = size
    if w / h > tw / th:                    # wider than 4:5: keep the full height
        cw = int(round(h * tw / th))
        x0 = (w - cw) // 2
        img = img[:, x0:x0 + cw]
    else:
        ch = int(round(w * th / tw))
        y0 = (h - ch) // 2
        img = img[y0:y0 + ch]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def feed_stills(video: str, plan, out_dir: str, size=FEED_SIZE) -> list[dict]:
    """Write the stills into out_dir; [{path, name, room_key, room_type, t}] in tour order."""
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for n, (room, rtype, t) in enumerate(still_times(plan), start=1):
        name = f"{n:02d}_{room}.jpg"
        path = os.path.join(out_dir, name)
        ok, buf = cv2.imencode(".jpg", crop_feed(_frame(video, t), size), [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        if not ok:
            raise ValueError(f"could not encode {name}")
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        out.append({"path": path, "name": name, "room_key": room, "room_type": rtype, "t": t})
    return out
