"""Pre-warp shots that use projection "rotate" (addendum 1.3.1) into per-frame images.

Chrome's CSS matrix3d path made renders non-deterministic: two renders of the same plan differed
in ~300-400 of 2250 frames, including crop shots sharing a render chunk with a rotate shot
(out/tour_v5/report.md); will-change and per-frame remounting did not fix it. So the rotation is
warped here with OpenCV (exact homography from modules/camera_path.rotate_homography, linear
interpolation, which matches Chrome's sampling of crop shots: SSIM 0.96-0.97, same sharpness)
and the renderer only shows the finished frames.

Frame f of a shot with n rendered frames is the camera at t = f / (n - 1), as in KenBurnsShot.
"""
import os

import cv2
import numpy as np

from modules.camera_path import KeyframePath, rotate_homography, DEFAULT_HFOV_DEG
from modules.timeline import compute_timeline

PREWARP_DIR = "prewarp"
JPEG_QUALITY = 95


def frame_name(f: int) -> str:
    return f"{f:05d}.jpg"


def prewarp_rotate_shots(plan, workdir: str) -> dict:
    """Writes <workdir>/images/prewarp/<shot_id>/<frame>.jpg for every rotate shot. Returns
    {shot_id: {"dir": "prewarp/<shot_id>", "frames": n}} for the renderer's props."""
    from modules.plan_io import compound_path  # plan_io imports camera_path; avoid a cycle
    out = {}
    images = os.path.join(workdir, "images")
    for shot, tl in zip(plan.shots, compute_timeline(plan)):
        cam = shot.camera
        if cam.projection != "rotate" or cam.type not in ("compound", "keyframe_path"):
            continue
        n = tl["render_frames"]
        ow, oh = plan.output.width, plan.output.height
        path = (compound_path(shot, ow, oh) if cam.type == "compound"
                else KeyframePath([k.model_dump() for k in cam.keyframes]))
        src = cv2.imread(os.path.join(images, shot.image_uri.split("/")[-1]), cv2.IMREAD_COLOR)
        if src is None:
            raise RuntimeError(f"prewarp: cannot read the image for shot {shot.shot_id}")
        sh, sw = src.shape[:2]
        if (sw, sh) != (shot.source.width, shot.source.height):
            raise RuntimeError(f"prewarp: shot {shot.shot_id} image is {sw}x{sh}, plan says "
                               f"{shot.source.width}x{shot.source.height}")
        rel = f"{PREWARP_DIR}/{shot.shot_id}"
        os.makedirs(os.path.join(images, rel), exist_ok=True)
        hfov = cam.hfov_deg if cam.hfov_deg is not None else DEFAULT_HFOV_DEG
        for f, (cx, cy, z) in enumerate(path.frames(n)):
            m = np.array(rotate_homography(cx, cy, z, sw, sh, ow, oh, hfov))
            frame = cv2.warpPerspective(src, m, (ow, oh), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(os.path.join(images, rel, frame_name(f)), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        out[shot.shot_id] = {"dir": rel, "frames": n}
    return out
