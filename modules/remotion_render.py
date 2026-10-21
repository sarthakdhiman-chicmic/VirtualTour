import os
import json
import subprocess
from modules.encode import MASTER_CRF, deliver
from modules.prewarp import prewarp_rotate_shots
from modules.timeline import compute_timeline

def render(plan, workdir, scale=1.0, concurrency=2, jpeg_quality=95, gl=None, warnings=None):
    """jpeg_quality: frame capture quality, None = Remotion's default (80). gl: Chrome GL backend
    (--gl), None = Remotion's default. Tour v2 renders use jpeg_quality=None, gl=None (the liked
    baseline's sharpness matches default JPEG quality). Passing --gl angle or swangle made two
    renders of the same plan differ (out/tour_v2/report.md), so leave gl unset."""
    # build props
    props_path = os.path.join(workdir, "props.json")
    master_path = os.path.join(workdir, "master.mp4")   # Remotion's render (modules/encode.py)
    out_path = os.path.join(workdir, "out.mp4")
    
    # Calculate frames
    timeline = compute_timeline(plan)
    # rotate shots are warped to frames outside the browser (modules/prewarp.py: Chrome's 3D path
    # was not deterministic)
    prewarp = prewarp_rotate_shots(plan, workdir, warnings=warnings)
    shots = []
    fps = plan.output.fps
    
    for i, shot in enumerate(plan.shots):
        basename = shot.image_uri.split("/")[-1]
        
        shots.append({
            "id": shot.shot_id,
            "image": basename,
            "source": {"width": shot.source.width, "height": shot.source.height},
            "camera": ({
                "type": "compound",
                "move": shot.camera.move,
                "start": {k: v for k, v in shot.camera.start.model_dump().items() if k in ("cx", "cy", "zoom")},
                "segments": [x.model_dump() for x in shot.camera.segments],
                "timing": shot.camera.timing.model_dump(exclude_none=True),
                **({"projection": shot.camera.projection} if shot.camera.projection else {}),
                **({"hfov_deg": shot.camera.hfov_deg} if shot.camera.hfov_deg is not None else {}),
            } if shot.camera.type == "compound" else {
                "type": "keyframe_path",
                "move": shot.camera.move,
                "keyframes": [k.model_dump(exclude_none=True) for k in shot.camera.keyframes],
                **({"projection": shot.camera.projection} if shot.camera.projection else {}),
                **({"hfov_deg": shot.camera.hfov_deg} if shot.camera.hfov_deg is not None else {}),
            } if shot.camera.type == "keyframe_path" else {
                "type": "ken_burns",
                "start": shot.camera.start.model_dump(),
                "end": shot.camera.end.model_dump(),
                "easing": shot.camera.easing
            }),
            **({"prewarp": prewarp[shot.shot_id]} if shot.shot_id in prewarp else {}),
        })
        
    total_frames = round((plan.total_duration_s * 100) * fps / 100)
        
    props = {
        "shots": shots,
        "timeline": timeline,
        "fps": fps,
        "width": plan.output.width,
        "height": plan.output.height,
        "total_frames": total_frames
    }
    
    with open(props_path, "w") as f:
        json.dump(props, f)
        
    # run remotion
    remotion_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "remotion")
    images_dir = os.path.join(workdir, "images")
    
    cmd = [
        "npx", "remotion", "render",
        "src/index.ts", "TourComposition",
        master_path,
        "--props", props_path,
        "--public-dir", images_dir,
        "--scale", str(scale),
        "--codec", plan.output.codec,
        "--pixel-format", "yuv420p",
        "--crf", str(MASTER_CRF),
        # frames are captured as JPEG before encoding; the default quality (80) adds
        # per-frame compression noise that crawls on fine texture as the camera moves
        "--timeout", "600000",
        "--concurrency", str(concurrency),
        "--log", "verbose"
    ]
    if jpeg_quality is not None:
        cmd += ["--jpeg-quality", str(jpeg_quality)]
    if gl is not None:
        cmd += ["--gl", gl]
    try:
        res = subprocess.run(cmd, cwd=remotion_dir, check=True, capture_output=True, text=True, timeout=1800)
        print("REMOTION OUT:", res.stdout)
        print("REMOTION ERR:", res.stderr)
    except subprocess.TimeoutExpired:
        raise RuntimeError("timeout")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"render_failed: {e.stderr}")

    # delivery encode: capped bitrate, 2 s keyframes, faststart (smooth playback; modules/encode.py)
    return deliver(master_path, out_path, plan.output.fps)
