import os
import json
import subprocess
from modules.timeline import compute_timeline

def render(plan, workdir, scale=1.0, concurrency=2):
    # build props
    props_path = os.path.join(workdir, "props.json")
    out_path = os.path.join(workdir, "out.mp4")
    
    # Calculate frames
    timeline = compute_timeline(plan)
    shots = []
    fps = plan.output.fps
    
    for i, shot in enumerate(plan.shots):
        basename = shot.image_uri.split("/")[-1]
        
        shots.append({
            "id": shot.shot_id,
            "image": basename,
            "source": {"width": shot.source.width, "height": shot.source.height},
            "camera": {
                "start": shot.camera.start.model_dump(),
                "end": shot.camera.end.model_dump()
            }
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
        out_path,
        "--props", props_path,
        "--public-dir", images_dir,
        "--scale", str(scale),
        "--codec", plan.output.codec,
        "--pixel-format", "yuv420p",
        "--crf", "18",
        "--timeout", "600000",
        "--concurrency", str(concurrency),
        "--log", "verbose"
    ]
    
    try:
        res = subprocess.run(cmd, cwd=remotion_dir, check=True, capture_output=True, text=True, timeout=1800)
        print("REMOTION OUT:", res.stdout)
        print("REMOTION ERR:", res.stderr)
    except subprocess.TimeoutExpired:
        raise RuntimeError("timeout")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"render_failed: {e.stderr}")
        
    return out_path
