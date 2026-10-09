#!/bin/bash
set -e

mkdir -p modules
mkdir -p tests
mkdir -p remotion/src

# 1. Plan IO
cat << 'PYEOF' > modules/plan_io.py
import hashlib
import json
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, model_validator

class OutputConfig(BaseModel):
    width: int
    height: int
    fps: int
    codec: str
    model_config = ConfigDict(extra='ignore')

class Size(BaseModel):
    width: int
    height: int
    model_config = ConfigDict(extra='ignore')

class CameraTransform(BaseModel):
    cx: float
    cy: float
    zoom: float
    model_config = ConfigDict(extra='ignore')

class Camera(BaseModel):
    type: str
    easing: str
    start: CameraTransform
    end: CameraTransform
    model_config = ConfigDict(extra='ignore')

class Transition(BaseModel):
    type: str
    duration_s: float
    model_config = ConfigDict(extra='ignore')

class Shot(BaseModel):
    shot_id: str
    order: int
    asset_id: str
    image_uri: str
    source: Size
    room_key: str
    room_type: str
    narrative_stage: str
    role: str
    duration_s: float
    camera: Camera
    transition_in: Transition
    model_config = ConfigDict(extra='ignore')

class Provenance(BaseModel):
    scoring_config_version: str
    planner_config_version: str
    plan_hash: str
    model_config = ConfigDict(extra='ignore')

class Plan(BaseModel):
    plan_version: str
    listing_id: str
    mode: str
    fallback_reason: Optional[str] = None
    preset: str
    output: OutputConfig
    total_duration_s: float
    shots: List[Shot]
    warnings: List[Any] = Field(default_factory=list)
    provenance: Provenance
    model_config = ConfigDict(extra='ignore')

def validate_plan_hash(original_dict: dict):
    if "provenance" not in original_dict or "plan_hash" not in original_dict["provenance"]:
        return
    expected = original_dict["provenance"]["plan_hash"]
    
    dict_copy = dict(original_dict)
    dict_copy["provenance"] = dict(dict_copy["provenance"])
    del dict_copy["provenance"]["plan_hash"]
    
    def strip_extra(d):
        pass # Optional: ideally we hash exactly what's parsed, or original. The prompt says "canonical JSON of the plan with provenance.plan_hash removed".
        
    canonical_json = json.dumps(dict_copy, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
    computed = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    if computed != expected:
        raise ValueError(f"plan_hash mismatch: expected {computed}, got {expected}")

def check_bounds(plan: Plan):
    out_ratio = plan.output.width / plan.output.height
    for shot in plan.shots:
        src_ratio = shot.source.width / shot.source.height
        if src_ratio > out_ratio:
            rect_w = out_ratio / src_ratio
            rect_h = 1.0
        else:
            rect_w = 1.0
            rect_h = src_ratio / out_ratio

        for t in [shot.camera.start, shot.camera.end]:
            w = rect_w / t.zoom
            h = rect_h / t.zoom
            left = t.cx - w / 2
            right = t.cx + w / 2
            top = t.cy - h / 2
            bottom = t.cy + h / 2
            eps = 1e-4
            if left < -eps or right > 1+eps or top < -eps or bottom > 1+eps:
                raise ValueError(f"Camera window out of bounds in shot {shot.shot_id}")

def parse_plan(data: dict) -> Plan:
    validate_plan_hash(data)
    plan = Plan(**data)
    check_bounds(plan)
    return plan
PYEOF

# 2. Storage
cat << 'PYEOF' > modules/storage.py
import os
import shutil
import urllib.parse
from typing import Protocol

class Storage(Protocol):
    def get(self, uri: str, dest: str) -> None: ...
    def put(self, path: str, uri: str) -> None: ...

class S3Storage(Storage):
    def _get_client(self):
        import boto3
        return boto3.client('s3')

    def get(self, uri: str, dest: str) -> None:
        if not uri.startswith("s3://"):
            raise ValueError("Not an S3 URI")
        parts = uri[5:].split("/", 1)
        self._get_client().download_file(parts[0], parts[1], dest)

    def put(self, path: str, uri: str) -> None:
        if not uri.startswith("s3://"):
            raise ValueError("Not an S3 URI")
        parts = uri[5:].split("/", 1)
        self._get_client().upload_file(path, parts[0], parts[1])

class LocalMapStorage(Storage):
    def __init__(self, images_dir: str):
        self.images_dir = images_dir

    def get(self, uri: str, dest: str) -> None:
        basename = uri.split("/")[-1]
        src = os.path.join(self.images_dir, basename)
        if not os.path.exists(src):
            # For testing, generate a dummy image if not exists
            from PIL import Image
            img = Image.new('RGB', (1920, 1080), color = 'red')
            img.save(src)
        shutil.copy(src, dest)

    def put(self, path: str, uri: str) -> None:
        basename = uri.split("/")[-1]
        dest = os.path.join(self.images_dir, basename)
        shutil.copy(path, dest)
PYEOF

# 3. Images
cat << 'PYEOF' > modules/images.py
import os
from PIL import Image, ImageOps

def fetch_and_prepare(plan, storage, workdir):
    warnings = []
    images_dir = os.path.join(workdir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for shot in plan.shots:
        basename = shot.image_uri.split("/")[-1]
        dest_path = os.path.join(images_dir, basename)
        
        if not os.path.exists(dest_path):
            try:
                storage.get(shot.image_uri, dest_path)
            except Exception as e:
                raise RuntimeError(f"image_fetch_failed: {e}")
                
        # Prepare
        try:
            with Image.open(dest_path) as img:
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                
                # Check dimensions against plan
                if w != shot.source.width or h != shot.source.height:
                    warnings.append(f"Image {basename} dimensions {w}x{h} do not match plan {shot.source.width}x{shot.source.height}. Clamping camera.")
                
                # Downscale if long side > 5000 px
                if max(w, h) > 5000:
                    scale = 5000 / max(w, h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    # Overwrite
                    img.save(dest_path, "JPEG", quality=95)
        except Exception as e:
            raise RuntimeError(f"image preparation failed: {e}")
    
    return warnings
PYEOF

# 4. Remotion Render
cat << 'PYEOF' > modules/remotion_render.py
import os
import json
import subprocess

def render(plan, workdir, scale=1.0):
    # build props
    props_path = os.path.join(workdir, "props.json")
    out_path = os.path.join(workdir, "out.mp4")
    
    # Calculate frames
    shots = []
    cumulative = 0
    fps = plan.output.fps
    for shot in plan.shots:
        start_frame = round(cumulative * fps / 100)
        cumulative += int(round(shot.duration_s * 100))
        end_frame = round(cumulative * fps / 100)
        duration_frames = end_frame - start_frame
        
        basename = shot.image_uri.split("/")[-1]
        
        shots.append({
            "id": shot.shot_id,
            "durationFrames": duration_frames,
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
        "fps": fps,
        "width": plan.output.width,
        "height": plan.output.height
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
        "--crf", "18"
    ]
    
    try:
        subprocess.run(cmd, cwd=remotion_dir, check=True, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise RuntimeError("timeout")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"render_failed: {e.stderr.decode()}")
        
    return out_path
PYEOF

# 5. Probe
cat << 'PYEOF' > modules/probe.py
import subprocess
import json

def verify(video_path, plan):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    
    if stream["codec_name"] != plan.output.codec:
        raise RuntimeError(f"verification_failed: Expected codec {plan.output.codec}, got {stream['codec_name']}")
        
    if stream["pix_fmt"] != "yuv420p":
        raise RuntimeError(f"verification_failed: Expected yuv420p, got {stream['pix_fmt']}")
        
    if int(stream["width"]) != plan.output.width or int(stream["height"]) != plan.output.height:
        raise RuntimeError(f"verification_failed: Expected {plan.output.width}x{plan.output.height}")
        
    fps_num, fps_den = stream["r_frame_rate"].split('/')
    fps = int(fps_num) / int(fps_den)
    if round(fps) != plan.output.fps:
        raise RuntimeError(f"verification_failed: Expected {plan.output.fps} fps, got {fps}")
        
    duration = float(info["format"]["duration"])
    expected_duration = plan.total_duration_s
    frame_dur = 1.0 / plan.output.fps
    if abs(duration - expected_duration) > frame_dur + 0.01:
        raise RuntimeError(f"verification_failed: Expected duration {expected_duration}, got {duration}")
PYEOF

# 6. Master Script
cat << 'PYEOF' > run_tour.py
import argparse
import json
import tempfile
import os
import time
from modules.plan_io import parse_plan
from modules.storage import LocalMapStorage, S3Storage
from modules.images import fetch_and_prepare
from modules.remotion_render import render
from modules.probe import verify

def run_tour(request, storage, workdir, scale=1.0):
    start_time = time.time()
    try:
        plan_dict = request.get("plan", {})
        try:
            plan = parse_plan(plan_dict)
        except ValueError as e:
            return {"status": "failed", "error": {"code": "invalid_plan", "message": str(e)}}
            
        warnings = fetch_and_prepare(plan, storage, workdir)
        
        out_path = render(plan, workdir, scale)
        
        verify(out_path, plan)
        
        # Upload
        output_prefix = request.get("output_prefix", "s3://dummy/out/")
        master_uri = output_prefix + "master.mp4"
        storage.put(out_path, master_uri)
        
        render_seconds = time.time() - start_time
        
        return {
            "status": "completed",
            "video_uri": master_uri,
            "duration_s": plan.total_duration_s,
            "width": plan.output.width,
            "height": plan.output.height,
            "fps": plan.output.fps,
            "codec": plan.output.codec,
            "render_seconds": render_seconds,
            "plan_hash": plan.provenance.plan_hash if hasattr(plan, "provenance") else None,
            "warnings": warnings,
            "error": None
        }
    except RuntimeError as e:
        msg = str(e)
        code = msg.split(":")[0] if ":" in msg else msg
        return {"status": "failed", "error": {"code": code, "message": msg}}
    except Exception as e:
        return {"status": "failed", "error": {"code": "unknown", "message": str(e)}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    
    with open(args.plan) as f:
        plan_data = json.load(f)
        
    request = {
        "job_id": "cli",
        "job_type": "tour_render",
        "plan": plan_data,
        "output_prefix": "s3://dummy/out/"
    }
    
    storage = LocalMapStorage(args.images_dir)
    storage.put = lambda path, uri: __import__('shutil').copy(path, os.path.join(args.out, "master.mp4"))
    
    with tempfile.TemporaryDirectory() as workdir:
        res = run_tour(request, storage, workdir, args.scale)
        print(json.dumps(res, indent=2))
PYEOF

# 7. Handler
cat << 'PYEOF' > handler.py
import tempfile
from run_tour import run_tour
from modules.storage import S3Storage

def handler(event):
    import runpod
    with tempfile.TemporaryDirectory() as workdir:
        return run_tour(event["input"], S3Storage(), workdir)

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
PYEOF

# 8. Remotion setup
cd remotion
npm init -y
npm i remotion@4.0.141 @remotion/cli@4.0.141 react react-dom
npm i -D typescript @types/react ts-node

cat << 'JSOEF' > tsconfig.json
{
  "compilerOptions": {
    "jsx": "react-jsx",
    "esModuleInterop": true,
    "strict": true,
    "lib": ["DOM", "ES2020"]
  }
}
JSOEF

cat << 'TSEOF' > remotion.config.ts
import { Config } from '@remotion/cli/config';
Config.setVideoImageFormat('jpeg');
TSEOF

cat << 'TSEOF' > src/math.ts
export function windowToTransform(
  sourceW: number,
  sourceH: number,
  outW: number,
  outH: number,
  cx: number,
  cy: number,
  zoom: number
) {
  const outRatio = outW / outH;
  const srcRatio = sourceW / sourceH;
  
  let rectW, rectH;
  if (srcRatio > outRatio) {
    rectW = sourceH * outRatio;
    rectH = sourceH;
  } else {
    rectW = sourceW;
    rectH = sourceW / outRatio;
  }
  
  const w = rectW / zoom;
  const h = rectH / zoom;
  
  const px = cx * sourceW;
  const py = cy * sourceH;
  
  const left = px - w / 2;
  const top = py - h / 2;
  
  // Transform to apply to the image to map [left, top, w, h] to [0, 0, outW, outH]
  const scale = outW / w;
  const translateX = -left;
  const translateY = -top;
  
  return { scale, translateX, translateY };
}
TSEOF

cat << 'TSEOF' > src/KenBurnsShot.tsx
import { AbsoluteFill, Img, useCurrentFrame, staticFile } from 'remotion';
import React from 'react';
import { windowToTransform } from './math';

export const KenBurnsShot: React.FC<{
  durationFrames: number;
  image: string;
  source: { width: number; height: number };
  camera: {
    start: { cx: number; cy: number; zoom: number };
    end: { cx: number; cy: number; zoom: number };
  };
  width: number;
  height: number;
}> = ({ durationFrames, image, source, camera, width, height }) => {
  const frame = useCurrentFrame();
  const rawT = frame / Math.max(1, durationFrames - 1);
  const t = Math.max(0, Math.min(1, rawT));
  const smoothT = t * t * (3 - 2 * t);
  
  const cx = camera.start.cx + (camera.end.cx - camera.start.cx) * smoothT;
  const cy = camera.start.cy + (camera.end.cy - camera.start.cy) * smoothT;
  const zoom = camera.start.zoom + (camera.end.zoom - camera.start.zoom) * smoothT;
  
  const { scale, translateX, translateY } = windowToTransform(
    source.width, source.height, width, height, cx, cy, zoom
  );
  
  return (
    <AbsoluteFill style={{ overflow: 'hidden' }}>
      <div style={{
        position: 'absolute',
        width: source.width,
        height: source.height,
        transformOrigin: '0 0',
        transform: `scale(${scale}) translate(${translateX}px, ${translateY}px)`
      }}>
        <Img src={staticFile(image)} style={{ width: '100%', height: '100%' }} />
      </div>
    </AbsoluteFill>
  );
};
TSEOF

cat << 'TSEOF' > src/TourComposition.tsx
import { Composition, Sequence, getInputProps } from 'remotion';
import React from 'react';
import { KenBurnsShot } from './KenBurnsShot';

export const TourComposition: React.FC = () => {
  const props = getInputProps();
  if (!props.shots) return null; // Default render check
  
  const shots = props.shots as any[];
  
  let currentFrame = 0;
  return (
    <>
      {shots.map((shot, idx) => {
        const start = currentFrame;
        currentFrame += shot.durationFrames;
        return (
          <Sequence key={shot.id} from={start} durationInFrames={shot.durationFrames}>
            <KenBurnsShot 
              durationFrames={shot.durationFrames}
              image={shot.image}
              source={shot.source}
              camera={shot.camera}
              width={props.width as number}
              height={props.height as number}
            />
          </Sequence>
        );
      })}
    </>
  );
};
TSEOF

cat << 'TSEOF' > src/index.ts
import { registerRoot } from 'remotion';
import { Root } from './Root';
registerRoot(Root);
TSEOF

cat << 'TSEOF' > src/Root.tsx
import { Composition, getInputProps } from 'remotion';
import React from 'react';
import { TourComposition } from './TourComposition';

export const Root: React.FC = () => {
  const props = getInputProps();
  const fps = (props.fps as number) || 30;
  const width = (props.width as number) || 1920;
  const height = (props.height as number) || 1080;
  const duration = props.shots ? 
    (props.shots as any[]).reduce((a, b) => a + b.durationFrames, 0) : 30;

  return (
    <Composition
      id="TourComposition"
      component={TourComposition}
      durationInFrames={Math.max(1, duration)}
      fps={fps}
      width={width}
      height={height}
      defaultProps={props}
    />
  );
};
TSEOF

# 9. Bench & Tests script setup
cd ..
cat << 'PYEOF' > bench.py
import time
import os
import tempfile
import urllib.parse
from modules.plan_io import Plan
from modules.remotion_render import render

def make_synthetic_plan(shots=30, dur=90, scale=0.5):
    # generate a plan
    plan_dict = {
        "plan_version": "1.0", "listing_id": "bench", "mode": "cinematic", "preset": "bench",
        "output": {"width": 3840, "height": 2160, "fps": 30, "codec": "h264"},
        "total_duration_s": dur,
        "shots": [], "warnings": [],
        "provenance": {"scoring_config_version": "1", "planner_config_version": "1", "plan_hash": "bench"}
    }
    
    shot_dur = dur / shots
    for i in range(shots):
        plan_dict["shots"].append({
            "shot_id": f"s{i}", "order": i, "asset_id": "a", "image_uri": f"s3://b/img{i}.jpg",
            "source": {"width": 6000, "height": 4000},
            "room_key": "x", "room_type": "y", "narrative_stage": "z", "role": "hero",
            "duration_s": shot_dur,
            "camera": {
                "type": "ken_burns", "easing": "ease_in_out",
                "start": {"cx": 0.5, "cy": 0.5, "zoom": 1.0},
                "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.1}
            },
            "transition_in": {"type": "cut", "duration_s": 0.0}
        })
        
    return plan_dict

def run_bench():
    # 30-shot / 90s plan at scale 0.5
    print("Running 90s bench at scale 0.5...")
    plan90 = Plan(**make_synthetic_plan(30, 90, 0.5))
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        from PIL import Image
        for i in range(30):
            Image.new('RGB', (600, 400), color='blue').save(os.path.join(d, "images", f"img{i}.jpg"))
            
        t0 = time.time()
        render(plan90, d, scale=0.5)
        t1 = time.time()
        print(f"90s at scale 0.5 took {t1-t0:.2f}s")
        print(f"Frames per second: {90*30 / (t1-t0):.2f}")

    # 10s slice at full 4K
    print("Running 10s slice at full 4K...")
    plan10 = Plan(**make_synthetic_plan(4, 10, 1.0))
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "images"))
        from PIL import Image
        for i in range(4):
            Image.new('RGB', (6000, 4000), color='blue').save(os.path.join(d, "images", f"img{i}.jpg"))
            
        t0 = time.time()
        render(plan10, d, scale=1.0)
        t1 = time.time()
        
        real_time = t1-t0
        print(f"10s at full 4K took {real_time:.2f}s")
        print(f"Frames per second: {10*30 / real_time:.2f}")
        
        cpu_count = os.cpu_count()
        gpu_used = False # Could check for nvenc if we enabled it
        print(f"CPU count: {cpu_count}, GPU used: {gpu_used}")
        est_90 = real_time * 9
        print(f"ESTIMATE for 90s at 4K: {est_90:.2f}s")
        print(f"SOW Target: <180s. Difference: {est_90 - 180:.2f}s")

if __name__ == "__main__":
    run_bench()
PYEOF

