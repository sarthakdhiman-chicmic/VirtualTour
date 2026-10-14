import subprocess
import json

from modules.encode import BUFSIZE, GOP_S, MAXRATE, playback_stats

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
        
    duration = float(stream.get("duration") or info["format"]["duration"])
    container_dur = float(info["format"]["duration"])
    expected_duration = plan.total_duration_s
    frame_dur = 1.0 / plan.output.fps
    if abs(duration - expected_duration) > frame_dur + 0.05 and abs(container_dur - expected_duration) > 0.15:
        raise RuntimeError(f"verification_failed: Expected duration {expected_duration}, got {duration}")

    # smooth playback (modules/encode.py)
    pb = playback_stats(video_path)
    if not pb["faststart"]:
        raise RuntimeError("verification_failed: moov atom is not before the media data (no faststart)")
    if pb["max_keyframe_gap_s"] > GOP_S + 0.05:
        raise RuntimeError(f"verification_failed: keyframe gap {pb['max_keyframe_gap_s']} s > {GOP_S} s")
    if pb["vbv_min_fullness"] < 0:
        raise RuntimeError(f"verification_failed: a player reading at {MAXRATE}bps with a {BUFSIZE}bit "
                           f"buffer would stall (buffer fullness {pb['vbv_min_fullness']})")
    if pb["level"] is not None and int(pb["level"]) > 41:
        raise RuntimeError(f"verification_failed: H.264 level {pb['level']} > 4.1")
    return pb
