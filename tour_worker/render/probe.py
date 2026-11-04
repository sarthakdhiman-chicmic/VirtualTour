import subprocess
import json

from tour_worker.errors import WorkerError
from tour_worker.render.encode import BUFSIZE, GOP_S, MAXRATE, playback_stats

def _expected_frames(plan) -> int:
    from tour_worker.render.timeline import compute_timeline
    return (compute_timeline(plan)[-1]["nominal_end"] if plan.shots
            else round(plan.total_duration_s * plan.output.fps))


def verify(video_path, plan):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise WorkerError("verification_failed", f"ffprobe could not read the file (exit {result.returncode})")
    info = json.loads(result.stdout)

    stream = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), None)
    if stream is None:
        raise WorkerError("verification_failed", "no video stream")

    if stream["codec_name"] != plan.output.codec:
        raise WorkerError("verification_failed", f"Expected codec {plan.output.codec}, got {stream['codec_name']}")

    if stream["pix_fmt"] != "yuv420p":
        raise WorkerError("verification_failed", f"Expected yuv420p, got {stream['pix_fmt']}")

    if int(stream["width"]) != plan.output.width or int(stream["height"]) != plan.output.height:
        raise WorkerError("verification_failed", f"Expected {plan.output.width}x{plan.output.height}")

    fps_num, fps_den = stream["r_frame_rate"].split('/')
    fps = int(fps_num) / int(fps_den)
    if abs(fps - plan.output.fps) > 1e-6:                   # 29.97 is not 30
        raise WorkerError("verification_failed", f"Expected {plan.output.fps} fps, got {fps}")

    if not any(s["codec_type"] == "audio" for s in info["streams"]):
        raise WorkerError("verification_failed", "no audio stream (the renderer always muxes one)")

    # the video stream itself must be complete: a short video under full-length audio has the
    # right container duration, so each is checked on its own
    expected_frames = _expected_frames(plan)
    if stream.get("nb_frames") and int(stream["nb_frames"]) != expected_frames:
        raise WorkerError("verification_failed", f"Expected {expected_frames} frames, got {stream['nb_frames']}")
    duration = float(stream.get("duration") or info["format"]["duration"])
    container_dur = float(info["format"]["duration"])
    expected_duration = plan.total_duration_s
    frame_dur = 1.0 / plan.output.fps
    if abs(duration - expected_duration) > frame_dur + 0.05 or abs(container_dur - expected_duration) > 0.15:
        raise WorkerError("verification_failed", f"Expected duration {expected_duration}, got video {duration} "
                          f"/ file {container_dur}")

    # smooth playback (tour_worker/render/encode.py)
    pb = playback_stats(video_path)
    if not pb["faststart"]:
        raise WorkerError("verification_failed", "moov atom is not before the media data (no faststart)")
    if pb["max_keyframe_gap_s"] > GOP_S + 0.05:
        raise WorkerError("verification_failed", f"keyframe gap {pb['max_keyframe_gap_s']} s > {GOP_S} s")
    if pb["vbv_min_fullness"] < 0:
        raise WorkerError("verification_failed", f"a player reading at {MAXRATE}bps with a {BUFSIZE}bit "
                           f"buffer would stall (buffer fullness {pb['vbv_min_fullness']})")
    if pb["level"] is not None and int(pb["level"]) > 41:
        raise WorkerError("verification_failed", f"H.264 level {pb['level']} > 4.1")
    return pb
