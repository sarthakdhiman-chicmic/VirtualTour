import subprocess
import json

from tour_worker.errors import WorkerError
from tour_worker.render.encode import GOP_S, playback_stats, video_profile

# delivery audio (client review 2026-11-05): integrated loudness within LOUDNESS_TOLERANCE of -14 LUFS
# and a true peak at or below -1 dBTP; a silent track (no music) has no loudness to check
LOUDNESS_TARGET, LOUDNESS_TOLERANCE, TRUE_PEAK_MAX = -14.0, 1.0, -1.0
SILENT_LUFS = -60.0

# ffprobe's level number: H.264's level x 10 (4.1 -> 41), H.265's general_level_idc x 30 (4.1 -> 123)
LEVEL_SCALE = {"h264": 10, "hevc": 30}


def verify(video_path, plan, codec=None):
    """Checks a rendered file against its plan. codec: the stream's codec when it isn't the plan's
    output.codec (the optional H.265 copy, "hevc"), which is also checked for the hvc1 tag."""
    codec = codec or plan.output.codec
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    
    if stream["codec_name"] != codec:
        raise WorkerError("verification_failed", f"Expected codec {codec}, got {stream['codec_name']}")
    if codec == "hevc" and stream.get("codec_tag_string") != "hvc1":
        raise WorkerError("verification_failed", f"Expected the hvc1 tag, got {stream.get('codec_tag_string')}")
        
    if stream["pix_fmt"] != "yuv420p":
        raise WorkerError("verification_failed", f"Expected yuv420p, got {stream['pix_fmt']}")
        
    if int(stream["width"]) != plan.output.width or int(stream["height"]) != plan.output.height:
        raise WorkerError("verification_failed", f"Expected {plan.output.width}x{plan.output.height}")
        
    fps_num, fps_den = stream["r_frame_rate"].split('/')
    fps = int(fps_num) / int(fps_den)
    if round(fps) != plan.output.fps:
        raise WorkerError("verification_failed", f"Expected {plan.output.fps} fps, got {fps}")
    tags = {k: stream.get(k) for k in ("color_space", "color_primaries", "color_transfer")}
    if any(v != "bt709" for v in tags.values()):
        raise WorkerError("verification_failed", f"Expected BT.709 colour, got {tags}")
        
    duration = float(stream.get("duration") or info["format"]["duration"])
    container_dur = float(info["format"]["duration"])
    expected_duration = plan.total_duration_s
    frame_dur = 1.0 / plan.output.fps
    if abs(duration - expected_duration) > frame_dur + 0.05 and abs(container_dur - expected_duration) > 0.15:
        raise WorkerError("verification_failed", f"Expected duration {expected_duration}, got {duration}")

    # smooth playback, at the level and cap this frame size and file-size limit were encoded with
    # (tour_worker/render/encode.py video_profile)
    max_bytes = getattr(plan.output, "max_bytes", None)
    vp = video_profile(plan.output.width, plan.output.height, plan.total_duration_s, max_bytes)
    pb = playback_stats(video_path, vp.maxrate, vp.bufsize)
    if not pb["faststart"]:
        raise WorkerError("verification_failed", "moov atom is not before the media data (no faststart)")
    if pb["max_keyframe_gap_s"] > GOP_S + 0.05:
        raise WorkerError("verification_failed", f"keyframe gap {pb['max_keyframe_gap_s']} s > {GOP_S} s")
    if pb["vbv_min_fullness"] < 0:
        raise WorkerError("verification_failed", f"a player reading at {vp.maxrate}bps with a {vp.bufsize}bit "
                           f"buffer would stall (buffer fullness {pb['vbv_min_fullness']})")
    if pb["level"] is not None and int(pb["level"]) > round(float(vp.level) * LEVEL_SCALE.get(codec, 10)):
        raise WorkerError("verification_failed", f"{codec} level {pb['level']} > {vp.level}")
    if max_bytes and pb["size_bytes"] > max_bytes:
        raise WorkerError("verification_failed", f"{pb['size_bytes']} bytes > the {max_bytes} byte limit")
    if not any(s["codec_type"] == "audio" for s in info["streams"]):
        raise WorkerError("verification_failed", "no audio track")
    pb.update(audio_loudness(video_path))
    if pb["loudness_lufs"] > SILENT_LUFS:
        if abs(pb["loudness_lufs"] - LOUDNESS_TARGET) > LOUDNESS_TOLERANCE:
            raise WorkerError("verification_failed", f"loudness {pb['loudness_lufs']} LUFS, not "
                                                     f"{LOUDNESS_TARGET} +- {LOUDNESS_TOLERANCE}")
        if pb["true_peak_dbtp"] > TRUE_PEAK_MAX:
            raise WorkerError("verification_failed", f"true peak {pb['true_peak_dbtp']} dBTP > {TRUE_PEAK_MAX}")
    return pb


def audio_loudness(path: str) -> dict:
    """EBU R128 of the file's audio: integrated loudness (LUFS) and true peak (dBTP)."""
    import re
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-map", "0:a:0", "-filter:a",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    summary = p.stderr[p.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary)
    tp = re.search(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    val = lambda m: -120.0 if m is None or m.group(1) == "-inf" else float(m.group(1))   # noqa: E731
    return {"loudness_lufs": val(i), "true_peak_dbtp": val(tp)}
