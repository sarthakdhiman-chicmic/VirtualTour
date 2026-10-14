"""Delivery encode: turns Remotion's master render into a file that plays smoothly everywhere.

Remotion encodes at a constant quality (CRF) with no bitrate ceiling. Tour v5 averaged 14.8 Mbps
and peaked at 36 Mbps per second (the exterior's foliage), above the 25 Mbps that H.264 High
Level 4.0 (the level it was tagged) allows, with a keyframe only every ~5 s. Hardware decoders
and anything streaming the file stalled and re-buffered on those peaks.

The delivery file:
- quality-targeted (CRF) but capped by a VBV buffer, so no stretch of video needs more than the
  cap (MAXRATE over BUFSIZE)
- High profile, Level 4.1, a keyframe at least every 2 s (also at every cut, via scenecut), so
  players start and seek quickly
- moov atom first (faststart), so playback starts before the whole file has downloaded
- AAC 128 kbps stereo (Remotion's track is silent; this keeps the audio behaviour)
- colour passed through unchanged (BT.601 matrix, limited range, tagged as Remotion tags it), so
  the picture looks exactly as before
x264 runs with sliced threads, which is deterministic (the default frame threading was not, with
a VBV cap), so the same master always gives a byte-identical delivery file.
"""
import json
import os
import subprocess

MASTER_CRF = 16          # Remotion's master render: near-transparent, re-encoded below
CRF = 20
MAXRATE = "10M"
BUFSIZE = "20M"
PRESET = "slow"
LEVEL = "4.1"
GOP_S = 2                # keyframe at least every GOP_S seconds
AUDIO_BITRATE = "128k"
VBV_INIT = 0.9           # x264's default initial buffer fullness


def _bits(rate: str) -> float:
    return float(rate[:-1]) * 1e6 if rate[-1] in "Mm" else float(rate)


def vbv_min_fullness(frame_bits_decode_order, fps, maxrate=MAXRATE, bufsize=BUFSIZE, init=VBV_INIT):
    """Leaky-bucket check of what smooth playback needs: a player that receives the file at
    maxrate into a bufsize buffer, starting init full, takes each frame out when it decodes it.
    Returns the lowest fullness as a fraction of bufsize; below 0 the player would stall."""
    rate, size = _bits(maxrate) / fps, _bits(bufsize)
    level, low = init * size, init * size
    for b in frame_bits_decode_order:
        level -= b
        low = min(low, level)
        level = min(size, level + rate)
    return low / size


def deliver(master_path: str, out_path: str, fps: int) -> str:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", master_path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
        "-maxrate", MAXRATE, "-bufsize", BUFSIZE,
        "-profile:v", "high", "-level:v", LEVEL,
        "-g", str(GOP_S * fps), "-keyint_min", str(fps),
        # sliced threads: x264's default frame threading with a VBV cap was not deterministic (374
        # of 600 frames differed between two encodes of the same master); slice threading is
        # byte-identical run to run, ~5.7x faster than one thread, files ~1.6% larger
        "-x264-params", "nal-hrd=vbr:sliced-threads=1",
        "-pix_fmt", "yuv420p", "-colorspace", "bt470bg", "-color_range", "tv",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2",
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"encode_failed: {e.stderr[-2000:]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("encode_failed: timeout")
    return out_path


def max_keyframe_gap(keys: list[float], packets: list[dict], fps: float) -> float:
    """Longest stretch without a keyframe, including the stretch from the last keyframe to the
    end of the video (a one-keyframe clip's gap is its whole length)."""
    end = max(float(p["pts_time"]) for p in packets) + 1 / fps
    marks = sorted(keys) + [end]
    return max(b - a for a, b in zip(marks, marks[1:]))


def playback_stats(path: str) -> dict:
    """Facts a smooth player depends on: per-second video bitrate, keyframe spacing, profile and
    level, and whether the moov atom comes before the media data."""
    pk = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "packet=pts_time,size,flags", "-of", "json", path],
                        check=True, capture_output=True, text=True).stdout
    packets = json.loads(pk)["packets"]   # ffprobe lists packets in decode order
    per_s, keys = {}, []
    for p in packets:
        t = float(p["pts_time"])
        per_s[int(t)] = per_s.get(int(t), 0) + int(p["size"]) * 8
        if "K" in p.get("flags", ""):
            keys.append(t)
    keys.sort()
    st = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=profile,level", "-of", "json", path],
                                   check=True, capture_output=True, text=True).stdout)["streams"][0]
    with open(path, "rb") as f:
        order = []
        pos, size = 0, os.path.getsize(path)
        while pos < size and len(order) < 8:
            f.seek(pos)
            h = f.read(16)
            n, typ = int.from_bytes(h[:4], "big"), h[4:8].decode("latin-1")
            if n == 1:
                n = int.from_bytes(h[8:16], "big")
            order.append(typ)
            if n <= 0:
                break
            pos += n
    rates = [v / 1e6 for _, v in sorted(per_s.items())]
    fr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=r_frame_rate", "-of", "json", path],
                                   check=True, capture_output=True, text=True).stdout)["streams"][0]["r_frame_rate"]
    fps = int(fr.split("/")[0]) / int(fr.split("/")[1])
    vbv = vbv_min_fullness([int(p["size"]) * 8 for p in packets], fps)
    return {"mbps_mean": round(sum(rates) / len(rates), 2), "mbps_max_1s": round(max(rates), 2),
            "vbv_min_fullness": round(vbv, 4),
            "keyframes": len(keys), "max_keyframe_gap_s": round(max_keyframe_gap(keys, packets, fps), 3),
            "profile": st.get("profile"), "level": st.get("level"),
            "faststart": order.index("moov") < order.index("mdat") if "moov" in order and "mdat" in order else False,
            "size_mb": round(os.path.getsize(path) / 1e6, 1)}
