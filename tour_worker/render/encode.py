"""Delivery encode settings and checks: a file that plays smoothly everywhere.

The direct renderer (tour_worker.render.direct) encodes with these settings; deliver() re-encodes
an existing master the same way. History: Remotion (the old renderer) encoded at a constant
quality (CRF) with no bitrate ceiling. Tour v5 averaged 14.8 Mbps and peaked at 36 Mbps per second (the exterior's foliage), above the 25 Mbps that H.264 High
Level 4.0 (the level it was tagged) allows, with a keyframe only every ~5 s. Hardware decoders
and anything streaming the file stalled and re-buffered on those peaks.

The delivery file:
- quality-targeted (CRF) but capped by a VBV buffer, so no stretch of video needs more than the
  cap (MAXRATE over BUFSIZE)
- High profile, Level 4.1, a keyframe at least every 2 s (also at every cut, via scenecut), so
  players start and seek quickly
- moov atom first (faststart), so playback starts before the whole file has downloaded
- AAC 128 kbps stereo (Remotion's track is silent; this keeps the audio behaviour)
- colour: BT.709 matrix, primaries and transfer, limited range (render/direct.py COLOUR_TAGS; it was
  BT.601 until the client review of 2026-11-05)
x264 runs single-threaded per chunk, chunks in parallel (see deliver), so the same master always
gives a byte-identical delivery file.
"""
import json
import os
import subprocess
from dataclasses import dataclass

from tour_worker.errors import WorkerError

CRF = 20
MAXRATE = "10M"
BUFSIZE = "20M"
PRESET = "slow"
LEVEL = "4.1"
GOP_S = 2                # keyframe at least every GOP_S seconds
AUDIO_BITRATE = "128k"
VBV_INIT = 0.9           # x264's default initial buffer fullness
# frames above 1080p (4K): Level 4.1 holds at most 8192 macroblocks a frame (1920x1088); 2160p
# needs Level 5.1, and a 40 Mbps cap (YouTube's 4K30 upload guidance is 35-45 Mbps)
MAX_MB_LEVEL_41 = 8192
LEVEL_4K = "5.1"
MAXRATE_4K = "40M"
BUFSIZE_4K = "80M"
SIZE_MARGIN = 0.95       # of a file-size limit, the share planned for (container overhead)
# the optional H.265 copy (SOW 3.2.1: "H.264 MP4, H.265 optional"), drawn from the same frames as
# the H.264 master. Same caps, levels and keyframe spacing (HEVC Main Level 4.1 allows 20 Mbps,
# 5.1 40 Mbps). x265 with one thread and no frame or lookahead threads is byte-identical run to run
# (checked 2026-11-02); closed GOPs, so chunks join by stream copy; tagged hvc1 for Apple players.
HEVC_CRF = 22            # about x264's CRF 20 in look, at a smaller file
HEVC_PRESET = "medium"
HEVC_PARAMS = "pools=1:frame-threads=1:lookahead-slices=0:open-gop=0:hrd=1:log-level=error"


def _bits(rate: str) -> float:
    unit = {"M": 1e6, "m": 1e6, "k": 1e3, "K": 1e3}.get(rate[-1])
    return float(rate[:-1]) * unit if unit else float(rate)


@dataclass(frozen=True)
class VideoProfile:
    level: str
    maxrate: str
    bufsize: str


def video_profile(width: int, height: int, duration_s: float | None = None,
                  max_bytes: int | None = None) -> VideoProfile:
    """The H.264 level and VBV cap for a frame size: up to 1080p (Level 4.1) exactly the settings
    above, larger frames Level 5.1 with the 4K cap. max_bytes (a platform's file-size limit, e.g.
    Instagram Reels' 100 MB) lowers the cap so the file fits: a file that passes the VBV check
    (tour_worker/render/probe.py) holds at most cap x duration + VBV_INIT x bufsize bits of video,
    and the audio comes on top."""
    mbs = -(-width // 16) * -(-height // 16)
    if mbs <= MAX_MB_LEVEL_41:
        level, maxrate, bufsize = LEVEL, MAXRATE, BUFSIZE
    else:
        level, maxrate, bufsize = LEVEL_4K, MAXRATE_4K, BUFSIZE_4K
    if max_bytes and duration_s:
        video_bits = max_bytes * 8 * SIZE_MARGIN - _bits(AUDIO_BITRATE) * duration_s
        cap = int(video_bits / (duration_s + 2 * VBV_INIT))     # bufsize = 2 x cap
        if cap < _bits(maxrate):
            maxrate, bufsize = f"{cap // 1000}k", f"{2 * (cap // 1000)}k"
    return VideoProfile(level, maxrate, bufsize)


def hevc_args(vp: VideoProfile, fps: int, vbv_init: float) -> list[str]:
    """ffmpeg output args for the H.265 copy, at the H.264 master's level, cap and GOP."""
    return ["-c:v", "libx265", "-preset", HEVC_PRESET, "-crf", str(HEVC_CRF),
            "-maxrate", vp.maxrate, "-bufsize", vp.bufsize, "-profile:v", "main",
            "-g", str(GOP_S * fps), "-keyint_min", str(fps), "-tag:v", "hvc1",
            "-x265-params", f"{HEVC_PARAMS}:vbv-init={vbv_init}:level-idc={vp.level}"]


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


CHUNK_GOPS = 2           # frames per parallel chunk = CHUNK_GOPS * GOP; each chunk starts on a keyframe
VBV_INIT_CHUNK = 0.5     # a chunk assumes the player's buffer is only half full when it starts


def _run(cmd):
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        raise WorkerError("encode_failed", e.stderr[-2000:])
    except subprocess.TimeoutExpired:
        raise WorkerError("encode_failed", "timeout")


def _frame_count(path: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", path],
                         check=True, capture_output=True, text=True).stdout
    return int(out.strip())


def deliver(master_path: str, out_path: str, fps: int, profile: VideoProfile | None = None) -> str:
    """Re-encode the master in parallel chunks, each with ONE x264 thread, then join them.
    Every multi-threaded x264 mode with a VBV cap varied run to run (frame threads, sliced threads,
    sliced + lookahead-threads=1: 4-5 distinct outputs in 5 encodes of the same clip); one thread
    is byte-identical. Chunks are whole GOPs, so each starts on a keyframe and the join is a plain
    stream copy. Each chunk assumes a half-full player buffer at its start (VBV_INIT_CHUNK), so the
    joined file stays inside the cap; tour_worker/render/probe.py simulates the whole file to confirm."""
    import concurrent.futures
    import os
    import tempfile

    profile = profile or VideoProfile(LEVEL, MAXRATE, BUFSIZE)
    n = _frame_count(master_path)
    step = CHUNK_GOPS * GOP_S * fps
    ranges = [(a, min(n, a + step)) for a in range(0, n, step)]
    with tempfile.TemporaryDirectory(prefix="deliver_") as td:
        def encode(k_ab):
            k, (a, b) = k_ab
            part = os.path.join(td, f"part_{k:04d}.mp4")
            _run(["ffmpeg", "-y", "-loglevel", "error", "-i", master_path, "-map", "0:v:0",
                  "-vf", f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS",
                  "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
                  "-maxrate", profile.maxrate, "-bufsize", profile.bufsize,
                  "-profile:v", "high", "-level:v", profile.level,
                  "-g", str(GOP_S * fps), "-keyint_min", str(fps),
                  "-x264-params", f"nal-hrd=vbr:threads=1:vbv-init={VBV_INIT_CHUNK}",
                  "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                  "-color_trc", "bt709", "-color_range", "tv",
                  "-r", str(fps), "-an", part])
            return part
        # the CPUs this process may use (taskset / container limits), not the machine's count:
        # each x264 process holds ~0.5 GB, so 20 of them on a 4-CPU job overran its memory
        workers = max(1, min(len(ranges), len(os.sched_getaffinity(0))))
        with concurrent.futures.ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(encode, enumerate(ranges)))
        listing = os.path.join(td, "parts.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
              "-i", master_path, "-map", "0:v:0", "-map", "1:a:0?",
              "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2",
              "-movflags", "+faststart", out_path])
    return out_path


def max_keyframe_gap(keys: list[float], packets: list[dict], fps: float) -> float:
    """Longest stretch without a keyframe, including the stretch from the last keyframe to the
    end of the video (a one-keyframe clip's gap is its whole length)."""
    end = max(float(p["pts_time"]) for p in packets) + 1 / fps
    marks = sorted(keys) + [end]
    return max(b - a for a, b in zip(marks, marks[1:]))


def playback_stats(path: str, maxrate: str = MAXRATE, bufsize: str = BUFSIZE) -> dict:
    """Facts a smooth player depends on: per-second video bitrate, keyframe spacing, profile and
    level, whether the moov atom comes before the media data, and the VBV buffer's lowest
    fullness for a player reading at maxrate into bufsize."""
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
    vbv = vbv_min_fullness([int(p["size"]) * 8 for p in packets], fps, maxrate, bufsize)
    return {"mbps_mean": round(sum(rates) / len(rates), 2), "mbps_max_1s": round(max(rates), 2),
            "vbv_min_fullness": round(vbv, 4),
            "keyframes": len(keys), "max_keyframe_gap_s": round(max_keyframe_gap(keys, packets, fps), 3),
            "profile": st.get("profile"), "level": st.get("level"),
            "faststart": order.index("moov") < order.index("mdat") if "moov" in order and "mdat" in order else False,
            "size_mb": round(os.path.getsize(path) / 1e6, 1), "size_bytes": os.path.getsize(path)}
