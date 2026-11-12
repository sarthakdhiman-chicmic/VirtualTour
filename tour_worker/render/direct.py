"""Direct renderer (2026-10-16): OpenCV draws every frame, ffmpeg encodes it.

Why: in the old Remotion path (removed 2026-10; branch archive/remotion-renderer) every frame went
OpenCV prewarp -> JPEG -> Chrome -> JPEG (quality 80) -> master H.264 -> delivery H.264, and Chrome was both the slowest stage (~8-10 min for a 75 s tour
on the dev box) and the source of render-to-render differences. Every shot can be drawn by the
ShotWarp (tour_worker.render.frames: crop, rotate, roll, ken_burns, motion regions), so frames now go straight from
OpenCV to the delivery encoder: one lossy step, byte-identical output for the same plan.

Composition is the old browser composition's, exactly: shots stacked in order over black, each faded in by
smoothstep over its fade frames (hard cuts: fade 0). Encoding is tour_worker.render.encode's delivery
encode: chunks of whole GOPs, x264 single-threaded per chunk (deterministic), VBV-capped, joined by
stream copy. Each chunk's process draws its own frames, so drawing and encoding run in parallel.

Audio: the plan's music (plan 1.5) - the track queued `loops` times, each copy cut at loop_s (the
bar where its beat stops), or sync v2's segments, trimmed to the tour, faded out over fade_out_s -
normalised to -14 LUFS / -1.5 dBTP (two-pass, linear gain), made once as AAC and stream-copied into
every copy; else a silent track (players and platforms expect one).
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile

import cv2
import numpy as np

from tour_worker.errors import WorkerError
from tour_worker.render.encode import (CHUNK_GOPS, CRF, GOP_S, PRESET, VBV_INIT_CHUNK, _run,
                                       hevc_args, video_profile)
from tour_worker.render.frames import ShotWarp, render_workers
from tour_worker.render.overlays import Overlays
from tour_worker.render.timeline import compute_timeline

AUDIO_BITRATE = "192k"
_STATE: dict = {}


def fade_opacity(f: int, fade: int) -> float:
    """Opacity of a shot f frames into its fade-in of `fade` frames (smoothstep; the Remotion
    composition's fadeOpacity)."""
    if fade <= 0 or f >= fade:
        return 1.0
    if f < 0:
        return 0.0
    x = (f + 1) / (fade + 1)
    return x * x * (3 - 2 * x)


COLOUR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
HEVC_NAME = "out_hevc.mp4"
MLS_NAME = "out_mls.mp4"


def wants_hevc(plan) -> bool:
    """The plan asks for the optional H.265 copy next to the H.264 master."""
    return bool(getattr(plan.output, "hevc", False))


def wants_mls(plan) -> bool:
    """The plan asks for the MLS copy: the same frames without the room labels (render/overlays.py)."""
    return bool(getattr(plan.output, "mls_copy", False))


def _hevc_part(part: str) -> str:
    return part[:-len(".mp4")] + "_hevc.mp4"


def _mls_part(part: str) -> str:
    return part[:-len(".mp4")] + "_mls.mp4"


def total_frames(plan) -> int:
    return compute_timeline(plan)[-1]["nominal_end"] if plan.shots else 0


def _init(plan, images: str, splat=None, supersample: float = 1.5) -> None:
    cv2.setNumThreads(1)
    _STATE.clear()
    timeline = compute_timeline(plan)
    # splat: the scene's renderer for plan 1.6 3D shots (render/splat_shot.py), drawn in this process
    _STATE.update(plan=plan, images=images, timeline=timeline, shots={}, splat=splat,
                  supersample=supersample, splat_stats={})
    overlays = Overlays(plan, timeline, images)
    _STATE["overlays"] = overlays if overlays.active else None


def _shot(i: int):
    """Shot i's frame source: a ShotWarp (photo) or a SplatShot (plan 1.6 3D), same interface."""
    cache = _STATE["shots"]
    if i not in cache:
        for k in [k for k in cache if k < i - 1]:            # shots run in order: keep the last two
            del cache[k]
        plan = _STATE["plan"]
        n = _STATE["timeline"][i]["render_frames"]
        if plan.shots[i].camera.type == "bezier_path":
            from tour_worker.render.splat_shot import SplatShot
            cache[i] = SplatShot(plan, plan.shots[i], n, _STATE["splat"], _STATE["supersample"], _STATE["splat_stats"])
        else:
            cache[i] = ShotWarp(plan, plan.shots[i], n, _STATE["images"])
    return cache[i]


WHIP_BLUR = 0.12        # a whip's peak motion blur, share of the frame width
PUSH_ZOOM = 0.35        # a push zooms into the outgoing shot by this much, and out of the incoming one
PUSH_TAPS = 4           # zoom samples averaged for the push's radial blur


def _whip(out: np.ndarray, inc: np.ndarray, p: float, direction: int) -> np.ndarray:
    """Both shots slide by a frame width (the camera whipping sideways): the outgoing leaves, the
    incoming arrives, with a horizontal motion blur peaking at the cut. direction +1: the picture
    moves left (the camera pans right)."""
    h, w = out.shape[:2]
    e = p * p * (3 - 2 * p)
    shift = int(round(e * w))
    canvas = np.empty_like(out)
    if direction >= 0:
        canvas[:, :w - shift] = out[:, shift:]
        canvas[:, w - shift:] = inc[:, :shift]
    else:
        canvas[:, shift:] = out[:, :w - shift]
        canvas[:, :shift] = inc[:, w - shift:]
    k = 1 + int(round(WHIP_BLUR * w * math.sin(math.pi * p)))
    return cv2.blur(canvas, (k, 1)) if k > 1 else canvas


def _zoom(img: np.ndarray, s: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = np.array([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]], np.float32)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _push(out: np.ndarray, inc: np.ndarray, p: float) -> np.ndarray:
    """Zoom through: the outgoing shot rushes in to its centre, the incoming one settles from close
    in; a radial blur (zoom samples averaged) peaks at the cut, where the two cross."""
    e = p * p * (3 - 2 * p)
    blur = PUSH_ZOOM * 0.25 * math.sin(math.pi * p)

    def rushed(img, s):
        taps = [_zoom(img, s * (1 + blur * k / max(1, PUSH_TAPS - 1))).astype(np.float32) for k in range(PUSH_TAPS)]
        return sum(taps) / len(taps)

    a = min(1.0, max(0.0, (p - 0.35) / 0.3))
    a = a * a * (3 - 2 * a)
    mix = rushed(out, 1 + PUSH_ZOOM * e) * (1 - a) + rushed(inc, 1 + PUSH_ZOOM * (1 - e)) * a
    return np.clip(mix + 0.5, 0, 255).astype(np.uint8)


def _whip_direction(shot) -> int:
    """+1 when the incoming shot's camera moves right (the whip continues its motion), else -1."""
    kf = getattr(shot.camera, "keyframes", None) or []
    return -1 if len(kf) >= 2 and kf[-1].cx < kf[0].cx else 1


def draw_frame(g: int) -> np.ndarray:
    """Output frame g of the tour (BGR uint8)."""
    plan, tl = _STATE["plan"], _STATE["timeline"]
    canvas = None
    for i, t in enumerate(tl):
        f = g - t["render_from"]
        if not 0 <= f < t["render_frames"]:
            continue
        img = _shot(i).frame(f)
        kind = plan.shots[i].transition_in.type
        if canvas is not None and kind in ("whip", "push") and f < t["fade_frames"]:
            p = (f + 1) / (t["fade_frames"] + 1)          # 0..1 through the transition, the cut at 0.5
            canvas = _whip(canvas, img, p, _whip_direction(plan.shots[i])) if kind == "whip" else _push(canvas, img, p)
            continue
        a = fade_opacity(f, t["fade_frames"])
        if a >= 1.0 or canvas is None and a >= 1.0:
            canvas = img
        else:
            base = canvas if canvas is not None else np.zeros_like(img)
            canvas = np.clip(base.astype(np.float32) * (1 - a) + img.astype(np.float32) * a + 0.5,
                             0, 255).astype(np.uint8)
    if canvas is None:
        canvas = np.zeros((plan.output.height, plan.output.width, 3), np.uint8)
    return canvas


def _encode_chunk(task) -> str:
    k, a, b, out_dir = task
    plan = _STATE["plan"]
    w, h, fps = plan.output.width, plan.output.height, plan.output.fps
    vp = video_profile(w, h, plan.total_duration_s, getattr(plan.output, "max_bytes", None))
    part = os.path.join(out_dir, f"part_{k:04d}.mp4")
    # BT.709 for HD and 4K, conversion and tags (client review 2026-11-05: the demo was tagged BT.601)
    to_yuv = ("scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,"
              "format=yuv420p")
    colour = [*COLOUR_TAGS, "-r", str(fps), "-an"]
    x264 = ["-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-maxrate", vp.maxrate, "-bufsize", vp.bufsize,
            "-profile:v", "high", "-level:v", vp.level, "-g", str(GOP_S * fps), "-keyint_min", str(fps),
            "-x264-params", f"nal-hrd=vbr:threads=1:vbv-init={VBV_INIT_CHUNK}"]
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    raw_in = list(cmd)
    if wants_hevc(plan):     # the same frames, converted once, to a second encoder in the same process
        cmd += ["-filter_complex", f"[0:v]{to_yuv},split=2[v264][v265]",
                "-map", "[v264]", *x264, *colour, part,
                "-map", "[v265]", *hevc_args(vp, fps, VBV_INIT_CHUNK), *colour, _hevc_part(part)]
    else:
        cmd += ["-vf", to_yuv, *x264, *colour, part]
    procs = [subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)]
    if wants_mls(plan):      # the MLS copy: its own x264, fed the frames before the room labels
        procs.append(subprocess.Popen(raw_in + ["-vf", to_yuv, *x264, *colour, _mls_part(part)],
                                      stdin=subprocess.PIPE, stderr=subprocess.PIPE))
    overlays = _STATE.get("overlays")
    try:
        for g in range(a, b):
            frame = draw_frame(g)
            if overlays is None:
                for p in procs:
                    p.stdin.write(frame.tobytes())
                continue
            procs[0].stdin.write(overlays.apply(frame, g).tobytes())
            if len(procs) > 1:
                procs[1].stdin.write(overlays.apply(frame, g, captions=False).tobytes())
        for p in procs:
            p.stdin.close()
    except BrokenPipeError:
        pass
    for p in procs:
        err = p.stderr.read().decode(errors="replace")
        if p.wait() != 0:
            raise WorkerError("encode_failed", err[-2000:])
    return part


# delivery loudness (client review 2026-11-05: the demo was -10.1 LUFS with a +0.4 dBTP peak, i.e.
# clipping): -14 LUFS integrated, true peak at most -1 dBTP. Normalised in two passes - measure,
# then one linear gain (loudnorm linear=true), so the music's dynamics are untouched and the
# same plan gives the same bytes. TP aims a little lower than the check so AAC's overshoot fits.
LOUDNESS_I, LOUDNESS_TP, LOUDNESS_LRA = -14.0, -1.5, 20.0
FADE_CURVE = "tri"           # linear gain: the fade is heard over its whole length (qsin held, then dropped)


def _music_graph(plan, music_path: str | None, first_input: int) -> tuple[list[str], str] | None:
    """(input args, filter graph ending in [music], 48 kHz) for the plan's music; None: no music."""
    end = round(plan.total_duration_s, 3)
    m = plan.music
    if m is None or not music_path or not os.path.exists(music_path):
        return None
    fade = min(m.fade_out_s, end)
    tail = (f"[q]atrim=0:{end},asetpts=PTS-STARTPTS,"
            f"afade=t=out:st={max(0.0, end - fade)}:d={fade}:curve={FADE_CURVE},aresample=48000[music]")
    if getattr(m, "segments", None):
        return _segment_inputs(m, music_path, first_input, tail)
    n = max(1, int(m.loops))
    loop = m.loop_s or m.track_duration_s or end
    start = max(0.0, float(getattr(m, "start_s", 0.0) or 0.0))      # sync v2 may start mid-track
    ins = []
    for _ in range(n):
        ins += ["-i", music_path]

    def trim(k: int) -> str:
        a = start if k == 0 else 0.0                                   # only the first copy starts late
        if k < n - 1:
            return f"atrim={a}:{loop},"
        return f"atrim=start={a}," if a > 0 else ""

    parts = [f"[{first_input + k}:a]" + trim(k) + f"asetpts=PTS-STARTPTS[a{k}]" for k in range(n)]
    return ins, ";".join(parts) + ";" + "".join(f"[a{k}]" for k in range(n)) + f"concat=n={n}:v=0:a=1[q];" + tail


def _loudness(ins: list[str], graph: str) -> dict:
    """loudnorm's first-pass measurement of [mix] (input_i, input_tp, input_lra, input_thresh,
    target_offset)."""
    import json
    target = f"I={LOUDNESS_I}:TP={LOUDNESS_TP}:LRA={LOUDNESS_LRA}"
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", *ins, "-filter_complex",
                        f"{graph};[mix]loudnorm={target}:print_format=json[o]", "-map", "[o]", "-f", "null", "-"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise WorkerError("encode_failed", p.stderr[-2000:])
    return json.loads(p.stderr[p.stderr.rindex("{"):p.stderr.rindex("}") + 1])


def _voice_graph(plan, voice_path: str, index: int, music: bool) -> str:
    """The voiceover from the tour's start, trimmed / padded to the tour; with music, the music
    ducks under it (a sidechain compressor keyed by the voice; voiceover.balance 0..1 sets how far)
    and the two are summed into [mix]."""
    end = round(plan.total_duration_s, 3)
    ratio = 2.0 + 14.0 * float(plan.voiceover.balance)
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"   # one format for the mix
    voice = f"[{index}:a]{fmt},atrim=0:{end},asetpts=PTS-STARTPTS,apad=whole_dur={end}"
    if not music:
        return voice + "[mix]"
    return (voice + "[vraw];[vraw]asplit=2[voice][key];"
            f"[music]{fmt}[mus];"
            f"[mus][key]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:attack=15:release=350[duck];"
            "[duck][voice]amix=inputs=2:normalize=0:duration=first[mix]")


def _audio_track(plan, music_path: str | None, out_path: str, voice_path: str | None = None) -> str:
    """The tour's audio as AAC in out_path, made once and muxed into every copy: the music (plan
    1.5) - the track queued `loops` times or played as sync v2 segments, trimmed to the tour, faded
    out - with the voiceover over it (the music ducking under the voice), normalised to LOUDNESS_I /
    LOUDNESS_TP; with neither, a silent track (players and platforms expect one)."""
    end = round(plan.total_duration_s, 3)
    built = _music_graph(plan, music_path, 0)
    voice = voice_path if (voice_path and os.path.exists(voice_path) and getattr(plan, "voiceover", None)) else None
    if built is None and voice is None:
        _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", str(end), "-i",
              "anullsrc=channel_layout=stereo:sample_rate=48000", "-c:a", "aac", "-b:a", "128k", "-ac", "2", out_path])
        return out_path
    ins, graph = built if built is not None else ([], "")
    if voice is not None:
        vg = _voice_graph(plan, voice, ins.count("-i"), built is not None)
        ins, graph = ins + ["-i", voice], (graph + ";" + vg) if graph else vg
    else:
        graph += ";[music]anull[mix]"
    m = _loudness(ins, graph)
    norm = (f"loudnorm=I={LOUDNESS_I}:TP={LOUDNESS_TP}:LRA={LOUDNESS_LRA}:measured_I={m['input_i']}:"
            f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
            f"offset={m['target_offset']}:linear=true")
    _run(["ffmpeg", "-y", "-loglevel", "error", *ins, "-filter_complex", f"{graph};[mix]{norm},aresample=48000[o]",
          "-map", "[o]", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2", "-t", str(end), out_path])
    return out_path


def _segment_inputs(m, music_path: str, first_input: int, tail: str) -> tuple[list[str], str]:
    """Sync v2 music edits: the segments back to back, each join an equal-power crossfade of
    crossfade_s CENTRED on it - each piece runs half the crossfade past its join (the next one
    starts half of it early), so every join's sound sits exactly where the planner put it."""
    segs, x = m.segments, float(m.crossfade_s or 0.0)
    h, n = x / 2, len(segs)
    ins, parts = [], []
    for k, s in enumerate(segs):
        ins += ["-i", music_path]
        a = s.from_s - (h if k > 0 else 0.0)
        b = s.to_s + (h if k < n - 1 else 0.0)
        parts.append(f"[{first_input + k}:a]atrim={a:.6f}:{b:.6f},asetpts=PTS-STARTPTS[a{k}]")
    if n == 1:
        chain = "[a0]anull[q]"
    else:
        steps, prev = [], "a0"
        for k in range(1, n):
            nxt = "q" if k == n - 1 else f"x{k}"
            steps.append(f"[{prev}][a{k}]acrossfade=d={x:.6f}:c1=qsin:c2=qsin[{nxt}]" if x > 0
                         else f"[{prev}][a{k}]concat=n=2:v=0:a=1[{nxt}]")
            prev = nxt
        chain = ";".join(steps)
    return ins, ";".join(parts) + ";" + chain + ";" + tail


def render_direct(plan, workdir: str, warnings: list | None = None, music_path: str | None = None,
                  workers: int | None = None, voice_path: str | None = None, timings=None,
                  progress=None, splat=None, splat_supersample: float = 1.5) -> str:
    """Draw and encode the whole tour; returns <workdir>/out.mp4 (and writes <workdir>/out_hevc.mp4,
    the H.265 copy with the same audio, and <workdir>/out_mls.mp4, the MLS copy without the room
    labels, when the plan asks for them). timings: a tour_worker.timing.Stopwatch (stages
    draw_encode, audio, join; info render_workers, chunks); progress(done, total): called as
    chunks finish. splat: the scene renderer for a plan 1.6 tour (render/splat_shot.make_renderer);
    its frames are drawn in this process, in order (one GPU), and fed to the same chunk encoders."""
    from contextlib import nullcontext
    stage = timings.stage if timings is not None else (lambda name: nullcontext())
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    images = os.path.join(workdir, "images")
    tl = compute_timeline(plan)
    three_d = any(s.camera.type == "bezier_path" for s in plan.shots)
    if three_d and splat is None:
        raise WorkerError("scene_render_failed", "a 3D plan needs its scene renderer")
    for shot, t in zip(plan.shots, tl):                       # validate early; collect warnings once
        if shot.camera.type != "bezier_path":
            ShotWarp(plan, shot, t["render_frames"], images, warnings)
    n = total_frames(plan)
    step = CHUNK_GOPS * GOP_S * plan.output.fps
    workers = 1 if three_d else render_workers(workers, plan.output.width, plan.output.height,
                                                hevc=wants_hevc(plan), mls=wants_mls(plan))
    out_path = os.path.join(workdir, "out.mp4")
    with tempfile.TemporaryDirectory(prefix="direct_", dir=workdir) as td:
        tasks = [(k, a, min(n, a + step), td) for k, a in enumerate(range(0, n, step))]
        if timings is not None:
            timings.info.update(render_workers=workers, chunks=len(tasks), frames=n)
        parts = []
        with stage("draw_encode"):
            if workers <= 1 or len(tasks) <= 1:
                _init(plan, images, splat, splat_supersample)
                for t in tasks:
                    parts.append(_encode_chunk(t))
                    if progress:
                        progress(len(parts), len(tasks))
                if three_d and timings is not None:
                    st = _STATE["splat_stats"]
                    timings.info.update(splat_backend=getattr(splat, "info", {}), supersample=splat_supersample,
                                        rasterise_s=round(st.get("rasterise_s", 0.0), 3),
                                        frames_rasterised=st.get("frames", 0))
            else:
                with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context("spawn"),
                                         initializer=_init, initargs=(plan, images)) as pool:
                    for part in pool.map(_encode_chunk, tasks):
                        parts.append(part)
                        if progress:
                            progress(len(parts), len(tasks))
        with stage("audio"):
            audio = _audio_track(plan, music_path, os.path.join(td, "audio.m4a"), voice_path)
        with stage("join"):
            _join(plan, parts, os.path.join(td, "parts.txt"), audio, out_path)
            if wants_hevc(plan):
                _join(plan, [_hevc_part(p) for p in parts], os.path.join(td, "parts_hevc.txt"), audio,
                      os.path.join(workdir, HEVC_NAME))
            if wants_mls(plan):
                _join(plan, [_mls_part(p) for p in parts], os.path.join(td, "parts_mls.txt"), audio,
                      os.path.join(workdir, MLS_NAME))
    return out_path


def _join(plan, parts: list[str], listing: str, audio: str, out_path: str) -> None:
    """The chunks back to back and the tour's audio track, both stream-copied, moov first."""
    with open(listing, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing, "-i", audio,
          "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-t", str(round(plan.total_duration_s, 3)),
          "-movflags", "+faststart", out_path])
