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
bar where its beat stops), trimmed to the tour, equal-power fade-out - as AAC; else a silent track
(players and platforms expect one, as Remotion's renders had).
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import cv2
import numpy as np

from tour_worker.errors import WorkerError
from tour_worker.render.encode import (BUFSIZE, CHUNK_GOPS, CRF, GOP_S, LEVEL, MAXRATE, PRESET, VBV_INIT_CHUNK,
                                       _run)
from tour_worker.render.frames import ShotWarp, render_workers
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


def total_frames(plan) -> int:
    return compute_timeline(plan)[-1]["nominal_end"] if plan.shots else 0


def _init(plan, images: str) -> None:
    cv2.setNumThreads(1)
    _STATE.clear()
    _STATE.update(plan=plan, images=images, timeline=compute_timeline(plan), shots={})


def _shot(i: int) -> ShotWarp:
    cache = _STATE["shots"]
    if i not in cache:
        for k in [k for k in cache if k < i - 1]:            # shots run in order: keep the last two
            del cache[k]
        plan = _STATE["plan"]
        cache[i] = ShotWarp(plan, plan.shots[i], _STATE["timeline"][i]["render_frames"], _STATE["images"])
    return cache[i]


def draw_frame(g: int) -> np.ndarray:
    """Output frame g of the tour (BGR uint8)."""
    plan, tl = _STATE["plan"], _STATE["timeline"]
    canvas = None
    for i, t in enumerate(tl):
        f = g - t["render_from"]
        if not 0 <= f < t["render_frames"]:
            continue
        a = fade_opacity(f, t["fade_frames"])
        img = _shot(i).frame(f)
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
    part = os.path.join(out_dir, f"part_{k:04d}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-vf", "scale=out_color_matrix=bt601:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,"
                  "format=yuv420p",
           "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-maxrate", MAXRATE, "-bufsize", BUFSIZE,
           "-profile:v", "high", "-level:v", LEVEL, "-g", str(GOP_S * fps), "-keyint_min", str(fps),
           "-x264-params", f"nal-hrd=vbr:threads=1:vbv-init={VBV_INIT_CHUNK}",
           "-colorspace", "bt470bg", "-color_primaries", "bt470bg", "-color_trc", "smpte170m",
           "-color_range", "tv", "-r", str(fps), "-an", part]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for g in range(a, b):
            proc.stdin.write(draw_frame(g).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise WorkerError("encode_failed", err[-2000:])
    return part


def _audio_args(plan, music_path: str | None, first_input: int) -> tuple[list[str], list[str]]:
    """(input args, output args) for the tour's audio track."""
    end = round(plan.total_duration_s, 3)
    m = plan.music
    if m is None or not music_path or not os.path.exists(music_path):
        return (["-f", "lavfi", "-t", str(end), "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"],
                ["-map", f"{first_input}:a:0", "-c:a", "aac", "-b:a", "128k", "-ac", "2"])
    n = max(1, int(m.loops))
    loop = m.loop_s or m.track_duration_s or end
    ins = []
    for _ in range(n):
        ins += ["-i", music_path]
    parts = [f"[{first_input + k}:a]" + (f"atrim=0:{loop}," if k < n - 1 else "") + f"asetpts=PTS-STARTPTS[a{k}]"
             for k in range(n)]
    fade = min(m.fade_out_s, end)
    graph = (";".join(parts) + ";" + "".join(f"[a{k}]" for k in range(n)) + f"concat=n={n}:v=0:a=1[q];"
             + f"[q]atrim=0:{end},asetpts=PTS-STARTPTS,"
             + f"afade=t=out:st={max(0.0, end - fade)}:d={fade}:curve=qsin,aresample=48000[music]")
    return ins, ["-filter_complex", graph, "-map", "[music]", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ac", "2"]


def render_direct(plan, workdir: str, warnings: list | None = None, music_path: str | None = None,
                  workers: int | None = None) -> str:
    """Draw and encode the whole tour; returns <workdir>/out.mp4."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    images = os.path.join(workdir, "images")
    tl = compute_timeline(plan)
    for shot, t in zip(plan.shots, tl):                       # validate early; collect warnings once
        ShotWarp(plan, shot, t["render_frames"], images, warnings)
    n = total_frames(plan)
    step = CHUNK_GOPS * GOP_S * plan.output.fps
    workers = render_workers(workers)
    out_path = os.path.join(workdir, "out.mp4")
    with tempfile.TemporaryDirectory(prefix="direct_", dir=workdir) as td:
        tasks = [(k, a, min(n, a + step), td) for k, a in enumerate(range(0, n, step))]
        if workers <= 1 or len(tasks) <= 1:
            _init(plan, images)
            parts = [_encode_chunk(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context("spawn"),
                                     initializer=_init, initargs=(plan, images)) as pool:
                parts = list(pool.map(_encode_chunk, tasks))
        listing = os.path.join(td, "parts.txt")
        with open(listing, "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        a_in, a_out = _audio_args(plan, music_path, 1)
        _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing, *a_in,
              "-map", "0:v:0", "-c:v", "copy", *a_out, "-t", str(round(plan.total_duration_s, 3)),
              "-movflags", "+faststart", out_path])
    return out_path
