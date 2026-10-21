"""modules/direct_render.py: OpenCV draws every frame, ffmpeg encodes (no Remotion / Chrome)."""
import hashlib
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
import pytest

from modules.direct_render import render_direct, total_frames
from modules.perceptual.effects import mask_relpath
from modules.plan_io import compute_plan_hash, parse_plan
from modules.prewarp import ShotWarp
from modules.probe import verify
from modules.timeline import compute_timeline

pytestmark = pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
                                reason="ffmpeg not installed")
SW, SH = 960, 640


def _plan(music=None, fade=0.0):
    kb = {"type": "ken_burns", "easing": "ease_in_out", "start": {"cx": 0.45, "cy": 0.5, "zoom": 1.15},
          "end": {"cx": 0.5, "cy": 0.5, "zoom": 1.02}}
    kp = {"type": "keyframe_path", "move": "reveal_crane",
          "keyframes": [{"t": 0, "cx": 0.5, "cy": 0.55, "zoom": 1.1}, {"t": 1, "cx": 0.5, "cy": 0.5, "zoom": 1.03}]}
    region = {"id": "obj_000", "effect": "fabric_sway", "mask_uri": "s3://b/m.png", "bbox": [0.2, 0.1, 0.45, 0.8],
              "amplitude_px": 5.0, "freq_hz": 0.3, "anchor": "top", "seed": 7, "label": "curtain"}
    shots = [
        {"shot_id": "s01", "order": 1, "asset_id": "a", "image_uri": "a.jpg", "source": {"width": SW, "height": SH},
         "room_key": "r", "room_type": "living_room", "narrative_stage": "living", "role": "hero", "duration_s": 1.5,
         "camera": kb, "transition_in": {"type": "cut", "duration_s": 0.0}},
        {"shot_id": "s02", "order": 2, "asset_id": "b", "image_uri": "b.jpg", "source": {"width": SW, "height": SH},
         "room_key": "r2", "room_type": "bedroom", "narrative_stage": "private", "role": "hero", "duration_s": 1.5,
         "camera": kp, "transition_in": {"type": "fade" if fade else "cut", "duration_s": fade},
         "motion_regions": [region]},
    ]
    plan = {"plan_version": "1.5" if music else "1.4", "listing_id": "t", "mode": "cinematic", "preset": "p",
            "output": {"width": 640, "height": 360, "fps": 30, "codec": "h264"}, "total_duration_s": 3.0,
            "shots": shots, "warnings": [], "provenance": {"scoring_config_version": "0", "planner_config_version": "0"}}
    if music:
        plan["music"] = music
    plan["provenance"]["plan_hash"] = compute_plan_hash(plan)
    return parse_plan(plan)


def _workdir(td):
    os.makedirs(os.path.join(td, "images", "masks"))
    rng = np.random.default_rng(3)
    for name in ("a.jpg", "b.jpg"):
        img = cv2.GaussianBlur((rng.random((SH, SW, 3)) * 255).astype(np.uint8), (0, 0), 2)
        cv2.imwrite(os.path.join(td, "images", name), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    m = np.zeros((SH, SW), np.uint8)
    m[64:512, 192:432] = 255
    cv2.imwrite(os.path.join(td, "images", mask_relpath("s02", "obj_000")), m)


def _frames(path, n):
    p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(n, 360, 640, 3)


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
                         capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: s for s in json.loads(out)["streams"]}


def test_direct_render_is_valid_deterministic_and_matches_opencv():
    plan = _plan()
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        _workdir(a), _workdir(b)
        out_a = render_direct(plan, a, workers=2)
        out_b = render_direct(plan, b, workers=1)
        assert hashlib.md5(open(out_a, "rb").read()).hexdigest() == hashlib.md5(open(out_b, "rb").read()).hexdigest()
        verify(out_a, plan)                                         # the production playback checks
        n = total_frames(plan)
        frames = _frames(out_a, n)
        tl = compute_timeline(plan)
        for i, shot in enumerate(plan.shots):                       # decoded == OpenCV's drawing
            sw = ShotWarp(plan, shot, tl[i]["render_frames"], os.path.join(a, "images"))
            for f in (0, tl[i]["render_frames"] // 2):
                ref = sw.frame(f).astype(float)
                got = frames[tl[i]["render_from"] + f].astype(float)
                psnr = 10 * np.log10(255 ** 2 / max(1e-9, ((ref - got) ** 2).mean()))
                assert psnr > 32, (shot.shot_id, f, psnr)
        streams = _probe(out_a)
        assert "audio" in streams                                   # a silent track, as before


def test_music_is_queued_trimmed_and_as_long_as_the_video():
    with tempfile.TemporaryDirectory() as td:
        _workdir(td)
        track = os.path.join(td, "track.wav")                       # 1.2 s: needs 3 copies for 3 s
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
                        "-ac", "2", track], check=True)
        music = {"track_id": "t", "uri": "file://" + track, "end_s": 3.0, "loops": 3, "loop_s": 1.0,
                 "track_duration_s": 1.2, "fade_out_s": 0.5, "synced": True}
        plan = _plan(music=music)
        out = render_direct(plan, td, music_path=track, workers=1)
        s = _probe(out)
        assert s["audio"]["codec_name"] == "aac"
        assert abs(float(s["audio"]["duration"]) - float(s["video"]["duration"])) <= 1 / 30 + 0.03


def test_fades_blend_like_the_browser_composition():
    plan = _plan(fade=0.4)
    with tempfile.TemporaryDirectory() as td:
        _workdir(td)
        out = render_direct(plan, td, workers=1)
        verify(out, plan)


def _warped_plan(warp):
    raw = _plan().model_dump(mode="json")
    raw["shots"][1]["camera"]["time_warp"] = warp
    raw["provenance"]["plan_hash"] = compute_plan_hash(raw)
    return parse_plan(raw)


def test_time_warp_keeps_the_move_and_lands_it_on_the_cut():
    """Beat sync: a 1.5 s shot planned as a 1.8 s move (hook earlier than planned). The first 40%
    of real time plays at normal pace, the rear catches up; first and last frames unchanged."""
    planned, synced = 1.8, 1.5
    h = 0.4 * synced
    warp = [[0.0, 0.0], [h / synced / 2, h / planned / 2], [h / synced, h / planned], [1.0, 1.0]]
    plain, warped = _plan(), _warped_plan(warp)
    tl = compute_timeline(plain)
    with tempfile.TemporaryDirectory() as td:
        _workdir(td)
        n = tl[1]["render_frames"]
        a = ShotWarp(plain, plain.shots[1], n, os.path.join(td, "images"))
        b = ShotWarp(warped, warped.shots[1], n, os.path.join(td, "images"))
    flat = lambda fr: [*fr[0], fr[1]]                                     # noqa: E731
    assert flat(a.frames[0]) == pytest.approx(flat(b.frames[0]))          # same start
    assert flat(a.frames[-1]) == pytest.approx(flat(b.frames[-1]))        # lands where planned
    za = [f[0][2] for f in a.frames]
    zb = [f[0][2] for f in b.frames]
    mid = n // 3                                   # early on the warped move is BEHIND the plain one
    assert abs(zb[mid] - za[0]) < abs(za[mid] - za[0])
    assert all(x >= y - 1e-9 for x, y in zip(zb, zb[1:])) or all(x <= y + 1e-9 for x, y in zip(zb, zb[1:]))


def test_a_bad_time_warp_is_rejected():
    with pytest.raises(ValueError, match="time_warp"):
        _warped_plan([[0.0, 0.0], [0.6, 0.7], [0.5, 0.8], [1.0, 1.0]])
    with pytest.raises(ValueError, match="time_warp"):
        _warped_plan([[0.0, 0.1], [1.0, 1.0]])


def _punched_plan(punches):
    raw = _plan().model_dump(mode="json")
    raw["shots"][1]["camera"]["punches"] = punches
    raw["provenance"]["plan_hash"] = compute_plan_hash(raw)
    return parse_plan(raw)


def test_punch_zooms_in_on_the_hit_and_settles():
    """Beat sync: a shot opening on a drop punches in 2.5% on its first frame and eases back; an
    accent mid-shot rises over its attack to its peak on the hit. Zoom only (centre unchanged)."""
    plain = _plan()
    punched = _punched_plan([{"at_s": 0.0, "amount": 0.025, "attack_s": 0.0, "decay_s": 0.35},
                             {"at_s": 1.0, "amount": 0.02, "attack_s": 0.067, "decay_s": 0.35}])
    tl = compute_timeline(plain)
    with tempfile.TemporaryDirectory() as td:
        _workdir(td)
        n = tl[1]["render_frames"]
        a = ShotWarp(plain, plain.shots[1], n, os.path.join(td, "images"))
        b = ShotWarp(punched, punched.shots[1], n, os.path.join(td, "images"))
    ratio = [fb[0][2] / fa[0][2] for fa, fb in zip(a.frames, b.frames)]
    assert ratio[0] == pytest.approx(1.025)                   # the hit: full punch on the cut
    assert ratio[15] == pytest.approx(1.0)                    # settled by 0.5 s
    assert ratio[30] == pytest.approx(1.02)                   # the accent's peak on its beat
    assert 1.0 < ratio[29] < 1.02                             # rising into it
    assert all(r >= 1.0 - 1e-12 for r in ratio)               # zoom in only
    assert all(fa[0][:2] == fb[0][:2] for fa, fb in zip(a.frames, b.frames))


def test_a_bad_punch_is_rejected():
    with pytest.raises(ValueError, match="punch"):
        _punched_plan([{"at_s": 0.0, "amount": 0.2, "decay_s": 0.35}])
