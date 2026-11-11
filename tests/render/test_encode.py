"""tour_worker/render/encode.py: the delivery file is capped, keyframed every 2 s, faststart, deterministic."""
import os
import subprocess
import tempfile

from tour_worker.render.encode import GOP_S, deliver, playback_stats, vbv_min_fullness


def _master(path, seconds=6):
    # 1080p30 noise: harder to compress than any photo tour, so it pushes against the cap
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=1920x1080:rate=30:duration={seconds},noise=alls=40:allf=t",
                    "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={seconds}", "-shortest",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
                   check=True)


def test_delivery_is_capped_keyframed_faststart_and_deterministic():
    with tempfile.TemporaryDirectory() as td:
        m = os.path.join(td, "master.mp4")
        _master(m)
        a, b = os.path.join(td, "a.mp4"), os.path.join(td, "b.mp4")
        deliver(m, a, 30)
        deliver(m, b, 30)
        st = playback_stats(a)
        assert st["faststart"] is True
        assert st["max_keyframe_gap_s"] <= GOP_S + 0.05
        assert st["vbv_min_fullness"] >= 0     # a 10 Mbps reader with a 20 Mbit buffer never stalls
        assert st["profile"] == "High" and int(st["level"]) == 41
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read()


def test_vbv_simulation_flags_a_stall():
    fps = 30
    steady = [10e6 / fps] * 90                       # exactly the cap: never drains
    assert abs(vbv_min_fullness(steady, fps) - (0.9 - 1 / 60)) < 1e-9   # first frame leaves before the refill
    burst = [10e6 / fps] * 30 + [3e6] * 10 + [10e6 / fps] * 30   # 30 Mbit in a third of a second
    assert vbv_min_fullness(burst, fps) < 0


def test_the_encode_profile_follows_the_frame_size_and_a_file_size_limit():
    from tour_worker.render.encode import video_profile
    for w, h in ((1920, 1080), (1080, 1920), (1440, 1080), (1280, 720)):   # up to 1080p: unchanged
        assert video_profile(w, h) == video_profile(1920, 1080)
        assert (video_profile(w, h).level, video_profile(w, h).maxrate) == ("4.1", "10M")
    for w, h in ((3840, 2160), (2880, 2160), (2160, 3840)):                 # 4K: Level 5.1, 40 Mbps
        p = video_profile(w, h)
        assert (p.level, p.maxrate, p.bufsize) == ("5.1", "40M", "80M")
    # Instagram Reels: 100 MB over 90 s - the cap leaves room for the buffer and the audio
    p = video_profile(1080, 1920, 90.0, 100_000_000)
    cap = int(p.maxrate[:-1]) * 1000
    assert 7_000_000 < cap < 10_000_000
    worst = cap * 90 + 0.9 * 2 * cap + 128_000 * 90                         # bits
    assert worst / 8 <= 100_000_000
    assert video_profile(1080, 1920, 20.0, 287_000_000).maxrate == "10M"   # a limit far away: no change


def test_a_4k_delivery_is_level_51_and_plays_within_its_cap():
    from tour_worker.render.encode import video_profile
    with tempfile.TemporaryDirectory() as td:
        m = os.path.join(td, "master.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=3840x2160:rate=30:duration=2,noise=alls=30:allf=t",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p", m], check=True)
        out = os.path.join(td, "4k.mp4")
        vp = video_profile(3840, 2160)
        deliver(m, out, 30, vp)
        st = playback_stats(out, vp.maxrate, vp.bufsize)
        assert int(st["level"]) == 51 and st["vbv_min_fullness"] >= 0


def test_a_file_size_limit_holds_on_hard_to_compress_video():
    from tour_worker.render.encode import video_profile
    with tempfile.TemporaryDirectory() as td:
        m = os.path.join(td, "master.mp4")
        _master(m)                                                          # 6 s of 1080p noise
        out = os.path.join(td, "capped.mp4")
        vp = video_profile(1920, 1080, 6.0, 3_000_000)
        deliver(m, out, 30, vp)
        st = playback_stats(out, vp.maxrate, vp.bufsize)
        assert st["vbv_min_fullness"] >= 0 and st["size_bytes"] <= 3_000_000
