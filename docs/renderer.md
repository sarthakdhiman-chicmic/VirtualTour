# The tour renderer

How a tour plan becomes the delivered video, and why the renderer is built this way. Read this before
changing anything under `tour_worker/render/`. The plan format itself is in `tour_worker/contract/`.

## What it does

`tour_render` (`tour_worker/jobs/render.py`) takes a **plan** (JSON from `tour_plan`, validated
against the contract) and produces:

| File | When | What |
| --- | --- | --- |
| `master.mp4` | always | H.264, the branded tour |
| `master_mls.mp4` | `output.mls_copy` | the unbranded MLS copy: the same frames without the room labels, agent card or logo |
| `master_hevc.mp4` | `output.hevc` | an H.265 copy of the master, from the same frames |

The same plan always gives byte-identical files. Every file is checked (`render/probe.py verify()`)
before it is uploaded. A failure returns `{"status": "failed", "error": {"code", "message"}}`.

## Why our own renderer

The first renderer was a Remotion (browser) composition. Each frame went OpenCV prewarp → JPEG →
Chrome → JPEG (quality 80) → master H.264 → delivery H.264. That caused three problems:

- **Slow:** Chrome was the slowest stage (8-10 minutes for a 75 s tour on a dev machine).
- **Not reproducible:** the same plan gave a different file each time, from Chrome's scheduling and
  WebGL.
- **Three lossy steps** before the delivery encode.

Every shot could already be drawn exactly by OpenCV (`ShotWarp`). So frames now go straight from
OpenCV to the delivery encoder:
- one lossy step;
- byte-identical output;
- drawing and encoding in parallel processes.

The composition layer's jobs are each covered by a module here:

| The composition layer provided | Where it is now |
| --- | --- |
| Shots stacked in order, fades, cuts | `direct.py draw_frame`, `timeline.py` |
| Template layouts: title card, end card, lower thirds | `overlays.py` (title card, end card, room-label lower thirds; fonts and placements chosen per plan) |
| Agent and brokerage branding, logo placement | `overlays.py` (end card, corner logo), from `output.branding` |
| Voiceover mixing | `direct.py _voice_graph`: the music ducks under the voice (sidechain compressor), then the mix is normalised |
| Branded and unbranded MLS versions | `direct.py _encode_chunk`: one set of frames, two encoders (with and without the branded overlays) |
| Music | `direct.py _music_graph` / `_segment_inputs`: the beat-synced track and its edits, faded out |

Remotion is kept on the `archive/remotion-renderer` branch.

## The pipeline

```
plan ──► fetch photos / masks / music / scene (jobs/render.py)
     ──► per shot, a frame source:
            ShotWarp   (render/frames.py)      photo shots: camera + motion regions
            SplatShot  (render/splat_shot.py)  3D shots: a camera path through a splat scene
     ──► draw_frame(g): compose the shots at output frame g (cuts, fades, whip, push)
     ──► overlays (render/overlays.py): branded frame, and the MLS frame
     ──► chunks of whole GOPs, each its own process: draw its frames, pipe them to x264
     ──► audio track, made once (music + voiceover, loudness-normalised, PCM)
     ──► per copy: join the chunks (stream copy) + encode the audio aligned to the video
     ──► verify() each file ──► upload
```

### Frame sources

Both kinds of source have one interface, `frame(f)`, which returns BGR uint8 at the output size.
Composition, overlays, encoding and audio don't know which kind they are drawing.

**ShotWarp (photos).** The camera is one homography from output pixels to source pixels per frame:
- the camera types are `ken_burns`, `keyframe_path`, `compound` and `bezier_path`, with crop, roll
  or rotate projections; the path math is in `camera_path.py`;
- beat-sync retiming (`time_warp`) maps output time to the move's progress, monotonically;
- the whole frame is drawn with one `warpPerspective`;
- where a motion region (water, sky, foliage, curtains, fire; `tour_worker/motion/`) can be on
  screen, that window is redrawn with a single `cv2.remap` whose maps are the camera homography
  plus the region's displacement. The content moves inside its mask with one resample, so there is
  no double blur;
- regions too faint to read at the output size are dropped (`visible_motion_px`).

**SplatShot (3D).** A Bezier camera path flies through a Gaussian-splat scene:
- **Backend:** gsplat on CUDA in production, with a CPU reference rasteriser (`splat_cpu.py`)
  for checks and previews.
- **Supersampling:** each frame is rasterised at 1.5× the output size and reduced by area
  averaging on the GPU, which calms the shimmer of small splats. Only the finished 8-bit frame
  comes back to the CPU.
- **One GPU, many encoders:** `GPU_LOCK` serialises GPU access, so frames are identical however
  many encoders run.
- **Motion:** 3D motion (curtains, plants, flames) moves individual Gaussians, from the scene's
  labels (`scene/motion_label.py`).

### Composition and transitions

`timeline.py` turns shot durations into exact frame ranges: frame `g` belongs to shot *k* when
`g` is in its range. The shot's cut frame is `round(start × fps)`, so cuts land on the frames the
beat sync aimed for.

| Transition | How it is drawn |
| --- | --- |
| cut | the next shot replaces the last at its first frame (the default) |
| fade | smoothstep opacity over the fade frames (used for daylight to twilight, and between parts of the home in 3D tours) |
| whip | both shots slide by a frame width in the incoming camera's direction, with a horizontal motion blur peaking at the cut (the midpoint) |
| push | the outgoing shot rushes in to its centre while the incoming one settles from close |

The planner chooses transitions (`planner/transitions.py`, `planner3d/plan.py`); the renderer
only draws them.

### Overlays

`overlays.py` builds each overlay once per process: the text is rendered with Pillow in the
bundled OFL fonts, into small premultiplied layers. The layers are composited per frame in float32,
the same arithmetic every run. Sizes are set for a 1080 px short side and scale with the output.

| Overlay | Branded master | MLS copy |
| --- | --- | --- |
| Room labels ("02 \| Kitchen", script font, animated write-on), `output.room_labels` | yes | no |
| "VIRTUALLY STAGED" on staged photos (a disclosure) | yes | yes |
| Title card: address, locality, optional kicker ("Just Listed"), over the opening shot | yes | yes |
| End card: agent photo, name, phone, email, brokerage logo, last 3.5 s | yes | no |
| Corner logo | yes | no |

Fonts: `sacramento`, `mr_dafoe`, `pacifico`, `kaushan_script`. Placements: bottom or top, left,
centre or right (right placements mirror the label).

### Audio

`_audio_track` makes the tour's audio once, as PCM:
1. **Music:** the beat-synced track. It's either queued copies cut on the bar where the beat
   stops, or sync v2's segments joined with centred 30 ms equal-power crossfades. It's trimmed
   to the tour and faded out over `music.fade_out_s` (3 s).
2. **Voiceover:** when there is one, it's mixed over the music, and the music ducks under it
   (sidechain compression; `voiceover.balance` sets how far).
3. **Loudness:** normalised in two passes (measure, then one linear gain) to −14 LUFS
   integrated, with a true peak at or below −1.5 dBTP. A linear gain keeps the music's
   dynamics and gives the same bytes every run.

Each copy encodes this PCM to AAC itself (`encode.mux_aligned`).

### Encoding

| Setting | Value | Why |
| --- | --- | --- |
| Codec | H.264 High (`libx264`, preset `slow`, CRF 20) | quality-targeted |
| Rate cap | VBV: 10 Mbps / 20 Mbit at 1080p (Level 4.1); 40 Mbps at 4K (Level 5.1) | no stretch of video exceeds what a player can stream; checked by simulating a player's buffer |
| Keyframes | at least every 2 s, and at every cut | fast start and seeking |
| Colour | BT.709 conversion and tags for HD and 4K | correct colour on every display |
| Container | MP4, moov first (faststart), **no edit lists** | plays before the download finishes; in sync in every player (below) |
| Audio | AAC stereo, 48 kHz | |

**Determinism.** x264 runs **single-threaded per chunk**. Every multi-threaded x264 mode with a VBV
cap gave 4-5 distinct outputs in 5 encodes of the same clip; one thread is byte-identical. Speed
comes from parallel chunks instead:
- each chunk is `CHUNK_GOPS` whole GOPs, so it starts on a keyframe;
- each chunk runs in its own process, which draws its frames and pipes them to x264;
- the chunks are joined by stream copy.

The number of processes is `render_workers()`: at most 8, and never more than the CPUs this
container may use (`cpus.py` reads the cgroup quota, not the host's count) or than 60% of its
memory holds (1080p ≈ 1.2 GB a process, 4K ≈ 4 GB).

**The MLS and H.265 copies.** Frames are drawn once. Each chunk's frames go to the master's x264,
then, with the room labels and branding left out, to the MLS copy's own x264. The H.265 copy shares
the master's colour conversion (one `split`).

**A/V sync without edit lists (`mux_aligned`).** Two encoder delays put the sound out of step
with the picture:
- **Video:** H.264 B-frames start the picture 66.7 ms into the track.
- **Audio:** AAC prepends 1024 samples of encoder priming (21.3 ms at 48 kHz).

MP4 edit lists tell a player to skip both, but some players ignore them (some browsers and editors,
older VLC). In those players every cut came about 45 ms after its beat. So instead:
- the video's first frame is placed at 0 with negative composition offsets;
- the first 1024 samples of the audio are dropped before the AAC encode, so the priming takes
  their place;
- the file carries no edit lists, so decoded sample *n* is the tour's sample *n* in every player.

The beat sync then places each cut on the music's audible attack and slides the track by up to
half a frame so the cuts sit nearest their beats. In practice cuts land within 15 ms.

## Checks (`probe.verify`)

Every copy, before upload, is checked for:
- **video:** codec (and the `hvc1` tag for H.265), `yuv420p`, size and frame rate as planned,
  BT.709 tags, duration;
- **container:** faststart, keyframe gap ≤ 2 s, the VBV buffer never running dry, the H.264 /
  H.265 level, the size limit (`max_bytes`, e.g. Instagram's 100 MB);
- **audio:** present, integrated loudness within tolerance of −14 LUFS, true peak ≤ −1 dBTP.

Any failure fails the job (`verification_failed`).

## Determinism: what it depends on

Byte-identical output holds for a given plan, worker version and host software (ffmpeg / x264
build; for 3D, GPU model, driver and gsplat version). The things that keep it:
- single-threaded x264 per chunk, and chunk boundaries fixed by the plan, not the worker count;
- float32 compositing in a fixed order, and overlays drawn once;
- seeded motion (seeds come from the listing and plan, not the clock);
- the linear-gain loudness pass;
- one GPU lock for splat frames.

Never add threaded x264, wall-clock seeds, or anything that depends on the number of workers.
`tests/render/test_direct.py` renders the same plan twice and compares bytes.

## Performance

| | Where the time goes |
| --- | --- |
| Photo tours | drawing (OpenCV) and x264, both on CPU, in parallel chunks; the MLS copy is a second x264 encode of every frame |
| 3D tours | gsplat rasterising on the GPU (about 0.15 s a 4K frame at 1.5× supersampling on an A100: 284 s for a 65 s tour); encoding on CPU |

Notes for sizing hosts:
- **No hardware encoder on the A100:** it has no NVENC, so all encoding is x264 on CPU. Encode
  speed scales with the container's CPU quota, up to the 8-process cap in `render_workers`.
- **Memory:** 4K needs about 4 GB per render process. `render_workers` lowers the count to fit,
  so a render never runs the host out of memory.
- **Per-stage timings:** every job result reports them (`timings.stages`).
  `tools/timing_report.py` turns them into a table.

## Changing the renderer

- **A new transition:** draw it in `direct.py draw_frame` from the two frame sources (keep it a
  pure function of the frame index). Add its name to the contract, and let the planner choose it.
- **A new overlay:** build it once in `Overlays.__init__` as layers, and apply it in
  `Overlays.apply`. Decide whether the MLS copy keeps it.
- **A new camera type:** add its math to `camera_path.py` (one homography per frame) or a new
  frame source with the `frame(f)` interface.
- **Encoder settings:** change `encode.py` constants, then confirm `verify()` and the
  byte-identical test still pass at 1080p and 4K.

After any change, run `pytest -q tests/render`. For anything visual, render a real tour and
compare frames.

## Files

| File | Role |
| --- | --- |
| `jobs/render.py` | the job: fetch, render, verify, upload |
| `render/direct.py` | composition, transitions, chunked encode, audio, join |
| `render/frames.py` | `ShotWarp` (photo shots), worker and memory sizing |
| `render/camera_path.py` | camera path math shared by the planner, the validator and the renderer |
| `render/timeline.py` | shot durations to exact frame ranges |
| `render/overlays.py` | room labels, staged label, title and end cards, corner logo |
| `render/encode.py` | delivery settings, VBV simulation, `mux_aligned`, `deliver()` re-encode |
| `render/probe.py` | `verify()` and loudness measurement |
| `render/splat_shot.py` | 3D shots on gsplat; `make_renderer` |
| `render/splat_cpu.py` | the CPU reference splat rasteriser |
| `tour_worker/motion/` | photo motion: water, sky, foliage, fabric, fire |
