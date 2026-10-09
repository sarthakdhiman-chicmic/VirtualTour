# Articulait Standalone Render Worker

## Repo Tree
```
.
├── bench.py
├── handler.py
├── modules
│   ├── images.py
│   ├── __init__.py
│   ├── plan_io.py
│   ├── probe.py
│   ├── remotion_render.py
│   └── storage.py
├── remotion
│   ├── package.json
│   ├── remotion.config.ts
│   ├── src
│   │   ├── index.ts
│   │   ├── KenBurnsShot.tsx
│   │   ├── math.ts
│   │   ├── Root.tsx
│   │   └── TourComposition.tsx
│   └── tsconfig.json
├── run_tests.py
└── run_tour.py
```

## How to run tests and the CLI

**Run Tests (requires no S3 or GPU)**
```bash
uv run --with pydantic --with pillow --with boto3 python3 run_tests.py
```

**Run CLI Locally**
```bash
uv run --with pydantic --with pillow --with boto3 python3 run_tour.py --plan my_plan.json --images-dir ./photos --out ./out --scale 0.5
```

## Real Test Results & Determinism Result
- **Ran 8 tests in 22.3s** (All tests `OK`)
- **Plan IO**: Validates `plan_hash`, rejects missing fields, ignores unknown fields safely.
- **Camera Window**: Successfully rejects out-of-bounds `cx`/`cy` constraints.
- **Frame Counts**: cumulative rounding sums exactly match `round(total * fps)`.
- **End-to-End**: Solid-colour quadrant synthetic image rendered correctly with the first frame matching the top-left color.
- **Determinism Check**: `IDENTICAL`. The per-frame hashes of two identical renders evaluated via `ffmpeg -f framemd5` perfectly matched.
- **Probe Verification**: Correctly throws an exception when the rendered video metadata does not match the plan's specs (wrong size, codec, fps, etc).

## Benchmark Numbers
- **90s at scale 0.5**: 203.59s (13.26 FPS)
- **10s slice at full 4K**: 54.79s (5.48 FPS)
- **CPU count**: 20 (GPU not used)
- **ESTIMATE for 90s at 4K**: ~493 seconds (8.2 minutes)
- **SOW Target**: <180 seconds.
- **Difference**: We are ~313 seconds off target. To hit the target for a 90-second 4K video, hardware acceleration (such as NVENC on an NVIDIA GPU) and potentially Remotion's lambda / concurrency scaling across nodes will be required, as CPU rendering alone is not fast enough.

## Decisions Made
- **Remotion Config**: Pinned Remotion and `@remotion/cli` to `4.0.141`. I also pinned `typescript@5.3.3` because newer TS versions break Remotion's internal Webpack `esbuild-loader` module (`sys.readFile` is undefined in newer TS APIs).
- **Remotion License Terms**: For a commercial SaaS product, Remotion **requires a paid commercial license** (either Startup or Company tier, depending on revenue/funding). You cannot use it for free for commercial automated rendering.
- **Image Preparation**: EXIF orientation is correctly applied before any dimension checks using `ImageOps.exif_transpose`. Images with long edges over 5000px are downscaled using `LANCZOS` to keep memory in check.
- **Timeouts**: Added `--timeout 120000` and `--concurrency 2` to the Remotion command so that 4K frames don't crash the renderer on a resource-constrained CPU machine.
- **Storage**: Abstracted storage into a `Storage` protocol with an `S3Storage` and `LocalMapStorage`. This allows testing and CLI runs without needing AWS credentials.

## What I could not run and why
- **Real S3 and RunPod integrations**: Could not test actual S3 fetch/upload or RunPod invocations since I do not have your AWS credentials or RunPod serverless environment access. The worker is mocked to use the local filesystem during tests.
- **Hardware Acceleration**: The tests and benchmarks were run on a CPU environment. I could not run or test GPU-accelerated `h264_nvenc` codecs.
