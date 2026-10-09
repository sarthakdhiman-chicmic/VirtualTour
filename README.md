# Articulait Cinematic Tour Worker

## Run
Run the handler locally using SDK test mode:
`WORKER_STORAGE=local IMAGES_DIR=/tmp python3 handler.py --test_input test_input.json`

Or using the CLI:
`python3 run_tour.py --plan plan.json --images-dir ./images --out ./out`

## Test
Run Python tests with pytest:
`uv run pytest tests/`

Run TypeScript tests:
`cd remotion && npm test`

## Benchmark
Run the benchmark script:
`uv run python bench.py`
`uv run python sweep.py`
## Environment Variables
- `WORKER_STORAGE`: Set to `local` to use local storage mock.
- `IMAGES_DIR`: Directory for local images when `WORKER_STORAGE=local`.
- `AWS_ACCESS_KEY_ID`: AWS access key.
- `AWS_SECRET_ACCESS_KEY`: AWS secret key.
- `AWS_DEFAULT_REGION`: AWS region.
