# Source this (". tools/runpod/env.sh") to load the worker's settings into the current shell:
# the env file plus the pod defaults. Everything the worker keeps on the pod lives under the
# repo's .pod/ folder (gitignored), so nothing is created anywhere else on the volume:
#   .pod/worker.env   settings + keys (copy tools/runpod/worker.env.example)
#   .pod/models       segmentation weights (MODELS_DIR)
#   .pod/run          the service's pid, log and job records (JOBS_DIR)
# Scene files are not kept on disk (fetched per job, deleted once loaded) unless SCENE_CACHE_DIR
# is set in the env file. start.sh sources this too. A setting already in the env file wins over these defaults.
_TW_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POD_DIR="${POD_DIR:-$_TW_REPO/.pod}"
ENV_FILE="${ENV_FILE:-$POD_DIR/worker.env}"
if [ ! -f "$ENV_FILE" ]; then
    echo "no env file at $ENV_FILE: cp tools/runpod/worker.env.example $ENV_FILE and fill it in" >&2
    return 1 2>/dev/null || exit 1
fi
set -a
. "$ENV_FILE"
MODELS_DIR="${MODELS_DIR:-$POD_DIR/models}"
JOBS_DIR="${JOBS_DIR:-$POD_DIR/run/jobs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SPLAT_BACKEND="${SPLAT_BACKEND:-gsplat}"
set +a
mkdir -p "$MODELS_DIR" "$JOBS_DIR"
