# Source this (". tools/runpod/env.sh") to load the worker's settings into the current shell:
# the env file plus the pod defaults. Everything the worker keeps on the pod lives under the
# repo's .pod/ folder (gitignored), so nothing is created anywhere else on the volume:
#   worker.env        settings + keys, in the repo folder itself (copy tools/runpod/worker.env.example)
#   .pod/models       segmentation weights (MODELS_DIR)
#   .pod/run          the service's pid, log and job records (JOBS_DIR)
#   .pod/torch_extensions  gsplat's compiled CUDA kernels (built once, on first use)
# Scene files are not kept on disk (fetched per job, deleted once loaded) unless SCENE_CACHE_DIR
# is set in the env file. start.sh sources this too. A setting already in the env file wins over these defaults.
_TW_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POD_DIR="${POD_DIR:-$_TW_REPO/.pod}"
ENV_FILE="${ENV_FILE:-$_TW_REPO/worker.env}"
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
# gsplat builds its CUDA kernels on first use: torch runs `ninja` from PATH (the venv has it), and
# keeps the build in TORCH_EXTENSIONS_DIR - on the volume, so a pod restart does not rebuild it
PATH="${VENV:-$_TW_REPO/.venv-pod}/bin:$PATH"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$POD_DIR/torch_extensions}"
# the container's CPU quota (a pod sees all host CPUs but may use ~16): thread pools sized to it,
# not to the host's count, or torch / BLAS oversubscribe the quota and crawl
_TW_CPUS="$("${VENV:-$_TW_REPO/.venv-pod}/bin/python" -c 'import sys; sys.path.insert(0, sys.argv[1]); from tour_worker.cpus import usable_cpus; print(usable_cpus())' "$_TW_REPO" 2>/dev/null || nproc)"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$_TW_CPUS}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-$_TW_CPUS}"
set +a
mkdir -p "$MODELS_DIR" "$JOBS_DIR"
echo "tour worker settings: $ENV_FILE ($_TW_CPUS CPUs usable)"
