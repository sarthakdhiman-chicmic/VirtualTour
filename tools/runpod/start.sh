#!/usr/bin/env bash
# Start / stop / check the tour worker's HTTP service on a RunPod GPU pod (no container image).
#
#   tools/runpod/start.sh            start in the background, wait until it answers, print /health
#   tools/runpod/start.sh stop       stop it
#   tools/runpod/start.sh status     is it running; /health
#   tools/runpod/start.sh logs       follow the log
#
# Settings come from an env file (default ~/.config/tour-worker/worker.env; ENV_FILE=... for
# another). tools/runpod/worker.env.example lists them. Keep the real file OUT of the repo: it
# holds WORKER_API_KEY and the AWS keys.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$HOME/.config/tour-worker/worker.env}"
VENV="${VENV:-$REPO/.venv-pod}"
RUN_DIR="${RUN_DIR:-$HOME/.local/state/tour-worker}"
PID_FILE="$RUN_DIR/service.pid"
LOG_FILE="$RUN_DIR/service.log"
mkdir -p "$RUN_DIR"

load_env() {
    [ -f "$ENV_FILE" ] || { echo "no env file at $ENV_FILE (copy tools/runpod/worker.env.example)"; exit 1; }
    set -a; . "$ENV_FILE"; set +a
    : "${WORKER_API_KEY:?WORKER_API_KEY must be set in $ENV_FILE}"
    export HOST="${HOST:-0.0.0.0}" PORT="${PORT:-8000}"
    export JOBS_DIR="${JOBS_DIR:-$RUN_DIR/jobs}" SPLAT_BACKEND="${SPLAT_BACKEND:-gsplat}"
    mkdir -p "$JOBS_DIR"
}
running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
health() { curl -fsS "http://127.0.0.1:${PORT:-8000}/health" && echo; }

case "${1:-start}" in
start)
    if running; then echo "already running (pid $(cat "$PID_FILE"))"; exit 0; fi
    load_env
    cd "$REPO"
    nohup "$VENV/bin/python" -u -m tour_worker.service >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "started pid $(cat "$PID_FILE"), log $LOG_FILE"
    for _ in $(seq 60); do
        if curl -fsS "http://127.0.0.1:$PORT/ping" >/dev/null 2>&1; then health; exit 0; fi
        running || { echo "the service exited:"; tail -20 "$LOG_FILE"; exit 1; }
        sleep 1
    done
    echo "no answer on :$PORT after 60 s"; tail -20 "$LOG_FILE"; exit 1 ;;
stop)
    if running; then kill "$(cat "$PID_FILE")"; rm -f "$PID_FILE"; echo "stopped"; else echo "not running"; fi ;;
status)
    if running; then echo "running (pid $(cat "$PID_FILE"))"; PORT="$(grep -E '^PORT=' "$ENV_FILE" 2>/dev/null | cut -d= -f2)"; health || true
    else echo "not running"; exit 1; fi ;;
logs)
    tail -f "$LOG_FILE" ;;
*)
    echo "usage: $0 [start|stop|status|logs]"; exit 2 ;;
esac
