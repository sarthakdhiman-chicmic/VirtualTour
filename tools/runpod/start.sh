#!/usr/bin/env bash
# Start / stop / check the tour worker's HTTP service on a RunPod GPU pod (no container image).
#
#   tools/runpod/start.sh            start in the background, wait until it answers, print /health,
#                                    then follow the log (Ctrl+C stops following; the worker keeps running)
#   tools/runpod/start.sh start -d   the same without following the log
#   tools/runpod/start.sh stop       stop it
#   tools/runpod/start.sh status     is it running; /health
#   tools/runpod/start.sh logs       the last job's lines, then follow the log
#
# Each job is one block in the log: its stages (▶ / ✓ and seconds), a progress bar with an ETA while
# it renders, and how long each stage took (tour_worker/progress.py).
#
# Settings come from <repo>/worker.env (gitignored; ENV_FILE=... for another), loaded by
# tools/runpod/env.sh with the pod defaults. tools/runpod/worker.env.example lists them. The pid,
# log and job records go to <repo>/.pod/run.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$REPO/.venv-pod}"
RUN_DIR="${RUN_DIR:-${POD_DIR:-$REPO/.pod}/run}"
PID_FILE="$RUN_DIR/service.pid"
LOG_FILE="$RUN_DIR/service.log"
mkdir -p "$RUN_DIR"

load_env() {
    . "$REPO/tools/runpod/env.sh" || exit 1
    : "${WORKER_API_KEY:?WORKER_API_KEY must be set in $ENV_FILE}"
}
running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
health() { curl -fsS "http://127.0.0.1:${PORT:-8000}/health" && echo; }
follow() {
    # from the start of the last job's block (or the last 20 lines), then new lines as they come
    local from
    from=$(grep -n "┌─ job" "$LOG_FILE" 2>/dev/null | tail -1 | cut -d: -f1)
    echo "--- following $LOG_FILE (Ctrl+C stops following; the worker keeps running) ---"
    if [ -n "$from" ]; then tail -n "+$from" -F "$LOG_FILE"; else tail -n 20 -F "$LOG_FILE"; fi
}

case "${1:-start}" in
start)
    if running; then echo "already running (pid $(cat "$PID_FILE"))"; exit 0; fi
    load_env
    cd "$REPO"
    # its own session: a Ctrl+C on the log we follow below never reaches the worker
    setsid nohup "$VENV/bin/python" -u -m tour_worker.service >> "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    echo "started pid $(cat "$PID_FILE"), log $LOG_FILE"
    for _ in $(seq 60); do
        if curl -fsS "http://127.0.0.1:$PORT/ping" >/dev/null 2>&1; then
            health
            [ "${2:-}" = "-d" ] && exit 0
            follow; exit 0
        fi
        running || { echo "the service exited:"; tail -20 "$LOG_FILE"; exit 1; }
        sleep 1
    done
    echo "no answer on :$PORT after 60 s"; tail -20 "$LOG_FILE"; exit 1 ;;
stop)
    if running; then kill "$(cat "$PID_FILE")"; rm -f "$PID_FILE"; echo "stopped"; else echo "not running"; fi ;;
status)
    if running; then echo "running (pid $(cat "$PID_FILE"))"; load_env >/dev/null; health || true
    else echo "not running"; exit 1; fi ;;
logs)
    follow ;;
*)
    echo "usage: $0 [start [-d]|stop|status|logs]"; exit 2 ;;
esac
