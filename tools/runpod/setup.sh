#!/usr/bin/env bash
# Install the tour worker on a RunPod GPU pod, ON TOP OF the pod's pinned GPU stack.
#
# The pod already has its pinned torch / CUDA / gsplat (and maybe transformers). This script never
# replaces them: it makes a virtualenv that sees the pod's packages (--system-site-packages),
# freezes what the pod has into a constraints file, and installs only what the worker adds
# (boto3, opencv-headless, pydantic, runpod, fastapi, uvicorn) under those constraints. If the
# worker would need a different version of something the pod pins, pip stops and says which -
# nothing is changed.
#
#   tools/runpod/setup.sh                  # render + plan jobs (3D included)
#   WITH_SEGMENT=1 tools/runpod/setup.sh   # also transformers / huggingface_hub / scipy for
#                                          # tour_segment (only if the pod lacks them)
#
# Settings: PYTHON (default python3: the interpreter that has the pod's torch), VENV (default
# <repo>/.venv-pod). Run it again after a git pull: it only adds what is missing.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$REPO/.venv-pod}"
say() { printf '\n== %s\n' "$*"; }

say "system tools: ffmpeg / ffprobe"
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends ffmpeg
fi
ffmpeg -hide_banner -version | head -1

say "the pod's GPU stack (left as it is)"
"$PYTHON" - <<'EOF'
import importlib.metadata as md
for name in ("torch", "torchvision", "gsplat", "transformers", "numpy"):
    try:
        print(f"  {name:13s} {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"  {name:13s} -")
try:
    import torch
    print(f"  cuda          {torch.version.cuda}  available={torch.cuda.is_available()}")
except Exception as e:                       # noqa: BLE001
    print(f"  torch import failed: {e}")
EOF

say "virtualenv $VENV (sees the pod's packages)"
if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv --system-site-packages "$VENV"
fi
CONSTRAINTS="$VENV/pod-constraints.txt"
if [ ! -f "$CONSTRAINTS" ]; then
    # what the pod had before the worker came: plain name==version lines only (pip refuses
    # constraints with URLs / editables). Kept from the first run, so later runs pin the same.
    "$VENV/bin/python" -m pip freeze --all 2>/dev/null | grep -E '^[A-Za-z0-9_.-]+==' > "$CONSTRAINTS"
fi
echo "  constraints: $(wc -l < "$CONSTRAINTS") pinned packages ($CONSTRAINTS)"

say "the worker (+ http extra) under the pod's constraints"
"$VENV/bin/python" -m pip install -q --upgrade-strategy only-if-needed -c "$CONSTRAINTS" -e "$REPO[http]"
if [ "${WITH_SEGMENT:-0}" = "1" ]; then
    say "tour_segment: transformers / huggingface_hub / scipy (only if missing)"
    "$VENV/bin/python" -m pip install -q --upgrade-strategy only-if-needed -c "$CONSTRAINTS" \
        "transformers>=4.46" "huggingface_hub>=0.25" "scipy>=1.10"
fi

say "check: the pinned GPU packages are unchanged"
"$VENV/bin/python" - "$CONSTRAINTS" <<'EOF'
import importlib.metadata as md, sys
pins = dict(l.strip().split("==", 1) for l in open(sys.argv[1]) if "==" in l)
bad = []
for name in ("torch", "torchvision", "gsplat", "numpy"):
    key = next((k for k in pins if k.lower().replace("_", "-") == name), None)
    if key is None:
        continue
    try:
        now = md.version(name)
    except md.PackageNotFoundError:
        now = None
    if now != pins[key]:
        bad.append(f"{name}: pod {pins[key]} -> now {now}")
print("\n".join("  CHANGED " + b for b in bad) or "  torch / torchvision / gsplat / numpy as the pod pinned them")
sys.exit(1 if bad else 0)
EOF

say "done"
echo "  next: $VENV/bin/python -m tools.runpod.preflight   (from $REPO)"
