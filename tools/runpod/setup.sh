#!/usr/bin/env bash
# Install the tour worker on a RunPod GPU pod, ON TOP OF the pod's pinned GPU stack.
#
# The pod already has its pinned torch / CUDA / gsplat (and maybe transformers). This script never
# replaces them: it makes a virtualenv that sees the packages of the environment PYTHON belongs
# to (system, conda or another venv - linked with a .pth file, read only), pins that
# environment's GPU stack (torch, CUDA libraries, gsplat, numpy, transformers) with a constraints
# file, and installs what the worker adds into the venv only. A package the worker needs at
# another version than the environment has (e.g. OpenCV 5 vs 4) is installed in the venv and
# shadows the other copy inside the worker only. If the worker would need a different version of
# the GPU stack, pip stops and says which - nothing is changed.
#
#   tools/runpod/setup.sh                  # render + plan jobs (3D included)
#   WITH_GSPLAT=1 tools/runpod/setup.sh    # a fresh pod: also gsplat (GSPLAT_VERSION, default
#                                          # 1.5.3, Apache-2.0) into the venv, if the pod has none
#   WITH_SEGMENT=1 tools/runpod/setup.sh   # also transformers / huggingface_hub / scipy for
#                                          # tour_segment (only if the pod lacks them)
# Everything goes into the venv: the pod's own Python environment is only read, never changed.
#
# Settings: PYTHON (default python3): the interpreter of the environment that has torch / gsplat,
# e.g. PYTHON=/workspace/<their-env>/bin/python. VENV (default <repo>/.venv-pod). Run it again
# after a git pull: it only adds what is missing. pip's cache goes next to the venv
# (PIP_CACHE_DIR), so the pod's small container disk is not filled.
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

say "virtualenv $VENV (reads the packages of $PYTHON's environment)"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "$VENV")/.pip-cache}"
if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv --system-site-packages "$VENV"
fi
# --system-site-packages only reaches the BASE interpreter's packages; when PYTHON is itself in a
# venv / conda env, link that env's site-packages too (appended after the venv's own, read only)
VENV_SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$PYTHON" -c 'import site, sys; print("\n".join(p for p in site.getsitepackages() if "site-packages" in p or "dist-packages" in p))' \
    > "$VENV_SITE/zz_pod_env.pth"
echo "  linked: $(tr '\n' ' ' < "$VENV_SITE/zz_pod_env.pth")"
# Only the GPU stack must be the environment's own: one torch / CUDA in the process, and the
# numpy it was built against. Anything else the worker needs at another version (e.g. OpenCV 5
# where the environment has 4) is installed in this venv, shadowing the other env's copy here only.
SHARED='^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+|gsplat|numpy|transformers|tokenizers|safetensors|huggingface[-_]hub|accelerate|xformers)=='
CONSTRAINTS="$VENV/pod-constraints.txt"
if [ ! -f "$CONSTRAINTS" ]; then
    # what that environment had before the worker came (read with importlib.metadata: no pip
    # needed there - uv venvs have none). Kept from the first run, so later runs pin the same.
    "$PYTHON" -c 'import importlib.metadata as m
seen = {}
for d in m.distributions():
    n = d.metadata["Name"]
    if n and n.lower() not in seen:
        seen[n.lower()] = f"{n}=={d.version}"
print("\n".join(sorted(seen.values())))' \
        | grep -i -E "$SHARED" > "$CONSTRAINTS" || true
fi
echo "  pinned to the environment's versions: $(tr '\n' ' ' < "$CONSTRAINTS" | cut -c1-400)"

say "the worker (+ http extra) under the pod's constraints"
"$VENV/bin/python" -m pip install -q --upgrade-strategy only-if-needed -c "$CONSTRAINTS" -e "$REPO[http]"
# gsplat compiles its CUDA kernels on first use with ninja (a small build tool; env.sh puts the
# venv's bin/ on PATH so torch finds it)
"$VENV/bin/python" -m pip install -q -c "$CONSTRAINTS" ninja
command -v nvcc >/dev/null || [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] \
    || echo "  WARNING: no nvcc found - gsplat cannot build its kernels (needs a CUDA toolkit / -devel image)"
if [ "${WITH_GSPLAT:-0}" = "1" ]; then
    say "gsplat ${GSPLAT_VERSION:=1.5.3} into the venv (only if the pod has none)"
    if "$VENV/bin/python" -c "import importlib.metadata as m; m.version('gsplat')" 2>/dev/null; then
        echo "  the pod has gsplat $("$VENV/bin/python" -c "import importlib.metadata as m; print(m.version('gsplat'))"): kept"
    else
        # PyPI's gsplat compiles its CUDA kernels on first use: that needs nvcc (a -devel image)
        command -v nvcc >/dev/null || echo "  WARNING: no nvcc on PATH - gsplat cannot build its kernels (use a CUDA -devel image)"
        "$VENV/bin/python" -m pip install -q -c "$CONSTRAINTS" "gsplat==$GSPLAT_VERSION"
    fi
fi
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
