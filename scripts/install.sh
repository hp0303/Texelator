#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required. Install a CUDA toolkit before running Texelator." >&2
  exit 1
fi

PYTHON_INCLUDE="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [[ ! -f "$PYTHON_INCLUDE/Python.h" ]]; then
  echo "Python development headers are required (for Ubuntu: apt install python3-dev)." >&2
  exit 1
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT"
"$ROOT/.venv/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch with a visible NVIDIA GPU is required")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
PY
echo "Activate with: source $ROOT/.venv/bin/activate"
