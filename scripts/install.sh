#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CUDA_ROOT="${CUDA_HOME:-${CUDA_PATH:-}}"
if ! command -v nvcc >/dev/null 2>&1; then
  for candidate in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.9 /usr/local/cuda-12.8; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      CUDA_ROOT="$candidate"
      export CUDA_HOME="$candidate"
      export PATH="$candidate/bin:$PATH"
      export LD_LIBRARY_PATH="$candidate/lib64:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required. Install a CUDA toolkit or set CUDA_HOME before running Texelator." >&2
  exit 1
fi
if [[ -z "$CUDA_ROOT" ]]; then
  CUDA_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

PYTHON_INCLUDE="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [[ ! -f "$PYTHON_INCLUDE/Python.h" ]]; then
  echo "Python development headers are required (for Ubuntu: apt install python3-dev)." >&2
  exit 1
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
TORCH_VERSION="${TEXELATOR_TORCH_VERSION:-2.11.0}"
TORCH_INDEX="${TEXELATOR_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
if ! "$ROOT/.venv/bin/python" - "$TORCH_VERSION" <<'PY'
import sys
try:
    import torch
except ImportError:
    raise SystemExit(1)
expected = sys.argv[1]
valid = (
    torch.__version__.split("+")[0] == expected
    and torch.version.cuda is not None
    and torch.version.cuda.startswith("12.8")
    and torch.cuda.is_available()
)
raise SystemExit(0 if valid else 1)
PY
then
  echo "Installing controlled CUDA 12.8 PyTorch ${TORCH_VERSION}..."
  "$ROOT/.venv/bin/python" -m pip install --upgrade --force-reinstall \
    "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
fi
"$ROOT/.venv/bin/python" -m pip install --upgrade 'jinja2>=3.1.0,<4'
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT"
if ! grep -q '# Texelator CUDA environment' "$ROOT/.venv/bin/activate"; then
  {
    echo
    echo '# Texelator CUDA environment'
    printf 'export CUDA_HOME=%q\n' "$CUDA_ROOT"
    echo 'export PATH="$CUDA_HOME/bin:$PATH"'
    echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"'
  } >> "$ROOT/.venv/bin/activate"
fi
"$ROOT/.venv/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch with a visible NVIDIA GPU is required")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
PY
echo "Activate with: source $ROOT/.venv/bin/activate"
