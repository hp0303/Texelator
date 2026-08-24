# Texelator quick start

## Native Windows 11

Install Python 3.10--3.14 x64, Visual Studio 2022 **Desktop development with C++**,
CUDA Toolkit 12.8, and a current NVIDIA driver. Download and extract the Windows ZIP
from the latest GitHub Release, then open PowerShell in `texelator_windows`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\texelator.cmd pull qwen3.8:27b
.\texelator.cmd benchmark qwen3.8:27b
.\texelator.cmd run qwen3.8:27b
```

The wrapper uses the isolated virtual environment without requiring activation.

## Linux or WSL2

Install a CUDA toolkit containing `nvcc`, Python development headers, a virtual
environment package, and build tools. Then:

```bash
bash scripts/install.sh
source .venv/bin/activate
texelator pull qwen3.8:27b
texelator benchmark qwen3.8:27b
texelator run qwen3.8:27b
```

The benchmark is mandatory once per model/GPU pair. Omit the prompt on `run` for
interactive chat. Use `/clear`, `/thinking on`, `/thinking off`, and `/bye` in chat.

## Convert a model yourself

```bash
texelator model install Qwen/Qwen2.5-3B --name qwen-3b
texelator ptq qwen-3b --name qwen-3b-awbc4 --output /models/qwen-3b-awbc4
texelator benchmark qwen-3b-awbc4
texelator run qwen-3b-awbc4
```

To reuse a local Hugging Face checkpoint without copying it:

```bash
texelator model register /models/Qwen2.5-3B --name qwen-3b
```

PTQ requires source-precision FP16, BF16, or FP32 weights. It cannot convert an
already packed GPTQ, AWQ, or GGUF artifact.
