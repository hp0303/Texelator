# Native Windows validation

Texelator 0.2.0 has been built and exercised directly in Windows 11 PowerShell on an
NVIDIA GeForce RTX 5080 (`sm_120`) with CUDA Toolkit 12.8. The validation covered:

- automatic MSVC x64 environment discovery;
- isolated Python environment creation with Python 3.10;
- native CUDA extension compilation and import;
- hardware-exact BC4 palette measurement;
- the command wrapper that runs without virtual-environment activation;
- local model loading and terminal generation.

The measured RTX 5080 palette contained 65,025 endpoint pairs and produced SHA256
`34670bd1c521c26b14ea88ed6f2d8f16c55720f20915909ac621dcabe0f3ed47`.

RTX 40-series hardware is validated under Linux/WSL. Native Windows RTX 40-series
validation remains desirable; the installer compiles for the visible device and fails
immediately if the CUDA extension or palette probe does not succeed.

Recommended smoke test:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\texelator.cmd doctor --force
.\texelator.cmd pull qwen3.8:27b
.\texelator.cmd benchmark qwen3.8:27b
.\texelator.cmd run qwen3.8:27b "Reply with one short sentence."
```
