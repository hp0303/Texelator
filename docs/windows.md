# Native Windows installation

Texelator runs directly in Windows 11 PowerShell; WSL is optional.

## Requirements

- NVIDIA RTX 40- or RTX 50-series GPU with a current display driver
- Python 3.10--3.14 x64 from python.org (`3.12` recommended)
- Visual Studio 2022 Build Tools with **Desktop development with C++**
- CUDA Toolkit 12.8 with `nvcc.exe`

Download `texelator-v0.2.0-windows.zip` from the latest GitHub Release and
extract it. In the extracted `texelator_windows` directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

The installer locates MSVC and CUDA, creates `.venv`, installs the controlled PyTorch
build, compiles the native extension for the visible GPU, and measures the hardware
BC4 palette. Every stage is fail-fast.

Run commands through the bundled wrapper; virtual-environment activation is optional:

```powershell
.\texelator.cmd pull qwen3.8:27b
.\texelator.cmd benchmark qwen3.8:27b
.\texelator.cmd run qwen3.8:27b
```

State is stored under `%LOCALAPPDATA%\Texelator`. Set `TEXELATOR_HOME` and `HF_HOME`
before setup or download to use another drive.

For PTQ of a source-precision model:

```powershell
.\texelator.cmd model install Qwen/Qwen2.5-3B --name qwen-3b
.\texelator.cmd ptq qwen-3b --name qwen-3b-awbc4 --output D:\models\qwen-3b-awbc4
.\texelator.cmd benchmark qwen-3b-awbc4
.\texelator.cmd run qwen-3b-awbc4
```
