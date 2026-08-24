[CmdletBinding()]
param(
    [string]$TorchVersion = "2.11.0",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Texelator requires 64-bit Windows."
}

function Assert-NativeSuccess([string]$Stage) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Stage failed with exit code $LASTEXITCODE. Fix this stage and rerun setup.ps1."
    }
}

function Import-MsvcEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Visual Studio 2022 Build Tools were not found. Install 'Desktop development with C++'."
    }
    $installation = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $installation) {
        throw "MSVC x64 tools were not found. Add 'Desktop development with C++' in Visual Studio Installer."
    }
    $devCmd = Join-Path $installation "Common7\Tools\VsDevCmd.bat"
    $command = "`"$devCmd`" -no_logo -arch=x64 -host_arch=x64 && set"
    & cmd.exe /s /c $command | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "MSVC environment initialization failed: cl.exe is still unavailable."
    }
}

function Find-CudaToolkit {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        return $env:CUDA_PATH
    }
    $root = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $root) {
        $candidate = Get-ChildItem $root -Directory | Sort-Object Name -Descending | Where-Object {
            Test-Path (Join-Path $_.FullName "bin\nvcc.exe")
        } | Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    throw "CUDA Toolkit with nvcc.exe was not found. Install CUDA Toolkit 12.8 or set CUDA_PATH."
}

function Find-Python {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        # Probe without allowing a missing interpreter version to become a terminating
        # NativeCommandError under the script-wide ErrorActionPreference=Stop.
        foreach ($version in @("3.12", "3.11", "3.10", "3.13", "3.14")) {
            $savedPreference = $ErrorActionPreference
            $ErrorActionPreference = "SilentlyContinue"
            & py.exe "-$version" -c "import sys; assert sys.maxsize > 2**32" *> $null
            $available = $LASTEXITCODE -eq 0
            $ErrorActionPreference = $savedPreference
            if ($available) { return @("py.exe", "-$version") }
        }
    }
    if (Get-Command python.exe -ErrorAction SilentlyContinue) {
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & python.exe -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,14) and sys.maxsize > 2**32" *> $null
        $available = $LASTEXITCODE -eq 0
        $ErrorActionPreference = $savedPreference
        if ($available) { return @("python.exe") }
    }
    throw "64-bit Python 3.10-3.14 was not found. Install Python from python.org; Python 3.12 x64 is recommended."
}

Write-Host "[1/5] MSVC x64 build environment"
Import-MsvcEnvironment
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"
Write-Host "MSVC toolset: $env:VCToolsVersion"

Write-Host "[2/5] CUDA Toolkit"
$cudaRoot = Find-CudaToolkit
$env:CUDA_PATH = $cudaRoot
$env:CUDA_HOME = $cudaRoot
$env:Path = "$(Join-Path $cudaRoot 'bin');$env:Path"
& (Join-Path $cudaRoot "bin\nvcc.exe") --version
Assert-NativeSuccess "CUDA compiler check"

Write-Host "[3/5] Isolated Python environment"
$pythonCommand = Find-Python
Write-Host "Using Python launcher: $($pythonCommand -join ' ')"
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    if ($pythonCommand.Count -eq 2) {
        & $pythonCommand[0] $pythonCommand[1] -m venv (Join-Path $Root ".venv")
    } else {
        & $pythonCommand[0] -m venv (Join-Path $Root ".venv")
    }
    Assert-NativeSuccess "virtual environment creation"
}
& $venvPython -m pip install --upgrade pip setuptools wheel ninja
Assert-NativeSuccess "Python build-tool installation"

Write-Host "[4/5] Controlled CUDA PyTorch and Texelator native extension"
$savedPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& $venvPython -c @"
import torch
ok = (
    torch.__version__.split('+')[0] == '$TorchVersion'
    and torch.version.cuda
    and torch.version.cuda.startswith('12.8')
    and torch.cuda.is_available()
)
raise SystemExit(0 if ok else 1)
"@ 2>$null
$torchReady = $LASTEXITCODE -eq 0
$ErrorActionPreference = $savedPreference
if (-not $torchReady) {
    & $venvPython -m pip install --upgrade --force-reinstall "torch==$TorchVersion" --index-url $TorchIndex
    Assert-NativeSuccess "CUDA PyTorch installation"
}
& $venvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is not visible to PyTorch'"
Assert-NativeSuccess "PyTorch CUDA check"
$arch = & $venvPython -c "import torch; a=torch.cuda.get_device_capability(); print(f'{a[0]}.{a[1]}')"
Assert-NativeSuccess "GPU architecture detection"
$env:TORCH_CUDA_ARCH_LIST = $arch.Trim()
$env:TEXELATOR_BUILD_CUDA = "1"
$env:MAX_JOBS = [Math]::Max(1, [Math]::Min(8, [Environment]::ProcessorCount)).ToString()
$buildLog = Join-Path $Root "texelator_build.log"
Write-Host "Native build log: $buildLog"
$savedPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $venvPython -m pip install --editable $Root --no-build-isolation --verbose 2>&1 |
    Tee-Object -FilePath $buildLog
$buildExit = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($buildExit -ne 0) {
    throw "Texelator native extension build failed with exit code $buildExit. See $buildLog"
}
& $venvPython -c "import torch; import texelator._cuda; print('Native CUDA extension: READY')"
Assert-NativeSuccess "Texelator native extension import"

Write-Host "[5/5] Hardware-exact BC4 probe"
$texelator = Join-Path $Root ".venv\Scripts\texelator.exe"
& $texelator doctor
Assert-NativeSuccess "Texelator hardware probe"

Write-Host ""
Write-Host "Setup complete. Run without activation using the bundled wrapper:"
Write-Host "  .\texelator.cmd pull qwen3.8:27b"
Write-Host "  .\texelator.cmd benchmark qwen3.8:27b"
Write-Host "  .\texelator.cmd run qwen3.8:27b"
Write-Host "A clean venv can also be activated with .\.venv\Scripts\Activate.ps1."
