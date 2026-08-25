param(
    [string]$Artifact = "qwen3.8:27b",
    [int]$Tokens = 512,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $Output) {
    $Output = Join-Path $Root "results\prefill-$Tokens.json"
}
$Directory = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $Directory | Out-Null

$Command = Join-Path $Root "texelator.cmd"
if (-not (Test-Path $Command)) {
    throw "texelator.cmd was not found at $Command"
}

& $Command prefill-benchmark $Artifact `
    --tokens $Tokens --warmup 1 --runs 3 --output $Output
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid prefill validation failed with exit code $LASTEXITCODE"
}
Write-Host "Hybrid prefill validation saved to $Output"
