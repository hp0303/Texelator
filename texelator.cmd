@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%.venv\Scripts\texelator.exe" (
  echo Texelator is not installed. Run PowerShell -ExecutionPolicy Bypass -File setup.ps1 first. 1>&2
  exit /b 1
)
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"
"%ROOT%.venv\Scripts\texelator.exe" %*
