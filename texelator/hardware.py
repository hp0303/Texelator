from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from .artifacts import environment_snapshot, sha256_file, write_json
from .store import STATE_HOME


def ensure_hardware(force: bool = False) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Texelator requires CUDA-enabled PyTorch and a visible NVIDIA GPU")
    major, minor = torch.cuda.get_device_capability()
    destination = STATE_HOME / "hardware" / f"sm_{major}{minor}"
    palette = destination / "palette.bin"
    if palette.exists() and not force:
        return palette
    build = STATE_HOME / "build" / f"sm_{major}{minor}"
    build.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    # A release wheel contains the probe in the native extension, so end users do
    # not need nvcc. Source installations retain the standalone nvcc fallback.
    try:
        from .runtime import extension

        print(f"[texelator] measuring BC4 reconstruction on {torch.cuda.get_device_name()}...", flush=True)
        values = extension().palette_probe().contiguous()
        values.numpy().astype("<f4", copy=False).tofile(palette)
        write_json(destination / "palette.json", {
            "device": torch.cuda.get_device_name(),
            "compute_capability": f"{major}.{minor}",
            "endpoint_pairs": 255 * 255,
            "values_per_pair": 8,
            "bytes": int(values.numel() * values.element_size()),
        })
        write_json(destination / "environment.json", environment_snapshot())
        return palette
    except Exception as native_error:
        if sys.platform == "win32":
            raise RuntimeError(
                "the native Windows CUDA extension is unavailable; rerun setup.ps1"
            ) from native_error
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError(
                "the prebuilt Texelator CUDA extension is unavailable and nvcc was not found"
            ) from native_error
    executable = build / ("dump_hw_palette.exe" if sys.platform == "win32" else "dump_hw_palette")
    source = Path(__file__).resolve().parent / "cuda" / "dump_hw_palette.cu"
    subprocess.run([nvcc, "-O3", "-std=c++17", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(destination)], check=True)
    if not palette.exists():
        raise RuntimeError("hardware probe did not produce palette.bin")
    write_json(destination / "environment.json", environment_snapshot())
    return palette


def doctor_payload(force: bool = False) -> dict:
    palette = ensure_hardware(force=force)
    metadata = json.loads(palette.with_name("palette.json").read_text())
    return {
        "status": "ready",
        "state_home": str(STATE_HOME),
        "palette": str(palette),
        "palette_sha256": sha256_file(palette),
        **metadata,
    }
