from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, check=False, text=True, capture_output=True).stdout.strip() or None
    except OSError:
        return None


def environment_snapshot() -> dict:
    gpu = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "gpu": gpu,
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"]),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "command": sys.argv,
        "cwd": os.getcwd(),
    }


def validate_encoded(encoded: Path) -> dict:
    metadata_path = encoded / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    failures = []
    for entry in metadata["entries"]:
        for key in ("blocks", "scales"):
            path = encoded / entry[f"{key}_file"]
            expected = entry[f"{key}_sha256"]
            actual = sha256_file(path)
            if actual != expected:
                failures.append({"module_name": entry["module_name"], "file": str(path), "expected": expected, "actual": actual})
    return {"ok": not failures, "entries": len(metadata["entries"]), "failures": failures}

