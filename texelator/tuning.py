from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from .artifacts import sha256_file, write_json
from .runtime import extension, free, pack_entry


CANDIDATES = (1, 2, 3, 4, 6, 8)


def _profile_path(artifact: Path) -> Path:
    major, minor = torch.cuda.get_device_capability()
    return artifact / "profiles" / f"sm_{major}{minor}.json"


def tune_artifact(
    artifact: Path,
    warmup: int = 10,
    measured: int = 50,
    runs: int = 3,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("tuning requires a visible CUDA GPU")
    manifest = json.loads((artifact / "texelator.json").read_text())
    weights = artifact / manifest["weights"]
    metadata_path = weights / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    entries = metadata["entries"]
    if not entries:
        raise RuntimeError("artifact contains no encoded linears")
    ext = extension()
    handles: list[int] = []
    inputs: list[torch.Tensor] = []
    try:
        generator = torch.Generator(device="cuda").manual_seed(0)
        for entry in entries:
            handles.append(pack_entry(weights, entry, lookahead=1))
            inputs.append(torch.randn((1, int(entry["K"])), device="cuda", dtype=torch.float16,
                                      generator=generator))

        reference = [ext.linear(handle, value).float() for handle, value in list(zip(handles, inputs))[:8]]
        rows = []
        for candidate in CANDIDATES:
            for handle in handles:
                ext.set_lookahead(handle, candidate)
            candidate_values = [
                ext.linear(handle, value).float()
                for handle, value in list(zip(handles, inputs))[:8]
            ]
            max_abs = max(
                float((got - expected).abs().max().item())
                for got, expected in zip(candidate_values, reference)
            )
            if max_abs > 1e-3:
                raise RuntimeError(f"lookahead K={candidate} failed correctness: max_abs={max_abs}")
            for _ in range(warmup):
                for handle, value in zip(handles, inputs):
                    ext.linear(handle, value)
            torch.cuda.synchronize()
            timings = []
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(measured):
                    for handle, value in zip(handles, inputs):
                        ext.linear(handle, value)
                stop.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(stop) / measured)
            row = {
                "lookahead": candidate,
                "max_abs_vs_k1": max_abs,
                "sweep_ms_median": statistics.median(timings),
                "sweep_ms_min": min(timings),
                "sweep_ms_max": max(timings),
                "runs": timings,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        selected = min(rows, key=lambda row: row["sweep_ms_median"])["lookahead"]
        major, minor = torch.cuda.get_device_capability()
        profile = {
            "schema_version": 1,
            "device": torch.cuda.get_device_name(),
            "compute_capability": f"{major}.{minor}",
            "weights_metadata_sha256": sha256_file(metadata_path),
            "linears": len(entries),
            "warmup_sweeps": warmup,
            "measured_sweeps": measured,
            "independent_runs": runs,
            "selected_lookahead": selected,
            "candidates": rows,
        }
        target = _profile_path(artifact)
        write_json(target, profile)
        print(f"[texelator] selected K={selected}; profile saved to {target}")
        return profile
    finally:
        free(handles)


def selected_lookahead(artifact: Path) -> tuple[int, Path | None]:
    target = _profile_path(artifact)
    if not target.exists():
        return 1, None
    profile = json.loads(target.read_text())
    expected = sha256_file(artifact / "weights" / "metadata.json")
    if profile.get("weights_metadata_sha256") != expected:
        raise RuntimeError("tuning profile does not match the encoded weights; rerun texelator tune")
    return int(profile["selected_lookahead"]), target

