from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from .artifacts import sha256_file, write_json
from .hardware import ensure_hardware
from .runtime import extension, free, pack_entry_handles
from .store import STATE_HOME, safe_name

CANDIDATES = (1, 2, 3, 4, 6, 8)


def _profile_path(artifact: Path) -> Path:
    major, minor = torch.cuda.get_device_capability()
    metadata_hash = sha256_file(artifact / "weights" / "metadata.json")
    device = safe_name(torch.cuda.get_device_name())
    return STATE_HOME / "profiles" / metadata_hash[:20] / f"sm_{major}{minor}-{device}.json"


def tune_artifact(
    artifact: Path,
    warmup: int = 10,
    measured: int = 50,
    runs: int = 3,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("tuning requires a visible CUDA GPU")
    manifest = json.loads((artifact / "texelator.json").read_text())
    runtime_dtype_name = manifest.get("runtime_dtype", "float16")
    runtime_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if runtime_dtype_name not in runtime_dtypes:
        raise RuntimeError(f"unsupported Texelator runtime dtype: {runtime_dtype_name!r}")
    runtime_dtype = runtime_dtypes[runtime_dtype_name]
    current_palette = ensure_hardware()
    expected_palette = manifest.get("hardware", {}).get("palette_sha256")
    if not expected_palette or sha256_file(current_palette) != expected_palette:
        raise RuntimeError("model BC4 palette does not match this GPU; benchmark aborted")
    weights = artifact / manifest["weights"]
    metadata_path = weights / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    entries = metadata["entries"]
    if not entries:
        raise RuntimeError("artifact contains no encoded linears")
    ext = extension()
    handles: list[int] = []
    handle_groups: list[list[int]] = []
    inputs: list[torch.Tensor] = []
    try:
        generator = torch.Generator(device="cuda").manual_seed(0)
        for entry in entries:
            group = pack_entry_handles(weights, entry, lookahead=1)
            handle_groups.append(group)
            handles.extend(group)
            inputs.append(torch.randn((1, int(entry["K"])), device="cuda", dtype=runtime_dtype,
                                      generator=generator))

        def evaluate(group: list[int], value: torch.Tensor) -> torch.Tensor:
            parts = [ext.linear(handle, value) for handle in group]
            return (parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)).float()

        paired = list(zip(handle_groups, inputs))
        sample = paired if len(paired) <= 8 else paired[:4] + paired[-4:]
        reference = [evaluate(group, value) for group, value in sample]
        rows = []
        for candidate in CANDIDATES:
            for handle in handles:
                ext.set_lookahead(handle, candidate)
            candidate_values = [
                evaluate(group, value)
                for group, value in sample
            ]
            max_abs = max(
                float((got - expected).abs().max().item())
                for got, expected in zip(candidate_values, reference)
            )
            if max_abs > 1e-3:
                raise RuntimeError(f"lookahead K={candidate} failed correctness: max_abs={max_abs}")
            for _ in range(warmup):
                for group, value in zip(handle_groups, inputs):
                    for handle in group:
                        ext.linear(handle, value)
            torch.cuda.synchronize()
            timings = []
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(measured):
                    for group, value in zip(handle_groups, inputs):
                        for handle in group:
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
            "palette_sha256": expected_palette,
            "artifact": str(artifact.resolve()),
            "runtime_dtype": runtime_dtype_name,
            "linears": len(entries),
            "warmup_sweeps": warmup,
            "measured_sweeps": measured,
            "independent_runs": runs,
            "selected_lookahead": selected,
            "candidates": rows,
        }
        target = _profile_path(artifact)
        write_json(target, profile)
        print(f"[texelator] selected K={selected}; local benchmark profile saved to {target}")
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
        raise RuntimeError("benchmark profile does not match the encoded weights; rerun texelator benchmark")
    manifest = json.loads((artifact / "texelator.json").read_text())
    if profile.get("palette_sha256") != manifest.get("hardware", {}).get("palette_sha256"):
        raise RuntimeError("benchmark profile does not match the model palette; rerun texelator benchmark")
    return int(profile["selected_lookahead"]), target
