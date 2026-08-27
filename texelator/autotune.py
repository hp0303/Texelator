from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch

from .runtime import extension, free, pack_entry_handles
from .store import STATE_HOME, safe_name


CANDIDATES = {
    "large_256x128x32": 5,
    "medium_128x128x32": 6,
    "medium_128x64x32": 7,
    "medium_64x128x32": 8,
    "small_32x32x64": 9,
}
TOKEN_BUCKETS = (32, 128, 512, 2048, 8192, 16384, 32768, 65535)


def profile_path(artifact: Path) -> Path:
    metadata_hash = _sha256(artifact / "weights" / "metadata.json")
    major, minor = torch.cuda.get_device_capability()
    device = safe_name(torch.cuda.get_device_name())
    return (STATE_HOME / "profiles" / metadata_hash[:20] /
            f"sm_{major}{minor}-{device}-prefill.json")


def token_bucket(tokens: int) -> int:
    if tokens < 1:
        raise ValueError("tokens must be positive")
    if tokens > TOKEN_BUCKETS[-1]:
        raise ValueError(f"tokens exceeds supported CUDA grid limit: {tokens}")
    return next(value for value in TOKEN_BUCKETS if tokens <= value)


def dispatch_key(tokens: int, m: int, k: int) -> str:
    return f"tokens<={token_bucket(tokens)}:M={m}:K={k}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_source_sha256() -> str:
    root = Path(__file__).resolve().parent / "cuda"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.cu")) + sorted(root.glob("*.cpp")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _error(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref, value = reference.float(), candidate.float()
    delta = value - ref
    ref_rms = torch.sqrt(torch.mean(ref.square())).item()
    denominator = math.sqrt(torch.sum(ref.square()).item() *
                            torch.sum(value.square()).item())
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_rms": float(torch.sqrt(torch.mean(delta.square())).item() /
                              max(ref_rms, 1e-30)),
        "cosine": float(torch.sum(ref * value).item() / max(denominator, 1e-30)),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
    }


def _eligible(name: str, m: int, k: int) -> tuple[bool, str | None]:
    if m % 8:
        return False, "output dimension is not divisible by 8"
    alignment = 64 if name == "small_32x32x64" else 32
    if k % alignment:
        return False, f"K is not divisible by {alignment}"
    return True, None


def _inventory(artifact: Path) -> tuple[list[int], dict[tuple[int, int], int], Counter]:
    metadata = json.loads((artifact / "weights" / "metadata.json").read_text())
    handles: list[int] = []
    for entry in metadata["entries"]:
        handles.extend(pack_entry_handles(
            artifact / "weights", entry, bias=None, lookahead=1,
            max_texture_rows=32768))
    ext = extension()
    representative: dict[tuple[int, int], int] = {}
    counts: Counter = Counter()
    for handle in handles:
        m, k = map(int, ext.handle_info(handle)[:2])
        counts[(m, k)] += 1
        representative.setdefault((m, k), handle)
        ext.set_prefill(handle, 1, 32768)
    return handles, representative, counts


def tune_artifact(
    artifact: Path,
    output_path: Path | None,
    tokens: list[int],
    warmup: int = 5,
    measured: int = 30,
    seed: int = 20260827,
) -> dict:
    if warmup < 3 or measured < 10:
        raise ValueError("autotuning requires at least 3 warmups and 10 samples")
    artifact = artifact.resolve()
    buckets = [token_bucket(value) for value in tokens]
    if len(buckets) != len(set(buckets)):
        raise ValueError(
            "provide at most one representative token count per dispatch bucket")
    metadata_path = artifact / "weights" / "metadata.json"
    ext = extension()
    handles, representative, counts = _inventory(artifact)
    try:
        info_functions = {
            "large_256x128x32": ext.cutlass_texture_kernel_info,
            "medium_128x128x32": ext.cutlass_t1_kernel_info,
            "medium_128x64x32": ext.cutlass_t2_kernel_info,
            "medium_64x128x32": ext.cutlass_t3_kernel_info,
            "small_32x32x64": ext.cutlass_smalln_kernel_info,
        }
        kernel_info = {name: list(map(int, function()))
                       for name, function in info_functions.items()}
        selections: dict[str, dict] = {}
        all_rows: list[dict] = []
        for token_count in tokens:
            bucket = token_bucket(token_count)
            for shape_index, ((m, k), handle) in enumerate(sorted(representative.items())):
                generator = torch.Generator(device="cuda")
                generator.manual_seed(seed + token_count * 1009 + shape_index)
                x = torch.randn((token_count, k), device="cuda",
                                dtype=torch.bfloat16, generator=generator)
                x.mul_(0.125)
                ext.set_prefill_backend(handle, CANDIDATES["large_256x128x32"])
                with torch.inference_mode():
                    reference = ext.linear(handle, x)
                valid: list[str] = []
                correctness: dict[str, dict] = {}
                rejected: dict[str, str] = {}
                for name, backend in CANDIDATES.items():
                    supported, reason = _eligible(name, m, k)
                    if not supported:
                        rejected[name] = reason or "unsupported"
                        continue
                    ext.set_prefill_backend(handle, backend)
                    with torch.inference_mode():
                        candidate = ext.linear(handle, x)
                    metrics = _error(reference, candidate)
                    correctness[name] = metrics
                    if metrics["bitwise_equal"]:
                        valid.append(name)
                    else:
                        rejected[name] = (
                            f"correctness: rrms={metrics['relative_rms']:.6g}, "
                            f"cosine={metrics['cosine']:.9g}")
                    del candidate
                if "large_256x128x32" not in valid:
                    raise RuntimeError(f"baseline failed correctness for M={m},K={k}")

                for name in valid:
                    ext.set_prefill_backend(handle, CANDIDATES[name])
                    with torch.inference_mode():
                        for _ in range(warmup):
                            value = ext.linear(handle, x)
                            del value
                torch.cuda.synchronize()
                samples = {name: [] for name in valid}
                for iteration in range(measured):
                    order = valid[iteration % len(valid):] + valid[:iteration % len(valid)]
                    for name in order:
                        ext.set_prefill_backend(handle, CANDIDATES[name])
                        start = torch.cuda.Event(enable_timing=True)
                        stop = torch.cuda.Event(enable_timing=True)
                        with torch.inference_mode():
                            start.record()
                            value = ext.linear(handle, x)
                            stop.record()
                        stop.synchronize()
                        samples[name].append(float(start.elapsed_time(stop)))
                        del value, start, stop
                medians = {name: statistics.median(values)
                           for name, values in samples.items()}
                winner = min(valid, key=lambda name: medians[name])
                key = dispatch_key(token_count, m, k)
                record = {
                    "key": key, "tokens_measured": token_count,
                    "token_bucket": bucket, "M": m, "K": k,
                    "occurrences": int(counts[(m, k)]),
                    "winner": winner, "backend": CANDIDATES[winner],
                    "median_ms": medians[winner],
                    "baseline_ms": medians["large_256x128x32"],
                    "speedup_vs_baseline": medians["large_256x128x32"] / medians[winner],
                    "candidate_median_ms": medians,
                    "correctness_vs_baseline": correctness,
                    "rejected": rejected,
                }
                selections[key] = record
                all_rows.append(record)
                print(f"[autotune] tokens={token_count} M={m} K={k} "
                      f"winner={winner} {medians[winner]:.4f} ms "
                      f"speedup={record['speedup_vs_baseline']:.3f}x", flush=True)
                del x, reference
                torch.cuda.empty_cache()

        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        profile = {
            "schema_version": 1,
            "status": "complete",
            "artifact": str(artifact),
            "metadata_sha256": _sha256(metadata_path),
            "kernel_source_sha256": kernel_source_sha256(),
            "device": {
                "name": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "multiprocessor_count": int(properties.multi_processor_count),
                "total_memory": int(properties.total_memory),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            "protocol": {"tokens": tokens, "warmup": warmup,
                         "measured": measured, "seed": seed,
                         "bitwise_correctness_required": True},
            "candidates": CANDIDATES,
            "kernel_info": kernel_info,
            "selections": selections,
            "rows": all_rows,
        }
        target = output_path or profile_path(artifact)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(profile, indent=2) + "\n")
        print(f"[texelator] prefill profile saved to {target}", flush=True)
        return profile
    finally:
        free(handles)


def apply_profile(handles: list[int], tokens: int, profile: dict) -> bool:
    ext = extension()
    expected_hash = profile.get("kernel_source_sha256")
    if expected_hash != kernel_source_sha256():
        raise RuntimeError("autotune profile kernel hash does not match this build")
    current_cc = list(torch.cuda.get_device_capability())
    if profile.get("device", {}).get("compute_capability") != current_cc:
        raise RuntimeError("autotune profile compute capability does not match this GPU")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    profiled_sms = profile.get("device", {}).get("multiprocessor_count")
    if profiled_sms != int(properties.multi_processor_count):
        raise RuntimeError("autotune profile SM count does not match this GPU")
    complete = True
    for handle in handles:
        m, k = map(int, ext.handle_info(handle)[:2])
        key = dispatch_key(tokens, m, k)
        selection = profile.get("selections", {}).get(key)
        if selection is None:
            # Profiles are intentionally finite. Unknown long-context buckets
            # retain the conservative production mainloop.
            ext.set_prefill(handle, 1, 32768)
            ext.set_prefill_backend(handle, CANDIDATES["large_256x128x32"])
            complete = False
            continue
        ext.set_prefill(handle, 1, 32768)
        ext.set_prefill_backend(handle, int(selection["backend"]))
    return complete


def selected_profile(artifact: Path) -> tuple[dict | None, Path]:
    target = profile_path(artifact)
    if not target.is_file():
        return None, target
    profile = json.loads(target.read_text())
    expected = _sha256(artifact / "weights" / "metadata.json")
    if profile.get("metadata_sha256") != expected:
        raise RuntimeError("prefill profile does not match the encoded weights; rerun benchmark")
    if profile.get("kernel_source_sha256") != kernel_source_sha256():
        raise RuntimeError("prefill profile does not match this Texelator build; rerun benchmark")
    return profile, target
