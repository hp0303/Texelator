from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import torch

from .artifacts import environment_snapshot, write_json
from .modeling import input_device
from .runtime import PREFILL_THRESHOLD, PREFILL_TILE_ROWS, extension, free
from .standalone import is_standalone_artifact, load_standalone_qwen38
from .tuning import selected_lookahead


def _set_prefill(handles: list[int], threshold: int, tile_rows: int) -> None:
    ext = extension()
    for handle in handles:
        ext.set_prefill(handle, threshold, tile_rows)


def _forward(model, input_ids: torch.Tensor, use_cache: bool):
    with torch.inference_mode():
        return model.model(input_ids=input_ids, use_cache=use_cache, return_dict=True)


def _measure(model, input_ids: torch.Tensor, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        output = _forward(model, input_ids, use_cache=True)
        del output
    torch.cuda.synchronize()
    wall_values: list[float] = []
    gpu_values: list[float] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        start.record()
        output = _forward(model, input_ids, use_cache=True)
        stop.record()
        torch.cuda.synchronize()
        wall_values.append(time.perf_counter() - before)
        gpu_values.append(start.elapsed_time(stop) / 1000.0)
        del output
    tokens = int(input_ids.numel())
    wall_median = statistics.median(wall_values)
    gpu_median = statistics.median(gpu_values)
    return {
        "tokens": tokens,
        "warmup": warmup,
        "runs": runs,
        "wall_seconds": wall_values,
        "gpu_seconds": gpu_values,
        "wall_seconds_median": wall_median,
        "gpu_seconds_median": gpu_median,
        "wall_tokens_per_second": tokens / wall_median,
        "gpu_tokens_per_second": tokens / gpu_median,
    }


def benchmark_prefill(
    artifact: Path,
    tokens: int,
    warmup: int,
    runs: int,
    output: Path,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("prefill benchmark requires a visible CUDA GPU")
    if not is_standalone_artifact(artifact):
        raise RuntimeError("the prefill benchmark currently requires a standalone Texelator artifact")
    if tokens < PREFILL_THRESHOLD:
        raise ValueError(f"tokens must be at least the hybrid threshold ({PREFILL_THRESHOLD})")
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")
    lookahead, profile = selected_lookahead(artifact)
    if profile is None:
        raise RuntimeError("run `texelator benchmark <artifact>` before the prefill benchmark")
    model, tokenizer, handles = load_standalone_qwen38(artifact, lookahead=lookahead)
    try:
        generator = torch.Generator(device="cpu").manual_seed(0)
        upper = min(len(tokenizer), int(model.get_input_embeddings().weight.shape[0]))
        input_ids = torch.randint(0, upper, (1, tokens), generator=generator, dtype=torch.int64)
        input_ids = input_ids.to(input_device(model))

        correctness_tokens = min(tokens, 32)
        correctness_ids = input_ids[:, :correctness_tokens]
        _set_prefill(handles, 1 << 30, PREFILL_TILE_ROWS)
        reference = _forward(model, correctness_ids, use_cache=False).last_hidden_state.float()
        _set_prefill(handles, 1, PREFILL_TILE_ROWS)
        candidate = _forward(model, correctness_ids, use_cache=False).last_hidden_state.float()
        difference = candidate - reference
        reference_rms = float(torch.sqrt(torch.mean(reference * reference)).item())
        difference_rms = float(torch.sqrt(torch.mean(difference * difference)).item())
        cosine = float(torch.nn.functional.cosine_similarity(
            reference.reshape(1, -1), candidate.reshape(1, -1)
        ).item())
        correctness = {
            "tokens": correctness_tokens,
            "max_abs": float(difference.abs().max().item()),
            "relative_rms": difference_rms / max(reference_rms, 1e-12),
            "cosine": cosine,
        }
        del reference, candidate, difference
        if not math.isfinite(correctness["relative_rms"]) or correctness["relative_rms"] > 0.02:
            raise RuntimeError(f"hybrid prefill correctness failed: {correctness}")

        _set_prefill(handles, 1 << 30, PREFILL_TILE_ROWS)
        scalar = _measure(model, input_ids, warmup, runs)
        _set_prefill(handles, 1, PREFILL_TILE_ROWS)
        hybrid = _measure(model, input_ids, warmup, runs)
        result = {
            "schema_version": 1,
            "artifact": str(artifact.resolve()),
            "lookahead": lookahead,
            "prefill_tile_rows": PREFILL_TILE_ROWS,
            "correctness": correctness,
            "scalar_texture": scalar,
            "hybrid_texture_cublas": hybrid,
            "speedup_wall": scalar["wall_seconds_median"] / hybrid["wall_seconds_median"],
            "speedup_gpu": scalar["gpu_seconds_median"] / hybrid["gpu_seconds_median"],
            "environment": environment_snapshot(),
        }
        write_json(output, result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        _set_prefill(handles, PREFILL_THRESHOLD, PREFILL_TILE_ROWS)
        free(handles)
