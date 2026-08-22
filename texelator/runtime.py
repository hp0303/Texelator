from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from .adapters import resolve_parent


PACKAGE_ROOT = Path(__file__).resolve().parent
_EXTENSION = None


def extension(verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load(
            name="texelator_cuda_ext",
            sources=[
                str(PACKAGE_ROOT / "cuda" / "extension.cpp"),
                str(PACKAGE_ROOT / "cuda" / "texelator_cuda.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=verbose,
        )
    return _EXTENSION


class TexelatorLinear(nn.Module):
    def __init__(self, handle: int):
        super().__init__()
        self.handle = handle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return extension().linear(self.handle, x.contiguous())


class DecodeOnlyLinear(nn.Module):
    def __init__(self, original: nn.Module, quantized: nn.Module):
        super().__init__()
        self.original = original
        self.quantized = quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == x.shape[-1]:
            return self.quantized(x)
        return self.original(x)


def _cpu_u8(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=np.uint8).copy())


def _cpu_f32(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype="<f4").copy())


def pack_entry(weights: Path, entry: dict, bias: torch.Tensor | None = None, lookahead: int = 1) -> int:
    handle = extension().pack_encoded(
        _cpu_u8(weights / entry["blocks_file"]),
        _cpu_f32(weights / entry["scales_file"]),
        bias,
    )
    extension().set_lookahead(handle, lookahead)
    return handle


def install(
    model: nn.Module,
    weights: Path,
    lookahead: int = 1,
    fp16_prefill: bool = False,
) -> list[int]:
    metadata = json.loads((weights / "metadata.json").read_text())
    handles: list[int] = []
    for entry in metadata["entries"]:
        parent, child = resolve_parent(model, entry["module_name"])
        original = getattr(parent, child)
        device = original.weight.device
        if device.type != "cuda":
            raise RuntimeError(
                f"{entry['module_name']} is on {device}; selected linears must be CUDA resident"
            )
        bias = original.bias if getattr(original, "bias", None) is not None else None
        with torch.cuda.device(device):
            handle = pack_entry(weights, entry, bias=bias, lookahead=lookahead)
        quantized = TexelatorLinear(handle)
        replacement = DecodeOnlyLinear(original, quantized) if fp16_prefill else quantized
        setattr(parent, child, replacement)
        handles.append(handle)
        if not fp16_prefill:
            del original
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model._texelator_handles = handles
    return handles


def free(handles: list[int]) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for handle in handles:
        extension().free(handle)

