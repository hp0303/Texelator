from __future__ import annotations

import gc
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.cpp_extension import load

from .adapters import resolve_parent

PACKAGE_ROOT = Path(__file__).resolve().parent
_EXTENSION = None
PREFILL_THRESHOLD = int(os.environ.get("TEXELATOR_PREFILL_THRESHOLD", "16"))
PREFILL_TILE_ROWS = int(os.environ.get("TEXELATOR_PREFILL_TILE_ROWS", "1024"))
FP4_HYBRID_SHAPES = {(5120, 17408), (17408, 5120)}
_FP4_HYBRID_ENABLED = False
_FP4_WORKSPACES: dict[tuple[int, int, int, int], dict[str, torch.Tensor]] = {}
_HANDLE_SHAPES: dict[int, tuple[int, int]] = {}


def extension(verbose: bool = False):
    global _EXTENSION
    if _EXTENSION is None:
        try:
            _EXTENSION = importlib.import_module("texelator._cuda")
            return _EXTENSION
        except ImportError:
            if sys.platform == "win32":
                raise RuntimeError(
                    "the native Windows CUDA extension is unavailable; rerun setup.ps1 "
                    "from this package to rebuild texelator._cuda.pyd"
                )
            if os.environ.get("TEXELATOR_DISABLE_JIT", "").upper() in {"1", "ON", "YES", "TRUE"}:
                raise RuntimeError(
                    "the prebuilt Texelator CUDA extension is unavailable and JIT compilation is disabled"
                )
        _EXTENSION = load(
            name="texelator_cuda_ext_v030",
            sources=[
                str(PACKAGE_ROOT / "cuda" / "extension.cpp"),
                str(PACKAGE_ROOT / "cuda" / "texelator_cuda.cu"),
                str(PACKAGE_ROOT / "cuda" / "cutlass_texture_mainloop.cu"),
                str(PACKAGE_ROOT / "cuda" / "cutlass_texture_t1.cu"),
                str(PACKAGE_ROOT / "cuda" / "cutlass_texture_t2.cu"),
                str(PACKAGE_ROOT / "cuda" / "cutlass_texture_t3.cu"),
                str(PACKAGE_ROOT / "cuda" / "cutlass_texture_smalln.cu"),
            ],
            extra_include_paths=[
                str(PACKAGE_ROOT.parent / "third_party" / "cutlass" / "include"),
            ],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math", "-lineinfo", "-Xptxas=-v",
            ],
            extra_ldflags=(["cublas.lib"] if sys.platform == "win32" else ["-lcublas"]),
            verbose=verbose,
        )
    return _EXTENSION


def set_fp4_hybrid(enabled: bool) -> None:
    """Enable the opt-in 512-token MLP FP4 prefill path.

    This is deliberately process-local and disabled by default because the
    measured hidden-state output is not numerically equivalent to BF16.
    """
    global _FP4_HYBRID_ENABLED
    _FP4_HYBRID_ENABLED = bool(enabled)


def _scale_storage_size(rows: int, columns: int) -> int:
    groups = columns // 16
    return ((rows + 127) // 128 * 128) * ((groups + 3) // 4 * 4)


def _fp4_workspace(device: int, tokens: int, m: int, k: int) -> dict[str, torch.Tensor]:
    key = (device, tokens, m, k)
    workspace = _FP4_WORKSPACES.get(key)
    if workspace is None:
        target = torch.device("cuda", device)
        workspace = {
            "activation": torch.empty((tokens, k // 2), dtype=torch.uint8,
                                      device=target),
            "activation_scale": torch.empty(_scale_storage_size(tokens, k),
                                            dtype=torch.uint8, device=target),
            "weight": torch.empty((m, k // 2), dtype=torch.uint8,
                                  device=target),
            "weight_scale": torch.empty(_scale_storage_size(m, k),
                                        dtype=torch.uint8, device=target),
        }
        _FP4_WORKSPACES[key] = workspace
    return workspace


def _fp4_staged_linear(handle: int, x: torch.Tensor, m: int, k: int) -> torch.Tensor:
    original_shape = list(x.shape)
    tokens = x.numel() // k
    flat = x.reshape(tokens, k)
    workspace = _fp4_workspace(x.device.index or 0, tokens, m, k)
    ext = extension()
    ext.quantize_activation_fp4_out(
        flat, workspace["activation"], workspace["activation_scale"])
    ext.decode_weight_fp4_out(
        handle, workspace["weight"], workspace["weight_scale"])
    output = torch._scaled_mm(
        workspace["activation"].view(torch.float4_e2m1fn_x2),
        workspace["weight"].view(torch.float4_e2m1fn_x2).transpose(0, 1),
        workspace["activation_scale"].view(torch.float8_e4m3fn),
        workspace["weight_scale"].view(torch.float8_e4m3fn),
        bias=None, out_dtype=torch.bfloat16)
    original_shape[-1] = m
    return output.reshape(original_shape)


def _linear_dispatch(handle: int, x: torch.Tensor) -> torch.Tensor:
    if _FP4_HYBRID_ENABLED and x.dtype == torch.bfloat16:
        shape = _HANDLE_SHAPES.get(handle)
        if shape is None:
            shape = tuple(map(int, extension().handle_info(handle)[:2]))
            _HANDLE_SHAPES[handle] = shape
        m, k = shape
        tokens = x.numel() // k
        if tokens == 512 and (m, k) in FP4_HYBRID_SHAPES:
            return _fp4_staged_linear(handle, x, m, k)
    return extension().linear(handle, x)


class TexelatorLinear(nn.Module):
    def __init__(self, handles: list[int], bias: nn.Parameter | None = None):
        super().__init__()
        if not handles:
            raise ValueError("TexelatorLinear requires at least one texture handle")
        self.handles = list(handles)
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contiguous = x.contiguous()
        outputs = [_linear_dispatch(handle, contiguous) for handle in self.handles]
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        return output if self.bias is None else output + self.bias


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
    extension().set_prefill(handle, PREFILL_THRESHOLD, PREFILL_TILE_ROWS)
    return handle


def pack_entry_handles(
    weights: Path,
    entry: dict,
    bias: torch.Tensor | None = None,
    lookahead: int = 1,
    max_texture_rows: int = 32768,
) -> list[int]:
    """Pack one matrix, splitting very tall outputs into independent textures."""
    m, k = int(entry["M"]), int(entry["K"])
    if max_texture_rows <= 0:
        raise ValueError("max_texture_rows must be positive")
    blocks_per_row = k // 16
    raw_blocks = np.memmap(weights / entry["blocks_file"], dtype=np.uint8, mode="r", shape=(m, blocks_per_row * 8))
    raw_scales = np.memmap(weights / entry["scales_file"], dtype="<f4", mode="r", shape=(m,))
    handles: list[int] = []
    try:
        for start in range(0, m, max_texture_rows):
            stop = min(start + max_texture_rows, m)
            block_tensor = torch.from_numpy(np.asarray(raw_blocks[start:stop]).copy().reshape(-1))
            scale_tensor = torch.from_numpy(np.asarray(raw_scales[start:stop]).copy())
            bias_slice = bias[start:stop] if bias is not None else None
            handle = extension().pack_encoded(block_tensor, scale_tensor, bias_slice)
            extension().set_lookahead(handle, lookahead)
            extension().set_prefill(handle, PREFILL_THRESHOLD, PREFILL_TILE_ROWS)
            handles.append(handle)
    except Exception:
        free(handles)
        raise
    return handles


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
            entry_handles = pack_entry_handles(weights, entry, bias=bias, lookahead=lookahead)
        quantized = TexelatorLinear(entry_handles)
        replacement = DecodeOnlyLinear(original, quantized) if fp16_prefill else quantized
        setattr(parent, child, replacement)
        handles.extend(entry_handles)
        if not fp16_prefill:
            del original
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model._texelator_handles = handles
    return handles


def install_standalone(
    model: nn.Module,
    weights: Path,
    entries: list[dict],
    name_mapper,
    lookahead: int = 1,
) -> list[int]:
    """Replace meta-initialized linears without retaining dense source weights."""
    handles: list[int] = []
    try:
        for entry in entries:
            mapped = name_mapper(entry["module_name"])
            parent, child = resolve_parent(model, mapped)
            original = getattr(parent, child)
            entry_handles = pack_entry_handles(weights, entry, lookahead=lookahead)
            setattr(parent, child, TexelatorLinear(entry_handles, bias=getattr(original, "bias", None)))
            handles.extend(entry_handles)
    except Exception:
        free(handles)
        raise
    model._texelator_handles = handles
    return handles


def free(handles: list[int]) -> None:
    global _FP4_HYBRID_ENABLED
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for handle in handles:
        extension().free(handle)
        _HANDLE_SHAPES.pop(handle, None)
    _FP4_HYBRID_ENABLED = False
    _FP4_WORKSPACES.clear()
