from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def load_palette(path: Path, device: torch.device | str = "cuda") -> torch.Tensor:
    raw = np.fromfile(path, dtype="<f4")
    expected = 255 * 255 * 8
    if raw.size != expected:
        raise ValueError(f"palette has {raw.size} floats, expected {expected}: {path}")
    return torch.from_numpy(raw.reshape(255, 255, 8)).to(device)


def actual_palette(lut: torch.Tensor, endpoint0: torch.Tensor, endpoint1: torch.Tensor) -> torch.Tensor:
    return lut[(endpoint1 + 127).long(), (endpoint0 + 127).long()]


def nearest(values: torch.Tensor, palette: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    best = torch.full_like(values, float("inf"))
    indices = torch.zeros(values.shape, device=values.device, dtype=torch.uint8)
    for index in range(8):
        distance = (values - palette[:, index:index + 1]).abs()
        better = distance < best
        best = torch.where(better, distance, best)
        indices = torch.where(better, torch.full_like(indices, index), indices)
    return indices, torch.gather(palette, 1, indices.long())


@torch.inference_mode()
def endpoint_opt(
    weight: torch.Tensor,
    hdiag: torch.Tensor,
    scale: torch.Tensor,
    lut: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hardware-exact E3 projection for one row chunk."""
    m, k = weight.shape
    if k % 16:
        raise ValueError(f"K={k} is not divisible by one BC4 block (16)")
    blocks = k // 16
    values = (weight.float() / scale[:, None]).clamp(-1, 1).reshape(-1, 16)
    block_h = hdiag.float().reshape(blocks, 16).repeat(m, 1)
    endpoint0 = torch.round(values.max(1).values * 127).clamp(-127, 127).to(torch.int16)
    endpoint1 = torch.round(values.min(1).values * 127).clamp(-127, 127).to(torch.int16)
    palette = actual_palette(lut, endpoint0, endpoint1)
    indices, _ = nearest(values, palette)

    high = (endpoint0 > endpoint1)[:, None]
    selector = indices.long()
    a = torch.zeros_like(values)
    b = torch.zeros_like(values)
    c = torch.zeros_like(values)
    a = torch.where(selector == 0, torch.ones_like(a), a)
    b = torch.where(selector == 1, torch.ones_like(b), b)
    for s in range(2, 8):
        selected = selector == s
        a = torch.where(selected & high, torch.full_like(a, (8 - s) / 7), a)
        b = torch.where(selected & high, torch.full_like(b, (s - 1) / 7), b)
        if s < 6:
            a = torch.where(selected & ~high, torch.full_like(a, (6 - s) / 5), a)
            b = torch.where(selected & ~high, torch.full_like(b, (s - 1) / 5), b)
        elif s == 6:
            c = torch.where(selected & ~high, torch.full_like(c, -1), c)
        else:
            c = torch.where(selected & ~high, torch.ones_like(c), c)

    target = values - c
    saa = (block_h * a * a).sum(1)
    sbb = (block_h * b * b).sum(1)
    sab = (block_h * a * b).sum(1)
    say = (block_h * a * target).sum(1)
    sby = (block_h * b * target).sum(1)
    determinant = saa * sbb - sab * sab
    fallback = determinant.abs() < 1e-12
    safe_det = determinant.clamp_min(1e-12)
    u = ((say * sbb - sby * sab) / safe_det).clamp(-1, 1)
    z = ((sby * saa - say * sab) / safe_det).clamp(-1, 1)
    u = torch.where(fallback, endpoint0.float() / 127, u)
    z = torch.where(fallback, endpoint1.float() / 127, z)
    center0 = torch.round(u * 127).to(torch.int16)
    center1 = torch.round(z * 127).to(torch.int16)

    best = torch.full((values.shape[0],), float("inf"), device=values.device)
    best0, best1, best_indices = endpoint0, endpoint1, indices
    for delta0 in (-1, 0, 1):
        for delta1 in (-1, 0, 1):
            candidate0 = (center0 + delta0).clamp(-127, 127)
            candidate1 = (center1 + delta1).clamp(-127, 127)
            candidate_palette = actual_palette(lut, candidate0, candidate1)
            candidate_indices, quantized = nearest(values, candidate_palette)
            error = (block_h * (values - quantized).square()).sum(1)
            better = error < best
            best = torch.where(better, error, best)
            best0 = torch.where(better, candidate0, best0)
            best1 = torch.where(better, candidate1, best1)
            best_indices = torch.where(better[:, None], candidate_indices, best_indices)
    return best0, best1, best_indices


def pack_blocks(endpoint0: torch.Tensor, endpoint1: torch.Tensor, indices: torch.Tensor) -> np.ndarray:
    first = endpoint0.cpu().numpy().astype(np.int8).view(np.uint8)
    second = endpoint1.cpu().numpy().astype(np.int8).view(np.uint8)
    selectors = indices.cpu().numpy().astype(np.uint64)
    bits = np.zeros(selectors.shape[0], dtype=np.uint64)
    for position in range(16):
        bits |= selectors[:, position] << (3 * position)
    packed = np.empty((selectors.shape[0], 8), dtype=np.uint8)
    packed[:, 0] = first
    packed[:, 1] = second
    for byte in range(6):
        packed[:, byte + 2] = (bits >> (8 * byte)) & 255
    return packed


@torch.inference_mode()
def encode_linear_e3(
    weight: torch.Tensor,
    hdiag: torch.Tensor,
    lut: torch.Tensor,
    blocks_path: Path,
    scales_path: Path,
    rows_per_chunk: int = 128,
) -> dict:
    """Encode without materializing endpoint-search temporaries for all rows."""
    m, k = map(int, weight.shape)
    if k % 16:
        raise ValueError(f"unsupported K={k}; BC4 needs K % 16 == 0")
    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    blocks_hash = hashlib.sha256()
    scales_hash = hashlib.sha256()
    with blocks_path.open("wb") as blocks_file, scales_path.open("wb") as scales_file:
        for start in range(0, m, rows_per_chunk):
            stop = min(start + rows_per_chunk, m)
            rows = weight[start:stop].detach().float().to(lut.device)
            scales = rows.abs().amax(1).clamp_min(1e-12)
            endpoint0, endpoint1, indices = endpoint_opt(rows, hdiag.to(lut.device), scales, lut)
            packed = pack_blocks(endpoint0, endpoint1, indices).tobytes()
            scale_bytes = scales.cpu().numpy().astype("<f4", copy=False).tobytes()
            blocks_file.write(packed)
            scales_file.write(scale_bytes)
            blocks_hash.update(packed)
            scales_hash.update(scale_bytes)
    return {
        "M": m,
        "K": k,
        "blocks_file": blocks_path.name,
        "blocks_bytes": blocks_path.stat().st_size,
        "blocks_sha256": blocks_hash.hexdigest(),
        "scales_file": scales_path.name,
        "scales_bytes": scales_path.stat().st_size,
        "scales_sha256": scales_hash.hexdigest(),
    }


def load_moments(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise TypeError(f"expected dict in {path}")
    return value


def metadata_sha(path: Path) -> str:
    return hashlib.sha256(json.dumps(json.loads(path.read_text()), sort_keys=True).encode()).hexdigest()
