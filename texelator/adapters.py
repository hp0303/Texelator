from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Iterable

import torch.nn as nn


DEFAULT_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Common fused names are opt-in because they change the comparison scope.
FUSED_SUFFIXES = (
    "qkv_proj", "query_key_value", "c_attn", "Wqkv",
    "gate_up_proj", "dense_h_to_4h", "dense_4h_to_h",
)

_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)"),
)


@dataclass(frozen=True)
class LinearSpec:
    module_name: str
    artifact_key: str
    layer: int
    op: str
    M: int
    K: int
    bias: bool
    supported: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def layer_index(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def artifact_key(name: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).replace(".", "__")
    digest = hashlib.sha256(name.encode()).hexdigest()[:10]
    return f"{readable}--{digest}"


def discover_linears(
    model: nn.Module,
    suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    include_regex: str | None = None,
) -> list[LinearSpec]:
    suffixes = tuple(suffixes)
    extra = re.compile(include_regex) if include_regex else None
    found: list[LinearSpec] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        layer = layer_index(name)
        if layer is None:
            continue
        op = name.rsplit(".", 1)[-1]
        if op not in suffixes and not (extra and extra.search(name)):
            continue
        M, K = map(int, module.weight.shape)
        supported = K % 16 == 0
        found.append(LinearSpec(
            module_name=name,
            artifact_key=artifact_key(name),
            layer=layer,
            op=op,
            M=M,
            K=K,
            bias=module.bias is not None,
            supported=supported,
            reason="ok" if supported else "input dimension K is not divisible by 16",
        ))
    return sorted(found, key=lambda item: (item.layer, item.module_name))


def resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, child = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child


def model_slug(model_id: str, revision: str | None = None) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "--", model_id).strip("-")
    if revision and revision not in ("main", "default"):
        base += "--" + re.sub(r"[^A-Za-z0-9_.-]+", "-", revision)[:24]
    return base

