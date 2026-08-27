from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn
from torch.nn import functional
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .artifacts import sha256_file
from .hardware import ensure_hardware
from .packager import artifact_to_text_name
from .runtime import free, install_standalone


class CpuOffloadedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, output_device: torch.device, output_dtype: torch.dtype):
        super().__init__()
        if embedding.weight.device.type != "cpu":
            raise RuntimeError("CPU-offloaded embedding must start on CPU")
        self.embedding = embedding
        self.output_device = output_device
        self.output_dtype = output_dtype

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = functional.embedding(
            input_ids.to("cpu"),
            self.embedding.weight,
            self.embedding.padding_idx,
            self.embedding.max_norm,
            self.embedding.norm_type,
            self.embedding.scale_grad_by_freq,
            self.embedding.sparse,
        )
        return values.to(device=self.output_device, dtype=self.output_dtype, non_blocking=True)


def is_standalone_artifact(path: str | Path) -> bool:
    target = Path(path).expanduser()
    manifest = target / "texelator.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(value.get("standalone")) and not bool(value.get("linked_source", True))


def _verify_hardware(path: Path, manifest: dict) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("standalone Texelator inference requires an NVIDIA CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    actual_capability = f"{major}.{minor}"
    expected_capability = manifest["hardware"].get("compute_capability")
    if expected_capability and actual_capability != expected_capability:
        raise RuntimeError(
            f"this model requires compute capability {expected_capability}, found {actual_capability}; "
            "download the artifact for this GPU architecture"
        )
    bundled = path / manifest["hardware"]["palette"]
    if not bundled.is_file() or sha256_file(bundled) != manifest["hardware"]["palette_sha256"]:
        raise RuntimeError("bundled model palette is missing or corrupt")
    measured = ensure_hardware()
    if sha256_file(measured) != manifest["hardware"]["palette_sha256"]:
        raise RuntimeError(
            "the model palette does not match hardware reconstruction on this GPU; "
            "use a compatible model variant"
        )
    return measured


def _load_residual(model: nn.Module, path: Path, index_name: str) -> dict:
    index_path = path / index_name
    index = json.loads(index_path.read_text())
    weight_map = dict(index.get("weight_map", {}))
    expected = dict(model.named_parameters())
    unknown = sorted(set(weight_map) - set(expected))
    if unknown:
        raise RuntimeError(f"residual index contains unknown model tensors: {unknown[:3]}")
    loaded: set[str] = set()
    for shard in sorted(set(weight_map.values())):
        selected = sorted(name for name, filename in weight_map.items() if filename == shard)
        with safe_open(str(path / shard), framework="pt", device="cpu") as handle:
            state = {name: handle.get_tensor(name) for name in selected}
        model.load_state_dict(state, strict=False, assign=True)
        loaded.update(state)
        print(f"[texelator] loaded residual {shard} ({len(state)} tensors)", flush=True)
    missing = sorted(name for name, value in model.named_parameters() if value.device.type == "meta")
    if missing:
        raise RuntimeError(f"standalone residual package left meta parameters: {missing[:5]}")
    return {"tensors": len(loaded), "shards": len(set(weight_map.values()))}


def load_standalone_qwen38(path: str | Path, lookahead: int = 1):
    artifact = Path(path).expanduser().resolve()
    manifest = json.loads((artifact / "texelator.json").read_text())
    standalone = manifest.get("standalone", {})
    if standalone.get("model_family") != "qwen3_5_text":
        raise RuntimeError(f"unsupported standalone model family: {standalone.get('model_family')!r}")
    _verify_hardware(artifact, manifest)
    runtime_dtype_name = manifest.get("runtime_dtype", "float16")
    runtime_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if runtime_dtype_name not in runtime_dtypes:
        raise RuntimeError(f"unsupported Texelator runtime dtype: {runtime_dtype_name!r}")
    runtime_dtype = runtime_dtypes[runtime_dtype_name]
    config = AutoConfig.from_pretrained(artifact, local_files_only=True, trust_remote_code=False)
    text_config = config.text_config
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(text_config, trust_remote_code=False)

    entries = json.loads((artifact / "weights" / "metadata.json").read_text())["entries"]
    handles: list[int] = []
    try:
        handles = install_standalone(
            model,
            artifact / "weights",
            entries,
            name_mapper=artifact_to_text_name,
            lookahead=lookahead,
        )
        residual = _load_residual(model, artifact, standalone["residual_index"])
        device = torch.device("cuda", torch.cuda.current_device())
        # Register this directly on the root model (before traversing children).
        # A real one-element parameter is intentional: Transformers 5.15 may skip
        # empty parameters when deriving `model.device` for generation tensors.
        model.register_parameter(
            "_texelator_device_anchor",
            nn.Parameter(torch.zeros(1, device=device, dtype=runtime_dtype), requires_grad=False),
        )
        if model.device != device or model.dtype != runtime_dtype:
            raise RuntimeError(
                f"failed to expose CUDA {runtime_dtype_name} generation device: "
                f"device={model.device}, dtype={model.dtype}"
            )
        embedding = model.model.embed_tokens.to("cpu")
        model.model.embed_tokens = CpuOffloadedEmbedding(embedding, device, runtime_dtype)
        model.model.layers.to(device=device, dtype=runtime_dtype)
        model.model.norm.to(device=device, dtype=runtime_dtype)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb.to(device=device)
        if (artifact / "generation_config.json").is_file():
            model.generation_config = GenerationConfig.from_pretrained(artifact, local_files_only=True)
        model.eval()
        model._texelator_artifact = str(artifact)
        model._texelator_residual = residual
        tokenizer = AutoTokenizer.from_pretrained(artifact, local_files_only=True, trust_remote_code=False)
        return model, tokenizer, handles
    except Exception:
        free(handles)
        raise
