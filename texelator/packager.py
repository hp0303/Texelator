from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import sha256_file, validate_encoded, write_json

TEXT_PREFIX = "model.language_model."
TOKENIZER_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "chat_template.jinja", "merges.txt", "vocab.json",
    "tokenizer.model", "preprocessor_config.json",
)
LEGAL_FILES = ("LICENSE", "LICENSE.txt", "NOTICE", "NOTICE.txt")


class CheckpointIndex:
    def __init__(self, source: Path):
        index = source / "model.safetensors.index.json"
        single = source / "model.safetensors"
        if index.is_file():
            self.weight_map = dict(json.loads(index.read_text()).get("weight_map", {}))
        elif single.is_file():
            with safe_open(str(single), framework="pt", device="cpu") as handle:
                self.weight_map = {name: single.name for name in handle}
        else:
            raise RuntimeError("standalone packaging requires a safetensors source checkpoint")
        if not self.weight_map:
            raise RuntimeError("source checkpoint has an empty weight map")
        self.shards = sorted(set(self.weight_map.values()))
        missing = [name for name in self.shards if not (source / name).is_file()]
        if missing:
            raise RuntimeError(f"source checkpoint is missing shards: {missing[:3]}")


def artifact_to_text_name(name: str) -> str:
    if name == "lm_head":
        return name
    if not name.startswith(TEXT_PREFIX):
        raise RuntimeError(f"unsupported standalone Qwen3.8 module name: {name}")
    return "model." + name[len(TEXT_PREFIX):]


def source_to_text_key(name: str) -> str | None:
    if name.startswith(TEXT_PREFIX):
        return "model." + name[len(TEXT_PREFIX):]
    return None


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _copy_public_files(source: Path, destination: Path) -> list[str]:
    copied = []
    for name in TOKENIZER_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
            copied.append(name)
    if "config.json" not in copied or "tokenizer_config.json" not in copied:
        raise RuntimeError("source config/tokenizer files are incomplete")
    for name in LEGAL_FILES:
        path = source / name
        if path.is_file():
            output_name = "UPSTREAM_" + name
            shutil.copy2(path, destination / output_name)
            copied.append(output_name)
    return copied


def _write_residual_weights(
    source: Path,
    destination: Path,
    index: CheckpointIndex,
    target_weights: set[str],
) -> dict:
    selected_by_shard: dict[str, list[str]] = {name: [] for name in index.shards}
    for source_key, shard in index.weight_map.items():
        mapped = source_to_text_key(source_key)
        if mapped is not None and source_key not in target_weights:
            selected_by_shard[shard].append(source_key)
    selected_by_shard = {key: sorted(value) for key, value in selected_by_shard.items() if value}
    if not selected_by_shard:
        raise RuntimeError("no source-precision Qwen3.8 text residual tensors were found")

    weight_map: dict[str, str] = {}
    shard_records = []
    total_size = 0
    count = len(selected_by_shard)
    for output_index, (source_shard, source_keys) in enumerate(selected_by_shard.items(), start=1):
        output_name = f"text-residual-{output_index:05d}-of-{count:05d}.safetensors"
        output_path = destination / output_name
        tensors = {}
        with safe_open(str(source / source_shard), framework="pt", device="cpu") as handle:
            for source_key in source_keys:
                mapped = source_to_text_key(source_key)
                if mapped is None or mapped in tensors or mapped in weight_map:
                    raise RuntimeError(f"invalid or duplicate residual tensor mapping: {source_key}")
                tensors[mapped] = handle.get_tensor(source_key).contiguous()
            temporary = output_path.with_suffix(output_path.suffix + ".tmp")
            save_file(tensors, temporary)
            temporary.replace(output_path)
        for mapped in tensors:
            weight_map[mapped] = output_name
        total_size += output_path.stat().st_size
        shard_records.append({
            "file": output_name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "tensors": len(tensors),
            "source_shard": source_shard,
        })
        print(f"[texelator] residual shard {output_index}/{count}: {output_name}", flush=True)
    payload = {
        "metadata": {"total_size": total_size, "tensor_count": len(weight_map)},
        "weight_map": weight_map,
        "shards": shard_records,
    }
    write_json(destination / "text-residual.safetensors.index.json", payload)
    return payload


def package_standalone_qwen38(
    artifact: str | Path,
    source: str | Path,
    output: str | Path,
    model_id: str = "Qwen/Qwen3.8-27B",
) -> Path:
    artifact = Path(artifact).expanduser().resolve()
    source = Path(source).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact / "texelator.json"
    weights = artifact / "weights"
    if not manifest_path.is_file() or not (weights / "metadata.json").is_file():
        raise RuntimeError(f"not a completed Texelator artifact: {artifact}")
    validation = validate_encoded(weights)
    if not validation["ok"]:
        raise RuntimeError("source artifact checksum validation failed")
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads((weights / "metadata.json").read_text())
    entries = metadata["entries"]
    if len(entries) != 497:
        raise RuntimeError(f"Qwen3.8-27B standalone package expects 497 matrices, found {len(entries)}")
    mapped_names = [artifact_to_text_name(entry["module_name"]) for entry in entries]
    if len(set(mapped_names)) != len(mapped_names):
        raise RuntimeError("standalone module mapping produced duplicate names")

    copied_public = _copy_public_files(source, destination)
    checkpoint = CheckpointIndex(source)
    target_weights = {
        entry.get("checkpoint_weight", f"{entry['module_name']}.weight") for entry in entries
    }
    residual = _write_residual_weights(source, destination, checkpoint, target_weights)

    destination_weights = destination / "weights"
    destination_weights.mkdir()
    link_modes = {"hardlink": 0, "copy": 0}
    for path in sorted(weights.iterdir()):
        if path.is_file():
            mode = _link_or_copy(path, destination_weights / path.name)
            link_modes[mode] += 1

    palette_source = Path(manifest["hardware"]["palette"])
    if not palette_source.is_file():
        raise RuntimeError(f"artifact hardware palette is missing: {palette_source}")
    palette_hash = sha256_file(palette_source)
    if palette_hash != manifest["hardware"]["palette_sha256"]:
        raise RuntimeError("artifact palette checksum mismatch")
    hardware_dir = destination / "hardware" / "common"
    hardware_dir.mkdir(parents=True)
    shutil.copy2(palette_source, hardware_dir / "palette.bin")
    for suffix in ("palette.json", "environment.json"):
        candidate = palette_source.with_name(suffix)
        if candidate.is_file():
            shutil.copy2(candidate, hardware_dir / suffix)

    standalone = {
        **manifest,
        "schema_version": 2,
        "name": "qwen3.8-27b-awbc4",
        "source": {"kind": "hub", "model_id": model_id, "revision": None},
        "linked_source": False,
        "hardware": {
            "compatibility_gate": "palette_sha256",
            "validated_compute_capabilities": ["8.9"],
            "palette": "hardware/common/palette.bin",
            "palette_sha256": palette_hash,
        },
        "standalone": {
            "model_family": "qwen3_5_text",
            "text_only": True,
            "weights": "weights",
            "residual_index": "text-residual.safetensors.index.json",
            "module_prefix_mapping": {"model.language_model.": "model."},
            "embedding_placement": "cpu-offload",
            "vision": "excluded",
            "mtp": "excluded",
        },
    }
    write_json(destination / "texelator.json", standalone)
    write_json(destination / "package_report.json", {
        "schema_version": 1,
        "source_model": model_id,
        "target_linears": len(entries),
        "bc4_bytes": metadata["bc4_bytes"],
        "scale_bytes": metadata["scale_bytes"],
        "residual_bytes": residual["metadata"]["total_size"],
        "residual_tensors": residual["metadata"]["tensor_count"],
        "public_files": copied_public,
        "weight_file_materialization": link_modes,
        "palette_sha256": palette_hash,
    })
    (destination / "README.md").write_text(
        "---\n"
        "base_model: Qwen/Qwen3.8-27B\n"
        "library_name: texelator\n"
        "pipeline_tag: text-generation\n"
        "tags:\n  - quantized\n  - bc4\n  - cuda\n"
        "---\n\n"
        "# Qwen3.8-27B Texelator AW-BC4\n\n"
        "Text-only AW-BC4 artifact for the Texelator runtime. The 64-layer text decoder "
        "and LM head use hardware-exact AW-BC4; embeddings and non-linear text tensors "
        "remain at source precision. Vision and MTP tensors are excluded.\n\n"
        "Source model: https://huggingface.co/Qwen/Qwen3.8-27B\n\n"
        "Requires an NVIDIA GPU whose measured BC4 palette matches the packaged palette hash.\n\n"
        "```bash\n"
        "texelator pull qwen3.8:27b\n"
        "texelator benchmark qwen3.8:27b\n"
        "texelator run qwen3.8:27b \"Hello\"\n"
        "```\n\n"
        "The one-time benchmark selects the request lookahead for the local GPU. "
        "Inference is blocked until that profile exists.\n\n"
        "The upstream model license and usage terms continue to apply.\n"
    )
    print(f"[texelator] standalone repository ready: {destination}")
    return destination
