from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .adapters import DEFAULT_SUFFIXES, discover_linears
from .artifacts import environment_snapshot, sha256_file, validate_encoded, write_json
from .encoder import encode_linear_e3, load_palette
from .hardware import ensure_hardware
from .modeling import input_device, load_model, load_tokenizer
from .store import ModelRecord, STATE_HOME


def _calibration_ids(record: ModelRecord, tokens: int, calibration_file: str | None) -> torch.Tensor:
    tokenizer = load_tokenizer(record)
    if calibration_file:
        text = Path(calibration_file).expanduser().read_text(errors="replace")
        source = str(Path(calibration_file).expanduser().resolve())
    else:
        from datasets import load_dataset

        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(row["text"] for row in dataset)
        source = "Salesforce/wikitext:wikitext-2-raw-v1/train"
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=tokens,
    ).input_ids.flatten()
    if encoded.numel() < tokens:
        raise RuntimeError(f"calibration source {source} has {encoded.numel()} tokens, need {tokens}")
    return encoded[:tokens]


def _inspect(record: ModelRecord, include_regex: str | None) -> tuple[dict, list]:
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(record.source, local_files_only=True)
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(config)
    specs = discover_linears(empty, DEFAULT_SUFFIXES, include_regex)
    if not specs:
        raise RuntimeError(
            "no supported transformer linears were discovered; use --include-regex for nonstandard names"
        )
    manifest = {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "linears": [item.to_dict() for item in specs],
        "target_linears": len(specs),
        "supported_linears": sum(item.supported for item in specs),
    }
    return manifest, specs


@torch.inference_mode()
def _collect_moments(
    record: ModelRecord,
    specs,
    calibration_tokens: int,
    calibration_chunk: int,
    calibration_file: str | None,
    device_map: str,
) -> tuple[dict[str, torch.Tensor], dict]:
    model = load_model(record, device_map=device_map)
    ids = _calibration_ids(record, calibration_tokens, calibration_file)
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []
    for spec in specs:
        if not spec.supported:
            continue
        module = model.get_submodule(spec.module_name)

        def hook(_module, inputs, key=spec.artifact_key):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            diagonal = values.square().sum(0).cpu()
            sums[key] = sums.get(key, torch.zeros_like(diagonal)) + diagonal
            counts[key] = counts.get(key, 0) + values.shape[0]

        hooks.append(module.register_forward_pre_hook(hook))
    for start in range(0, calibration_tokens, calibration_chunk):
        stop = min(start + calibration_chunk, calibration_tokens)
        model(ids[start:stop].view(1, -1).to(input_device(model)), use_cache=False)
        print(f"[texelator] calibration {stop}/{calibration_tokens}", flush=True)
    for hook in hooks:
        hook.remove()
    moments = {key: sums[key] / counts[key] for key in sums}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return moments, {
        "tokens": calibration_tokens,
        "chunk": calibration_chunk,
        "source": str(Path(calibration_file).expanduser().resolve()) if calibration_file else
                  "Salesforce/wikitext:wikitext-2-raw-v1/train",
    }


def convert_model(
    record: ModelRecord,
    output: str | None = None,
    name: str | None = None,
    calibration_tokens: int = 8192,
    calibration_chunk: int = 2048,
    calibration_file: str | None = None,
    rows_per_chunk: int = 128,
    device_map: str = "cuda",
    include_regex: str | None = None,
    resume: bool = True,
) -> Path:
    destination = (
        Path(output).expanduser().resolve()
        if output else STATE_HOME / "artifacts" / (name or f"{record.name}-bc4")
    )
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / "weights"
    weights.mkdir(exist_ok=True)
    palette_path = ensure_hardware()
    palette_hash = sha256_file(palette_path)
    completed_artifact = destination / "texelator.json"
    if resume and completed_artifact.exists() and (weights / "metadata.json").exists():
        previous = json.loads(completed_artifact.read_text())
        same_source = previous.get("source", {}).get("source") == record.source
        same_palette = previous.get("hardware", {}).get("palette_sha256") == palette_hash
        same_tokens = previous.get("calibration", {}).get("tokens") == calibration_tokens
        validation = validate_encoded(weights)
        if same_source and same_palette and same_tokens and validation["ok"]:
            print(f"[texelator] verified artifact already ready: {destination}")
            return destination
        raise RuntimeError(
            "the output directory contains a completed artifact with different source, "
            "palette, calibration settings, or checksums; choose a new output directory"
        )
    inspection, specs = _inspect(record, include_regex)
    supported = [spec for spec in specs if spec.supported]
    if not supported:
        raise RuntimeError("the model contains no BC4-compatible linear weights")

    moments_path = destination / "calibration" / "diagonal_moments.pt"
    calibration_metadata = destination / "calibration" / "metadata.json"
    if resume and moments_path.exists() and calibration_metadata.exists():
        moments = torch.load(moments_path, map_location="cpu", weights_only=True)
        calibration = json.loads(calibration_metadata.read_text())
        if calibration.get("tokens") != calibration_tokens:
            raise RuntimeError("existing calibration token count differs; choose a new output directory")
        print("[texelator] using verified calibration cache", flush=True)
    else:
        moments, calibration = _collect_moments(
            record, specs, calibration_tokens, calibration_chunk,
            calibration_file, device_map,
        )
        moments_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = moments_path.with_suffix(".tmp")
        torch.save(moments, temporary)
        temporary.replace(moments_path)
        calibration["sha256"] = sha256_file(moments_path)
        write_json(calibration_metadata, calibration)

    lut = load_palette(palette_path)
    model = load_model(record, device_map=device_map)
    progress_path = weights / "progress.json"
    entries = json.loads(progress_path.read_text()).get("entries", []) if resume and progress_path.exists() else []
    completed = {entry["module_name"]: entry for entry in entries}
    for index, spec in enumerate(supported, start=1):
        blocks = weights / f"{spec.artifact_key}.bc4"
        scales = weights / f"{spec.artifact_key}.scales.f32"
        previous = completed.get(spec.module_name)
        if previous and blocks.exists() and scales.exists():
            if (sha256_file(blocks) == previous["blocks_sha256"] and
                    sha256_file(scales) == previous["scales_sha256"]):
                print(f"[texelator] skip verified {index}/{len(supported)} {spec.module_name}")
                continue
            entries.remove(previous)
        module = model.get_submodule(spec.module_name)
        if module.weight.device.type == "meta":
            raise RuntimeError(
                f"{spec.module_name} is disk-offloaded. Conversion requires selected weights on CPU or CUDA."
            )
        result = encode_linear_e3(
            module.weight.detach(), moments[spec.artifact_key], lut,
            blocks, scales, rows_per_chunk,
        )
        entry = {**spec.to_dict(), **result}
        entries.append(entry)
        completed[spec.module_name] = entry
        write_json(progress_path, {"entries": entries})
        print(f"[texelator] encoded {index}/{len(supported)} {spec.module_name}", flush=True)
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    metadata = {
        "schema_version": 1,
        "format": "BC4_SNORM + one FP32 scale per output row",
        "encoder": "hardware-exact activation-aware",
        "palette_sha256": palette_hash,
        "entries": entries,
        "bc4_bytes": sum(item["blocks_bytes"] for item in entries),
        "scale_bytes": sum(item["scales_bytes"] for item in entries),
    }
    write_json(weights / "metadata.json", metadata)
    progress_path.unlink(missing_ok=True)
    validation = validate_encoded(weights)
    write_json(destination / "validation.json", validation)
    if not validation["ok"]:
        raise RuntimeError("encoded artifact checksum validation failed")
    artifact = {
        "schema_version": 1,
        "format": "texelator-bc4-v1",
        "name": name or destination.name,
        "source": record.to_dict(),
        "inspection": inspection,
        "calibration": calibration,
        "hardware": {
            "palette": str(palette_path),
            "palette_sha256": palette_hash,
        },
        "weights": "weights",
        "validation": "validation.json",
        "linked_source": True,
    }
    write_json(destination / "texelator.json", artifact)
    write_json(destination / "environment.json", environment_snapshot())
    print(f"[texelator] artifact ready: {destination}")
    return destination
