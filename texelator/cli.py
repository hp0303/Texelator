from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .converter import convert_model
from .hardware import doctor_payload, ensure_hardware
from .modeling import input_device, load_model, load_tokenizer
from .store import (
    ModelRecord,
    artifact_path,
    install_hub,
    records,
    register_local,
    resolve_source,
)


def _print_record(record: ModelRecord) -> None:
    print(json.dumps(record.to_dict(), indent=2))


def command_model_install(args) -> None:
    _print_record(install_hub(
        args.model, name=args.name, revision=args.revision,
        local_dir=args.output, replace=args.replace,
    ))


def command_model_register(args) -> None:
    _print_record(register_local(args.path, name=args.name, replace=args.replace))


def command_model_list(_args) -> None:
    items = records()
    if not items:
        print("No models registered. Use `texelator model install` or `texelator model register`.")
        return
    print(f"{'NAME':26} {'KIND':8} SOURCE")
    for item in items:
        print(f"{item.name[:26]:26} {item.kind:8} {item.source}")


def command_doctor(args) -> None:
    print(json.dumps(doctor_payload(force=args.force), indent=2))


def command_convert(args) -> None:
    record = resolve_source(args.source)
    convert_model(
        record,
        output=args.output,
        name=args.name,
        calibration_tokens=args.calibration_tokens,
        calibration_chunk=args.calibration_chunk,
        calibration_file=args.calibration_file,
        rows_per_chunk=args.rows_per_chunk,
        device_map=args.device_map,
        include_regex=args.include_regex,
        resume=not args.no_resume,
    )


def command_tune(args) -> None:
    from .tuning import tune_artifact

    tune_artifact(
        artifact_path(args.artifact),
        warmup=args.warmup,
        measured=args.measured,
        runs=args.runs,
    )


def _prompt_ids(tokenizer, messages: list[dict], raw: bool) -> torch.Tensor:
    if not raw and getattr(tokenizer, "chat_template", None):
        value = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        return value if isinstance(value, torch.Tensor) else value.input_ids
    text = "\n".join(f"{item['role'].capitalize()}: {item['content']}" for item in messages)
    return tokenizer(text + "\nAssistant:", return_tensors="pt").input_ids


def _generate(model, tokenizer, messages: list[dict], args) -> str:
    from transformers import TextStreamer

    inputs = _prompt_ids(tokenizer, messages, args.raw).to(input_device(model))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation = {
        "input_ids": inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "streamer": streamer,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation.update({"temperature": args.temperature, "top_p": args.top_p})
    with torch.inference_mode():
        output = model.generate(**generation)
    return tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)


def _artifact_source(artifact: Path) -> ModelRecord:
    manifest = json.loads((artifact / "texelator.json").read_text())
    return ModelRecord(**manifest["source"])


def command_run(args) -> None:
    from .runtime import free, install
    from .tuning import selected_lookahead

    handles: list[int] = []
    if args.backend == "fp16":
        record = resolve_source(args.target)
        artifact = None
    else:
        artifact = artifact_path(args.target)
        record = _artifact_source(artifact)
        if not Path(record.source).exists():
            raise RuntimeError(
                f"linked source model is missing: {record.source}. Register or restore the original model."
            )
        current_palette = ensure_hardware()
        artifact_manifest = json.loads((artifact / "texelator.json").read_text())
        from .artifacts import sha256_file
        if artifact_manifest["hardware"]["palette_sha256"] != sha256_file(current_palette):
            raise RuntimeError("artifact palette does not match this GPU; reconvert the model on this machine")

    print(f"[texelator] loading {record.source}", flush=True)
    model = load_model(record, device_map=args.device_map)
    tokenizer = load_tokenizer(record)
    if artifact is not None:
        lookahead, profile = selected_lookahead(artifact)
        if profile is None:
            print("[texelator] no tuning profile; using safe K=1. Run `texelator tune ARTIFACT`.")
        handles = install(
            model,
            artifact / "weights",
            lookahead=lookahead,
            fp16_prefill=args.fp16_prefill,
        )
        print(f"[texelator] installed BC4 linears with K={lookahead}", flush=True)

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    try:
        if args.prompt is not None:
            messages.append({"role": "user", "content": args.prompt})
            _generate(model, tokenizer, messages, args)
            return
        print("Texelator chat. Type /bye to exit or /clear to reset.\n")
        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text in ("/bye", "/exit", "/quit"):
                break
            if text == "/clear":
                messages = [{"role": "system", "content": args.system}] if args.system else []
                print("conversation cleared")
                continue
            if not text:
                continue
            messages.append({"role": "user", "content": text})
            reply = _generate(model, tokenizer, messages, args)
            messages.append({"role": "assistant", "content": reply})
    finally:
        free(handles)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="texelator",
        description="Run local Hugging Face language models through GPU BC4 texture hardware",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    model = commands.add_parser("model", help="download or register source models")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    install_parser = model_commands.add_parser("install", help="download a Hugging Face model only")
    install_parser.add_argument("model")
    install_parser.add_argument("--name")
    install_parser.add_argument("--revision", default="main")
    install_parser.add_argument("--output", help="optional local download directory")
    install_parser.add_argument("--replace", action="store_true")
    install_parser.set_defaults(function=command_model_install)
    register_parser = model_commands.add_parser("register", help="register an existing local model without copying")
    register_parser.add_argument("path")
    register_parser.add_argument("--name", required=True)
    register_parser.add_argument("--replace", action="store_true")
    register_parser.set_defaults(function=command_model_register)
    list_parser = model_commands.add_parser("list", help="list installed and registered source models")
    list_parser.set_defaults(function=command_model_list)

    doctor = commands.add_parser("doctor", help="verify CUDA BC4 support and measure this GPU's palette")
    doctor.add_argument("--force", action="store_true")
    doctor.set_defaults(function=command_doctor)

    convert = commands.add_parser("convert", help="convert a registered source model to Texelator BC4")
    convert.add_argument("source", help="registered model name or existing local model directory")
    convert.add_argument("--name")
    convert.add_argument("--output")
    convert.add_argument("--device-map", default="cuda")
    convert.add_argument("--calibration-file")
    convert.add_argument("--calibration-tokens", type=int, default=8192)
    convert.add_argument("--calibration-chunk", type=int, default=2048)
    convert.add_argument("--rows-per-chunk", type=int, default=128)
    convert.add_argument("--include-regex")
    convert.add_argument("--no-resume", action="store_true")
    convert.set_defaults(function=command_convert)

    tune = commands.add_parser("tune", help="select rolling texture-request lookahead for this GPU")
    tune.add_argument("artifact")
    tune.add_argument("--warmup", type=int, default=10)
    tune.add_argument("--measured", type=int, default=50)
    tune.add_argument("--runs", type=int, default=3)
    tune.set_defaults(function=command_tune)

    run = commands.add_parser("run", help="run a converted artifact or an FP16 source model")
    run.add_argument("target")
    run.add_argument("prompt", nargs="?")
    run.add_argument("--backend", choices=("texelator", "fp16"), default="texelator")
    run.add_argument("--device-map", default="cuda")
    run.add_argument("--fp16-prefill", action="store_true", help="retain dense linears for prefill (uses more VRAM)")
    run.add_argument("--system")
    run.add_argument("--max-new-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=0.9)
    run.add_argument("--raw", action="store_true")
    run.set_defaults(function=command_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
