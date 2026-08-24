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


def _resolve_runtime_artifact(target: str, auto_pull: bool = True) -> Path:
    from .catalog import MODEL_CATALOG, pull_artifact, resolve_artifact

    artifact = resolve_artifact(target)
    if artifact is not None:
        return artifact
    if auto_pull and target.lower() in MODEL_CATALOG:
        return Path(pull_artifact(target).path)
    if not auto_pull and target.lower() in MODEL_CATALOG:
        raise RuntimeError(
            f"model {target!r} is not downloaded; run `texelator pull {target}` first"
        )
    # Preserve the original linked-artifact lookup and its useful error message.
    return artifact_path(target)


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
    from .catalog import artifact_records
    artifacts = artifact_records()
    if not items and not artifacts:
        print("No models installed. Use `texelator pull` or `texelator model install`.")
        return
    if items:
        print("SOURCE MODELS")
        print(f"{'NAME':26} {'KIND':8} SOURCE")
        for item in items:
            print(f"{item.name[:26]:26} {item.kind:8} {item.source}")
    if artifacts:
        print("\nTEXELATOR ARTIFACTS")
        print(f"{'NAME':26} {'GPU':8} PATH")
        for item in artifacts:
            print(f"{item.name[:26]:26} {(item.compute_capability or '-')[:8]:8} {item.path}")


def command_doctor(args) -> None:
    print(json.dumps(doctor_payload(force=args.force), indent=2))


def command_ptq(args) -> None:
    from .catalog import ArtifactRecord, current_capability, register_artifact

    record = resolve_source(args.source)
    artifact = convert_model(
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
    registered = register_artifact(ArtifactRecord(
        name=args.name or artifact.name,
        path=str(artifact),
        compute_capability=current_capability(),
    ))
    print(json.dumps(registered.to_dict(), indent=2))


def command_pull(args) -> None:
    from .catalog import pull_artifact

    record = pull_artifact(
        args.model, repo_id=args.repo, revision=args.revision, output=args.output,
    )
    print(json.dumps(record.to_dict(), indent=2))


def command_package(args) -> None:
    from .packager import package_standalone_qwen38

    source = resolve_source(args.source)
    result = package_standalone_qwen38(
        _resolve_runtime_artifact(args.artifact, auto_pull=False),
        source.source,
        args.output,
        model_id=args.model_id or source.model_id or source.source,
    )
    print(result)


def command_benchmark(args) -> None:
    from .tuning import tune_artifact

    tune_artifact(
        _resolve_runtime_artifact(args.artifact, auto_pull=False),
        warmup=args.warmup,
        measured=args.measured,
        runs=args.runs,
    )


def _prompt_ids(tokenizer, messages: list[dict], raw: bool, thinking: bool = False) -> torch.Tensor:
    if not raw and getattr(tokenizer, "chat_template", None):
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
        return value if isinstance(value, torch.Tensor) else value.input_ids
    text = "\n".join(f"{item['role'].capitalize()}: {item['content']}" for item in messages)
    return tokenizer(text + "\nAssistant:", return_tensors="pt").input_ids


def _remaining_context_tokens(model, tokenizer, prompt_tokens: int) -> int:
    limits = []
    model_config = getattr(model, "config", None)
    for config in (model_config, getattr(model_config, "text_config", None)):
        value = getattr(config, "max_position_embeddings", None)
        if isinstance(value, int) and 0 < value < 2**31:
            limits.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 2**31:
        limits.append(tokenizer_limit)
    context_limit = min(limits) if limits else 2**31 - 1
    remaining = context_limit - prompt_tokens
    if remaining <= 0:
        raise RuntimeError(
            f"the prompt already occupies the model context window ({context_limit} tokens)"
        )
    return remaining


def _generate(model, tokenizer, messages: list[dict], args) -> str:
    from transformers import TextStreamer

    inputs = _prompt_ids(tokenizer, messages, args.raw, args.thinking).to(input_device(model))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation = {
        "input_ids": inputs,
        "do_sample": args.temperature > 0,
        "streamer": streamer,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if args.max_new_tokens is not None and args.max_new_tokens > 0:
        generation["max_new_tokens"] = args.max_new_tokens
    else:
        # No Texelator response cap: generation ends at EOS or the model context boundary.
        generation["max_new_tokens"] = _remaining_context_tokens(
            model, tokenizer, int(inputs.shape[-1])
        )
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
    from .standalone import is_standalone_artifact, load_standalone_qwen38
    from .tuning import selected_lookahead

    handles: list[int] = []
    lookahead: int | None = None
    if args.backend == "fp16":
        record = resolve_source(args.target)
        artifact = None
    else:
        artifact = _resolve_runtime_artifact(args.target, auto_pull=False)
        if is_standalone_artifact(artifact):
            lookahead, profile = selected_lookahead(artifact)
            if profile is None:
                raise RuntimeError(
                    "this model has not been benchmarked on the current GPU; run "
                    f"`texelator benchmark {args.target}` once before inference"
                )
            print(f"[texelator] loading standalone {artifact}", flush=True)
            model, tokenizer, handles = load_standalone_qwen38(artifact, lookahead=lookahead)
            print(f"[texelator] installed standalone AW-BC4 linears with K={lookahead}", flush=True)
            return _run_chat(model, tokenizer, handles, args)
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
        lookahead, profile = selected_lookahead(artifact)
        if profile is None:
            raise RuntimeError(
                "this model has not been benchmarked on the current GPU; run "
                f"`texelator benchmark {args.target}` once before inference"
            )

    print(f"[texelator] loading {record.source}", flush=True)
    model = load_model(record, device_map=args.device_map)
    tokenizer = load_tokenizer(record)
    if artifact is not None:
        assert lookahead is not None
        handles = install(
            model,
            artifact / "weights",
            lookahead=lookahead,
            fp16_prefill=args.fp16_prefill,
        )
        print(f"[texelator] installed BC4 linears with K={lookahead}", flush=True)

    return _run_chat(model, tokenizer, handles, args)


def _run_chat(model, tokenizer, handles: list[int], args) -> None:
    from .runtime import free

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    try:
        if args.prompt is not None:
            messages.append({"role": "user", "content": args.prompt})
            _generate(model, tokenizer, messages, args)
            return
        print("Texelator CLI chat. Type /help for commands.\n")
        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text in ("/bye", "/exit", "/quit"):
                break
            if text == "/help":
                print("/clear  clear conversation history")
                print("/thinking on   enable model reasoning")
                print("/thinking off  disable model reasoning")
                print("/bye    exit Texelator")
                continue
            if text in ("/thinking on", "/thinking off"):
                args.thinking = text.endswith("on")
                print(f"thinking mode: {'on' if args.thinking else 'off'}")
                continue
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

    pull = commands.add_parser(
        "pull", aliases=["download"],
        help="download a published standalone Texelator model",
    )
    pull.add_argument("model", help="model alias, for example qwen3.8:27b")
    pull.add_argument("--repo", help="override the Hugging Face artifact repository")
    pull.add_argument("--revision", default="main")
    pull.add_argument("--output")
    pull.set_defaults(function=command_pull)

    ptq = commands.add_parser(
        "ptq", aliases=["convert"],
        help="create activation-aware BC4 weights from a registered floating-point model",
    )
    ptq.add_argument("source", help="registered model name or existing local model directory")
    ptq.add_argument("--name")
    ptq.add_argument("--output")
    ptq.add_argument("--device-map", default="cuda")
    ptq.add_argument("--calibration-file")
    ptq.add_argument("--calibration-tokens", type=int, default=8192)
    ptq.add_argument("--calibration-chunk", type=int, default=2048)
    ptq.add_argument("--rows-per-chunk", type=int, default=128)
    ptq.add_argument("--include-regex")
    ptq.add_argument("--no-resume", action="store_true")
    ptq.set_defaults(function=command_ptq)

    package = commands.add_parser("package", help="build a standalone Hugging Face Texelator repository")
    package.add_argument("artifact", help="completed linked Texelator artifact")
    package.add_argument("--source", required=True, help="registered name or source checkpoint directory")
    package.add_argument("--output", required=True)
    package.add_argument("--model-id", help="upstream Hugging Face model identifier")
    package.set_defaults(function=command_package)

    benchmark = commands.add_parser(
        "benchmark", aliases=["tune"],
        help="select and save the texture-request lookahead for this GPU",
    )
    benchmark.add_argument("artifact")
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--measured", type=int, default=50)
    benchmark.add_argument("--runs", type=int, default=3)
    benchmark.set_defaults(function=command_benchmark)

    run = commands.add_parser("run", help="run a converted artifact or an FP16 source model")
    run.add_argument("target")
    run.add_argument("prompt", nargs="?")
    run.add_argument("--backend", choices=("texelator", "fp16"), default="texelator")
    run.add_argument("--device-map", default="cuda")
    run.add_argument("--fp16-prefill", action="store_true", help="retain dense linears for prefill (uses more VRAM)")
    run.add_argument("--system")
    run.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="optional response cap; omit or set 0 to generate until EOS/context limit",
    )
    thinking = run.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking", dest="thinking", action="store_true",
        help="allow models with a thinking-aware chat template to emit reasoning tokens",
    )
    thinking.add_argument(
        "--no-thinking", dest="thinking", action="store_false",
        help="disable reasoning tokens (default)",
    )
    run.set_defaults(thinking=False)
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
