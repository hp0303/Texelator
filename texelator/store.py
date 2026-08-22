from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

STATE_HOME = Path(
    os.environ.get("TEXELATOR_HOME", Path.home() / ".cache" / "texelator")
).expanduser().resolve()
REGISTRY_PATH = STATE_HOME / "models.json"

MODEL_ALIASES = {
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B",
    "deepseek-coder:1.3b": "deepseek-ai/deepseek-coder-1.3b-base",
    "mistral:7b": "mistralai/Mistral-7B-v0.3",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


@dataclass(frozen=True)
class ModelRecord:
    name: str
    source: str
    kind: str
    model_id: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _load() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    value = json.loads(REGISTRY_PATH.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"invalid Texelator registry: {REGISTRY_PATH}")
    return value


def _save(records: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REGISTRY_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    temporary.replace(REGISTRY_PATH)


def validate_local_model(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise RuntimeError(f"not a Hugging Face model directory (config.json missing): {path}")
    weight_markers = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    index_markers = list(path.glob("*.safetensors.index.json")) + list(path.glob("pytorch_model*.bin.index.json"))
    if not weight_markers and not index_markers:
        raise RuntimeError(f"no safetensors or PyTorch checkpoint weights found in {path}")
    return path


def register(record: ModelRecord, replace: bool = False) -> ModelRecord:
    records = _load()
    if record.name in records and not replace:
        raise RuntimeError(
            f"model name {record.name!r} is already registered; use --replace to update it"
        )
    records[record.name] = record.to_dict()
    _save(records)
    return record


def register_local(path: str | Path, name: str | None = None, replace: bool = False) -> ModelRecord:
    resolved = validate_local_model(Path(path))
    return register(
        ModelRecord(name=name or safe_name(resolved.name), source=str(resolved), kind="local"),
        replace=replace,
    )


def install_hub(
    model: str,
    name: str | None = None,
    revision: str = "main",
    local_dir: str | None = None,
    replace: bool = False,
) -> ModelRecord:
    from huggingface_hub import snapshot_download

    model_id = MODEL_ALIASES.get(model.lower(), model)
    destination = str(Path(local_dir).expanduser().resolve()) if local_dir else None
    path = snapshot_download(repo_id=model_id, revision=revision, local_dir=destination)
    resolved = validate_local_model(Path(path))
    default_name = safe_name(model if model.lower() in MODEL_ALIASES else model_id.rsplit("/", 1)[-1])
    return register(
        ModelRecord(
            name=name or default_name,
            source=str(resolved),
            kind="hub",
            model_id=model_id,
            revision=revision,
        ),
        replace=replace,
    )


def records() -> list[ModelRecord]:
    return [ModelRecord(**value) for _, value in sorted(_load().items())]


def resolve_source(value: str) -> ModelRecord:
    candidate = Path(value).expanduser()
    if candidate.exists():
        resolved = validate_local_model(candidate)
        return ModelRecord(name=safe_name(resolved.name), source=str(resolved), kind="local")
    item = _load().get(value)
    if item:
        record = ModelRecord(**item)
        validate_local_model(Path(record.source))
        return record
    raise RuntimeError(
        f"unknown model {value!r}. Install it with `texelator model install {value}` or "
        "register an existing directory with `texelator model register PATH --name NAME`."
    )


def artifact_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir() and (candidate / "texelator.json").exists():
        return candidate.resolve()
    managed = STATE_HOME / "artifacts" / value
    if (managed / "texelator.json").exists():
        return managed.resolve()
    raise RuntimeError(f"Texelator artifact not found: {value}")
