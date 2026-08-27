from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .store import STATE_HOME, safe_name

ARTIFACT_REGISTRY = STATE_HOME / "artifacts.json"
MODEL_CATALOG = {
    "qwen3.8:27b": {
        "8.9": "hp0303/Qwen3.8-27B-Texelator-AWBC4",
        "12.0": "hp0303/Qwen3.8-27B-Texelator-AWBC4-BF16Cal",
    },
}


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    path: str
    repo_id: str | None = None
    revision: str | None = None
    compute_capability: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _load_registry() -> dict[str, dict]:
    if not ARTIFACT_REGISTRY.is_file():
        return {}
    value = json.loads(ARTIFACT_REGISTRY.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"invalid artifact registry: {ARTIFACT_REGISTRY}")
    return value


def _save_registry(value: dict[str, dict]) -> None:
    ARTIFACT_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_REGISTRY.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(ARTIFACT_REGISTRY)


def register_artifact(record: ArtifactRecord, replace: bool = True) -> ArtifactRecord:
    path = Path(record.path).expanduser().resolve()
    if not (path / "texelator.json").is_file():
        raise RuntimeError(f"not a Texelator artifact: {path}")
    records = _load_registry()
    if record.name in records and not replace:
        raise RuntimeError(f"artifact name already registered: {record.name}")
    normalized = ArtifactRecord(**{**record.to_dict(), "path": str(path)})
    records[record.name] = normalized.to_dict()
    _save_registry(records)
    return normalized


def artifact_records() -> list[ArtifactRecord]:
    return [ArtifactRecord(**value) for _, value in sorted(_load_registry().items())]


def resolve_artifact(value: str) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.is_dir() and (candidate / "texelator.json").is_file():
        return candidate.resolve()
    item = _load_registry().get(value)
    if item:
        path = Path(item["path"])
        if not (path / "texelator.json").is_file():
            raise RuntimeError(f"registered artifact is missing: {path}")
        return path.resolve()
    managed = STATE_HOME / "artifacts" / value
    if (managed / "texelator.json").is_file():
        return managed.resolve()
    return None


def current_capability() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("a visible NVIDIA CUDA GPU is required to select a Texelator model")
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def catalog_repo(alias: str, capability: str | None = None) -> str:
    normalized = alias.lower()
    capability = capability or current_capability()
    env_key = "TEXELATOR_MODEL_REPO_" + re.sub(r"[^A-Z0-9]+", "_", normalized.upper()).strip("_")
    override = os.environ.get(env_key)
    if override:
        return override
    variants = MODEL_CATALOG.get(normalized)
    if not variants:
        raise RuntimeError(f"unknown Texelator model alias: {alias}")
    repo = variants.get(capability)
    if not repo:
        available = ", ".join(sorted(variants))
        raise RuntimeError(
            f"{alias} is not published for compute capability {capability}; available variants: {available}"
        )
    return repo


def pull_artifact(
    alias: str,
    repo_id: str | None = None,
    revision: str = "main",
    output: str | Path | None = None,
) -> ArtifactRecord:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    if os.environ.get("TEXELATOR_HTTP_ONLY", "").upper() in {"1", "ON", "YES", "TRUE"}:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    capability = current_capability()
    repo_id = repo_id or catalog_repo(alias, capability)
    destination = (
        Path(output).expanduser().resolve()
        if output else STATE_HOME / "artifacts" / safe_name(alias)
    )
    destination.mkdir(parents=True, exist_ok=True)
    print(
        f"[texelator] pulling shared {repo_id}; local kernel target sm_{capability.replace('.', '')}",
        flush=True,
    )
    resolved = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=destination,
        max_workers=int(os.environ.get("TEXELATOR_DOWNLOAD_WORKERS", "4")),
    )).resolve()
    manifest_path = resolved / "texelator.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"downloaded repository is not a standalone Texelator model: {resolved}")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("hardware", {}).get("compute_capability")
    if expected and expected != capability:
        raise RuntimeError(f"downloaded model requires compute capability {expected}, found {capability}")
    return register_artifact(ArtifactRecord(
        name=alias.lower(),
        path=str(resolved),
        repo_id=repo_id,
        revision=revision,
        compute_capability=capability,
    ))
