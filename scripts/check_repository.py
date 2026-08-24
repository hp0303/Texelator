#!/usr/bin/env python3
"""Fail if model artifacts, caches, or machine-local paths enter the public tree."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", ".ruff_cache", "__pycache__", "build", "dist"}
BLOCKED_SUFFIXES = {
    ".safetensors", ".bc4", ".f32", ".pt", ".pth", ".gguf", ".bin", ".tar", ".gz"
}
MAX_BYTES = 5 * 1024 * 1024
ALLOWED_LARGE = {Path("research/paper/Texelator.pdf")}
PERSONAL_MARKERS = (
    b"/home/" + b"hp/",
    b"/root/" + b".cache/",
    b"Desktop/" + b"share",
)


def main() -> None:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked artifact type: {relative}")
        if path.stat().st_size > MAX_BYTES and relative not in ALLOWED_LARGE:
            failures.append(f"file exceeds 5 MiB: {relative}")
        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
            data = path.read_bytes()
            for marker in PERSONAL_MARKERS:
                if marker in data:
                    failures.append(f"machine-local path {marker!r}: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Repository hygiene: PASS")


if __name__ == "__main__":
    main()
