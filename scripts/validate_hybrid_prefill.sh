#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT="${1:-qwen3.8:27b}"
TOKENS="${2:-512}"
OUTPUT="${3:-$ROOT/results/prefill-${TOKENS}.json}"

if [[ ! -x "$ROOT/.venv/bin/texelator" ]]; then
  echo "Texelator is not installed in $ROOT/.venv; run bash scripts/install.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
"$ROOT/.venv/bin/texelator" prefill-benchmark "$ARTIFACT" \
  --tokens "$TOKENS" --warmup 1 --runs 3 --output "$OUTPUT"

echo "Hybrid prefill validation saved to $OUTPUT"
