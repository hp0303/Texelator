#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE_CHECKPOINT COMPLETED_BC4_ARTIFACT OUTPUT_REPOSITORY" >&2
  exit 2
fi

source_checkpoint=$1
bc4_artifact=$2
output_repository=$3

texelator model register "$source_checkpoint" --name qwen38-package-source --replace
texelator package "$bc4_artifact" \
  --source qwen38-package-source \
  --model-id Qwen/Qwen3.8-27B \
  --output "$output_repository"

echo "Package complete: $output_repository"
echo "Validate on the target GPU before uploading it to Hugging Face."
