# Existing local models

Registering a local model stores only its absolute path in Texelator's registry:

```bash
texelator model register /data/models/Qwen2.5-3B --name qwen-3b
```

The directory must contain `config.json` and either safetensors or PyTorch checkpoint
weights. Sharded checkpoints and Hugging Face snapshot directories are accepted.

The source directory is opened read-only by the converter. BC4 blocks, row scales,
calibration moments, checksums, and tuning profiles are written only to the explicit
artifact output directory.

Examples:

```bash
# A manually downloaded model
texelator model register /mnt/models/llama-local --name llama-local

# An existing Hugging Face cache snapshot
texelator model register \
  ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/REVISION \
  --name qwen-cache
```

Moving or deleting a linked source invalidates the converted artifact. Register the
new path and reconvert; artifact manifests are deliberately immutable so a changed
checkpoint cannot be mistaken for the encoded source.

