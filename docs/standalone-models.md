# Standalone models

Published Texelator models contain the AW-BC4 linear payload, source-precision
non-linear text tensors, tokenizer files, configuration, and the exact measured BC4
palette. They do not require the original Hugging Face checkpoint at inference time.

## User flow

```bash
texelator pull qwen3.8:27b
texelator benchmark qwen3.8:27b
texelator run qwen3.8:27b
```

The first command downloads the shared RTX 40/50 repository. The second creates the
mandatory local K profile. Omitting the prompt on `run` starts persistent terminal chat
with `/clear`, `/help`, and `/bye` commands.

An unpublished or private artifact can be selected explicitly:

```bash
texelator pull qwen3.8:27b --repo OWNER/PRIVATE-REPOSITORY
texelator benchmark qwen3.8:27b
texelator run qwen3.8:27b
```

Set `HF_TOKEN` for a private repository. If Hugging Face Xet is unreliable, retry in a
new shell with `TEXELATOR_HTTP_ONLY=1`.

## Publisher flow

The current Qwen3.8-27B package builder converts an already completed linked artifact
into a text-only standalone repository:

```bash
texelator model register /models/Qwen3.8-27B --name qwen38-source
texelator package /artifacts/Qwen--Qwen3.8-27B-bc4-e3 \
  --source qwen38-source \
  --model-id Qwen/Qwen3.8-27B \
  --output /models/Qwen3.8-27B-Texelator-AWBC4
texelator benchmark /models/Qwen3.8-27B-Texelator-AWBC4
texelator run /models/Qwen3.8-27B-Texelator-AWBC4 "Reply with one sentence."
```

The packager verifies every encoded matrix, excludes vision and MTP tensors, maps the
multimodal checkpoint's `model.language_model.*` names to the text-only runtime, copies
only required residual tensors, and includes a package report. Run the resulting model
locally before uploading it with `huggingface-cli upload` or `hf upload`.

The encoded weights are shared when GPUs produce the same bitwise BC4 palette. Runtime
compatibility is gated by the measured palette hash, while CUDA compilation and tuning
profiles remain local to each compute capability.
