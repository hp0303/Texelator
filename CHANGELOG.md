# Changelog

## 0.3.0

- Added portable BF16 prefill autotuning across five CUTLASS Texture IteratorB tile
  shapes while preserving the existing texture GEMV decode path.
- Added an explicitly experimental MLP-only FP4 prefill mode for 512-token speed
  studies. It is disabled by default because it is not quality-equivalent to BF16.
- Added BF16-source/BF16-calibrated standalone artifact selection for RTX 5080.
- Added `texelator prefill-benchmark` with scalar-vs-hybrid correctness, wall timing,
  CUDA-event timing, throughput, environment capture, and JSON output.
- Made the one-time `benchmark` command save both decode and prefill selections.

## 0.2.0

- Added a native Windows 11 installer and command wrapper, validated on RTX 5080.
- Unified RTX 40/50 runtime selection around local palette verification and K tuning.
- Added terminal thinking-mode controls and removed the default generated-token cap.
- Separated end-user runtime files from paper data under `research/`.
- Added GPU-aware standalone model aliases and explicit Hugging Face model download.
- Added terminal chat with persistent conversation history.
- Require a one-time, correctness-checked K benchmark on each model/GPU pair before
  Texelator inference; profiles are kept in local machine state.
- Added `ptq` as the primary activation-aware BC4 conversion command while retaining
  `convert` as a compatibility alias.
- Added a Qwen3.8-27B text-only standalone packager with residual-weight sharding.
- Added output-row texture sharding for very tall projections such as the 248,320-row LM head.
- Added an in-extension hardware palette probe so precompiled wheels do not require `nvcc` at runtime.
- Require Jinja2 3.1 or newer for Transformers chat-template rendering.
- Keep CPU-offloaded embeddings while exposing the CUDA execution device to Transformers generation.
- Use an isolated environment with controlled PyTorch 2.11/CUDA 12.8 instead of inheriting an incompatible system build.
- Share one AW-BC4 model across RTX 40/50 GPUs when their measured hardware palette hashes match.

## 0.1.0

- Initial public Texelator runtime.
- Separate model installation, local registration, conversion, tuning, and execution commands.
- Hardware-exact activation-aware BC4 encoder.
- Rolling `tex2Dgather()` CUDA runtime with lookahead candidates 1, 2, 3, 4, 6, and 8.
- Read-only registration of existing Hugging Face model directories.
