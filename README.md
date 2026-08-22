# Texelator

Texelator is an experimental inference runtime that stores supported language-model
linear weights as signed BC4 blocks and reconstructs them with the GPU Texture Unit.
It does **not** change the Transformer architecture. The source model remains a normal
Hugging Face checkpoint; conversion creates a separate, linked BC4 artifact.

This project is not affiliated with the unrelated Texelator Core tool for Autodesk Maya.

The public repository is for trying the runtime. It intentionally excludes model
weights, Hugging Face caches, research benchmark suites, raw profiler traces, and the
Q0--Q7 experiment directories.

> Current scope: NVIDIA CUDA, FP16, dense decoder-only models, and batch-one inference.
> RTX 4060 and RTX 4080 SUPER are validated. Other GPUs are checked at runtime and
> should be treated as experimental until independently validated.

## Install

WSL2 or Linux, a CUDA-enabled PyTorch installation, Python development headers,
Ninja, and a CUDA toolkit containing `nvcc` are required. On Ubuntu/WSL, install
`python3-venv`, `python3-dev`, and `build-essential` first. Install the PyTorch
build appropriate for your GPU before running:

```bash
git clone https://github.com/OWNER/texelator.git
cd texelator
bash scripts/install.sh
source .venv/bin/activate
texelator doctor
```

`doctor` measures and hashes the BC4 reconstruction palette of the current GPU.

## Five explicit commands

Texelator does not hide model download or conversion inside `run`.

### 1. Download a model

```bash
texelator model install Qwen/Qwen2.5-3B --name qwen-3b
```

### 2. Or register a model you already have

```bash
texelator model register /mnt/models/Qwen2.5-3B --name qwen-3b
```

Registration is read-only: files are not copied or modified. Supported sources are
ordinary Hugging Face directories with `config.json`, tokenizer files, and FP16,
BF16, or FP32 safetensors/PyTorch weights. GGUF and already packed GPTQ/AWQ weights
cannot be converted because the encoder needs the original floating-point weights.

```bash
texelator model list
```

### 3. Convert separately

```bash
texelator convert qwen-3b --output /mnt/models/qwen-3b-texelator
```

The default encoder uses 8,192 WikiText-2 calibration tokens, the measured hardware
palette, activation-weighted endpoint fitting, and the frozen ±1 integer search.
Conversion resumes verified matrix files after interruption. The source model is
never overwritten.

Use a local calibration text if network access is unavailable:

```bash
texelator convert qwen-3b \
  --calibration-file /mnt/data/calibration.txt \
  --output /mnt/models/qwen-3b-texelator
```

### 4. Tune the texture-request lookahead

```bash
texelator tune /mnt/models/qwen-3b-texelator
```

Candidates `{1,2,3,4,6,8}` must pass an output-equivalence check before timing.
The selected profile is stored inside the converted artifact. Without a matching
profile, execution uses the safe `K=1` schedule and prints a warning.

### 5. Run

```bash
texelator run /mnt/models/qwen-3b-texelator "Explain BC4 compression."
```

Omit the prompt for an interactive session:

```bash
texelator run /mnt/models/qwen-3b-texelator
```

Run the registered original model without BC4:

```bash
texelator run qwen-3b --backend fp16 "Hello"
```

By default, converted linears are used for both prefill and decode so dense linear
weights can be released after installation. This saves VRAM but does not accelerate
prefill. `--fp16-prefill` retains the original linears and routes multi-token shapes
through them, matching the paper's decode-only operator boundary at higher VRAM cost.

## Local state

The default state directory is `~/.cache/texelator`:

```text
~/.cache/texelator/
├── models.json
├── hardware/sm_89/{palette.bin,palette.json,environment.json}
├── artifacts/<name>/
└── build/
```

Move it to another disk with:

```bash
export TEXELATOR_HOME=/mnt/fast/texelator
export HF_HOME=/mnt/models/huggingface
```

Converted artifacts are linked: embeddings, tokenizer, configuration, and unsupported
weights are loaded from the registered source directory. Keep that directory available.

## Model compatibility

Texelator discovers ordinary projections by suffix rather than hard-coding a Qwen
layer path. The default set is `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`. A weight is eligible when its input dimension is divisible
by 16. Unsupported modules remain on the original FP16 path.

- Validated: Qwen2.5 0.5B, 1.5B, 3B; DeepSeek-Coder 1.3B.
- Structurally compatible but not yet fully validated: other dense Hugging Face causal LMs
  exposing ordinary `torch.nn.Linear` projections.
- Not automatic: MoE scheduling, `Conv1D`, non-CUDA backends, tensor-parallel conversion,
  and arbitrary fused projection semantics.

See [local model guidance](docs/local-models.md) and
[support boundaries](docs/supported-models.md).

For a compact end-to-end walkthrough, see [QUICKSTART.md](QUICKSTART.md).

## Paper

The current manuscript is included as [paper/Texelator.pdf](paper/Texelator.pdf).
Performance claims in the paper use an audited native-vLLM harness; the simple
Transformers CLI in this repository is intended for accessibility, not for reproducing
the paper's comparison tables.

## License

Texelator is released under the MIT License. Model checkpoints and datasets retain
their own licenses and are not distributed in this repository.
