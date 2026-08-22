# Texelator Quick Start

Texelator keeps model acquisition, local-model registration, BC4 conversion,
GPU tuning, and inference as separate operations. The `run` command never
downloads or converts a model implicitly.

## Install

Install a CUDA-enabled PyTorch build, a CUDA toolkit containing `nvcc`,
`python3-dev`, `python3-venv`, and `build-essential` first. Then run:

```bash
bash scripts/install.sh
source .venv/bin/activate
texelator doctor
```

`texelator doctor` checks the environment and measures the hardware BC4
reconstruction palette for the current GPU.

## Download a Hugging Face model

```bash
texelator model install Qwen/Qwen2.5-3B --name qwen-3b
```

## Register a model already on disk

Registration records the path without copying or modifying the source model.

```bash
texelator model register /mnt/models/Qwen2.5-3B --name qwen-3b
texelator model list
```

## Convert the model

```bash
texelator convert qwen-3b --output /mnt/models/qwen-3b-texelator
```

The default conversion uses 8,192 WikiText-2 calibration tokens and the BC4
palette measured on the current GPU. If conversion is interrupted, run the same
command again to resume from verified matrix artifacts.

## Tune the request lookahead for the GPU

```bash
texelator tune /mnt/models/qwen-3b-texelator
```

The tuner checks correctness before measuring each candidate and saves the
selected profile in the converted artifact.

## Run the converted model

```bash
texelator run /mnt/models/qwen-3b-texelator "Explain texture compression."
```

Omit the prompt to start an interactive session:

```bash
texelator run /mnt/models/qwen-3b-texelator
```

You can also run the registered source model through the FP16 backend:

```bash
texelator run qwen-3b --backend fp16 "Hello"
```

A converted artifact remains linked to its source checkpoint. Keep the source
model at the registered path; moving or deleting it prevents the artifact from
loading tokenizer, configuration, and unsupported weights.
