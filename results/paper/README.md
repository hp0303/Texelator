# Frozen Paper Results

This directory contains the compact, paper-facing results used by the public
Texelator manuscript. It does not contain model checkpoints, converted BC4
weights, prompts, user data, profiler traces, or machine-local paths.

Files:

- `kernel_progression.csv`: matched RTX 4060 24-layer down-projection sweep.
- `rtx4080_context512.csv`: native-vLLM context-512 throughput medians and run ranges.
- `rtx4080_context_speedup.csv`: Texelator/GPTQ-Marlin throughput ratio at five contexts.
- `quality_scaling.csv`: held-out logit, perplexity, and downstream macro metrics.
- `environment.json`: hardware, software, protocol, palette hash, and metric definitions.

Regenerate the public figures with:

```bash
python -m pip install -e '.[paper]'
python scripts/plot_paper_results.py
```

The plotting command writes PNG and PDF files to `paper_figures/`. These CSVs
reproduce the published plots and tables; they do not rerun model inference.
