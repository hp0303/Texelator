# Texelator research artifacts

This directory separates publication material from the end-user runtime.

- `paper/Texelator.pdf`: current manuscript.
- `results/paper/`: compact, non-weight measurements used by the public tables and
  figures, plus protocol and environment notes.
- `scripts/plot_paper_results.py`: regenerates publication figures from the frozen
  CSV files. It does not rerun inference.

Original checkpoints, converted BC4 weights, prompts, private evaluation data, raw
profiler traces, and internal Q0--Q7 workspaces are not distributed here.

From the repository root:

```bash
python -m pip install -e '.[paper]'
python research/scripts/plot_paper_results.py
```

Generated figures are written to `research/paper_figures/`.
