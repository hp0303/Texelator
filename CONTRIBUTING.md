# Contributing

Bug reports and narrowly scoped portability fixes are welcome. Before opening a
pull request:

1. do not commit model weights, datasets, encoded weights, tokens, or machine caches;
2. preserve the BC4 representation and clearly identify any kernel arithmetic change;
3. add a shape/correctness test for every new model adapter or fused projection;
4. report the GPU, driver, CUDA toolkit, PyTorch version, model revision, command,
   correctness tolerance, and raw timing protocol for performance claims;
5. run `pytest` and `bash -n scripts/*.sh`.

Performance pull requests must provide before/after raw data from the same
process protocol. Results from different GPUs or unmatched model revisions are
not accepted as direct speedups.

