# Hybrid prefill

Texelator's original CUDA operator was designed for batch-one decode. For an input
with `T` tokens it launched the same texture GEMV independently `T` times. That is the
right work shape for `T=1`, but it provides no weight reuse during prompt prefill and
therefore makes long prompts unnecessarily slow.

The hybrid path keeps the published BC4 representation resident and switches only the
arithmetic schedule:

1. `tex2Dgather()` reconstructs four BC4 values per texture request into a bounded
   FP16 row-tile workspace.
2. cuBLAS reuses that tile across all prompt tokens with a Tensor Core GEMM.
3. The workspace is reused for the next row tile and released when the linear call
   completes.

Single-token decode continues to use the tuned rolling-gather GEMV. There is no
second dense checkpoint and no change to the BC4 payload, row scales, or palette.

## Verify on a GPU

Run the normal one-time decode benchmark first, then compare the legacy and hybrid
prefill paths in one process:

```bash
texelator benchmark qwen3.8:27b
texelator prefill-benchmark qwen3.8:27b \
  --tokens 512 --warmup 1 --runs 3 \
  --output texelator-prefill-512.json
```

The benchmark calls the Transformer body rather than the language-model head. Its
token/s number therefore measures prompt processing, not generated-token decode. It
records both wall-clock and CUDA-event timing. The run fails instead of publishing a
speed number when the hybrid full-model hidden state differs from the scalar path by
more than the correctness gate.

## Controls

- `TEXELATOR_PREFILL_THRESHOLD` selects the minimum flattened token count for the
  hybrid path. The default is `16`.
- `TEXELATOR_PREFILL_TILE_ROWS` caps the reconstructed row tile. The default is
  `1024`; the runtime reduces it automatically when free VRAM is low.

These are diagnostic controls, not model-format parameters. They do not affect the
saved K/lookahead profile used for single-token decode.
