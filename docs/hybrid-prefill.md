# BF16 and experimental FP4 prefill

Texelator keeps its texture-native GEMV for single-token decode. Multi-token BF16
prefill uses a CUTLASS Tensor Core mainloop whose IteratorB reconstructs BC4 weights
with `tex2Dgather()` directly into the shared-memory pipeline. It does not retain or
write a second dense checkpoint.

## One-time GPU autotuning

Run:

```bash
texelator benchmark qwen3.8:27b
```

The command first tunes decode lookahead, then benchmarks five bitwise-equivalent
BF16 prefill tiles at 32, 128, and 512 prompt tokens. Profiles are keyed by encoded
weight hash, CUDA kernel hash, compute capability, device name, and SM count. A
profile from another GPU or build is rejected rather than silently reused.

The candidate tiles are `256x128x32`, `128x128x32`, `128x64x32`, `64x128x32`, and
`32x32x64`. On Qwen3.8-27B and RTX 5080, this changed 512-token prefill from
599.8 ms to 475.4 ms (853.6 to 1076.9 token/s) with bitwise-identical full-model
output. Results are hardware-specific and must not be copied to another GPU.

## Runtime modes

Quality-preserving BF16 is the default:

```bash
texelator run qwen3.8:27b --prefill-mode bf16
```

An experimental 512-token MLP-only path reconstructs BC4 into packed FP4 scratch,
quantizes activations, and uses native FP4 Tensor Core GEMM:

```bash
texelator run qwen3.8:27b --prefill-mode fp4
```

The FP4 mode is a speed experiment, not a quality-preserving backend. Its RTX 5080
full-model prefill result was 386.2 ms / 1325.8 token/s versus 480.3 ms / 1066.1
token/s for autotuned BF16, but the final hidden-state comparison was cosine 0.872
and relative RMS 0.507. It is disabled by default and currently activates only for
exactly 512 tokens and the Qwen3.8-27B gate/up/down projection shapes. Other shapes
fall back to the selected BF16 path.

## Verification

```bash
texelator prefill-benchmark qwen3.8:27b \
  --tokens 512 --warmup 1 --runs 3 \
  --output texelator-prefill-512.json
```

The benchmark records wall time, CUDA-event time, throughput, and correctness. The
normal runtime must use the profile produced by `texelator benchmark`.
