# Supported model boundary

Texelator replaces eligible `torch.nn.Linear` projections. It does not modify attention,
normalization, positional encoding, KV-cache semantics, tokenizer behavior, or the
Transformer graph.

An ordinary projection is supported when:

- its input dimension is divisible by the 16-weight BC4 block width;
- its source weight is available in floating-point form during conversion;
- the selected module is resident on CUDA during runtime installation; and
- v0.2 artifacts use FP16; BF16-calibrated artifacts may declare BF16 activations and
  outputs in their manifest.

The generic discovery path recognizes separate Q/K/V/O and gate/up/down projections.
Use `--include-regex` only after inspecting a nonstandard fused architecture. A fused
name matching a regular expression does not by itself prove that its row ordering is
compatible.

Texelator falls back by leaving unsupported modules unchanged. It fails rather than
silently converting a model when no eligible projections are found.
