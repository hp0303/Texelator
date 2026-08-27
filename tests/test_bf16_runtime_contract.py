from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cuda_operator_accepts_bf16_decode_and_prefill():
    source = (ROOT / "texelator" / "cuda" / "texelator_cuda.cu").read_text()
    assert "texelator_bc4_bf16_kernel" in source
    assert "texelator_bf16_prefill_kernel" in source
    assert "x.scalar_type() == at::kBFloat16" in source
    assert "__float2bfloat16" in source
    assert "__bfloat162float" in source


def test_standalone_runtime_dtype_comes_from_manifest():
    source = (ROOT / "texelator" / "standalone.py").read_text()
    assert 'manifest.get("runtime_dtype", "float16")' in source
    assert '"bfloat16": torch.bfloat16' in source
    assert "CpuOffloadedEmbedding(embedding, device, runtime_dtype)" in source
