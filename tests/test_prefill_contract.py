from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_prefill_is_separate_from_single_token_decode():
    source = (ROOT / "texelator" / "cuda" / "texelator_cuda.cu").read_text()
    assert "if (tokens >= handle->prefill_threshold)" in source
    assert "return texelator_prefill(handle, x, tokens, shape, stream);" in source
    assert "texelator_bc4_kernel<KVALUE>" in source


def test_hybrid_prefill_uses_gather_and_tensor_core_gemm():
    source = (ROOT / "texelator" / "cuda" / "texelator_cuda.cu").read_text()
    assert "tex2Dgather<float4>" in source
    assert "cublasGemmEx" in source
    assert "CUBLAS_GEMM_DEFAULT_TENSOR_OP" in source


def test_native_extension_links_cublas():
    setup = (ROOT / "setup.py").read_text()
    runtime = (ROOT / "texelator" / "runtime.py").read_text()
    assert 'libraries=["cublas"]' in setup
    assert '"-lcublas"' in runtime
