from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_prefill_is_separate_from_single_token_decode():
    source = (ROOT / "texelator" / "cuda" / "texelator_cuda.cu").read_text()
    assert "if (tokens >= handle->prefill_threshold)" in source
    assert "return texelator_prefill(handle, x, tokens, shape, stream);" in source
    assert "texelator_bc4_kernel<KVALUE>" in source


def test_bf16_prefill_reuses_texture_tiles_in_tensor_core_gemm():
    source = (ROOT / "texelator" / "cuda" / "texelator_cuda.cu").read_text()
    assert "BF16_PREFILL_TOKEN_TILE" in source
    assert "BF16_PREFILL_K_TILE" in source
    assert "tex2Dgather<float4>" in source
    assert "wmma::mma_sync" in source
    assert "cuda_bf16.h" in source
    assert "No full dense matrix is materialized" in source


def test_bf16_prefill_does_not_call_dense_cublas_path():
    source = (ROOT / "texelator" / "cuda" / "texelator_cuda.cu").read_text()
    dispatch = source[source.index("if (tokens >= handle->prefill_threshold)"):]
    bf16_branch = dispatch[:dispatch.index("return texelator_prefill(handle")]
    assert "texelator_bf16_prefill_kernel" in bf16_branch
    assert "cublasGemmEx" not in bf16_branch


def test_native_extension_links_cublas():
    setup = (ROOT / "setup.py").read_text()
    runtime = (ROOT / "texelator" / "runtime.py").read_text()
    assert 'libraries=["cublas"]' in setup
    assert '"-lcublas"' in runtime
