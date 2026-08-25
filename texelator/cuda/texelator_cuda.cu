#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  TORCH_CHECK(error == cudaSuccess, "CUDA: ", cudaGetErrorString(error)); \
} while (0)

#define CUBLAS_CHECK(call) do { \
  cublasStatus_t status = (call); \
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", int(status)); \
} while (0)

struct Handle {
  int M, K, blocks_per_row, device, lookahead = 2;
  int prefill_threshold = 16, prefill_tile_rows = 1024;
  cudaArray_t array = nullptr;
  cudaTextureObject_t texture = 0;
  float *scales = nullptr;
  __half *bias = nullptr;
  ~Handle() {
    cudaSetDevice(device);
    if (texture) cudaDestroyTextureObject(texture);
    if (array) cudaFreeArray(array);
    if (scales) cudaFree(scales);
    if (bias) cudaFree(bias);
  }
};

__device__ __forceinline__ void consume(
    float4 values, const __half *x, int k, float scale, float &a0, float &a1) {
  a0 = fmaf(values.w * scale, __half2float(x[k]), a0);
  a1 = fmaf(values.z * scale, __half2float(x[k + 1]), a1);
  a0 = fmaf(values.x * scale, __half2float(x[k + 4]), a0);
  a1 = fmaf(values.y * scale, __half2float(x[k + 5]), a1);
}

__device__ __forceinline__ float4 issue_gather(
    cudaTextureObject_t texture, int request, int row, int bin, int qx, int qy) {
  int block = request * 8 + bin;
  return tex2Dgather<float4>(texture, block * 4 + qx + 1.f,
                             row * 4 + qy + 1.f, 0);
}

// Generic rolling request window. LOOKAHEAD is compile-time fixed so the
// compiler can keep independent TLD4 requests and their registers visible.
template<int LOOKAHEAD>
__global__ void texelator_bc4_kernel(cudaTextureObject_t texture, const __half *x,
                       const float *scales, const __half *bias, __half *y,
                       int M, int K, int blocks_per_row, int tokens) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * 8 + warp;
  int token = blockIdx.y;
  if (row >= M || token >= tokens) return;
  x += size_t(token) * K;
  y += size_t(token) * M;
  int bin = lane >> 2;
  int quadrant = lane & 3;
  int qx = (quadrant & 1) * 2;
  int qy = (quadrant >> 1) * 2;
  float a00 = 0, a01 = 0, a10 = 0, a11 = 0;
  float scale = scales[row];
  int requests = (blocks_per_row + 7) / 8;
  float4 pending[LOOKAHEAD];
#pragma unroll
  for (int index = 0; index < LOOKAHEAD; ++index) {
    pending[index] = index < requests
        ? issue_gather(texture, index, row, bin, qx, qy) : make_float4(0, 0, 0, 0);
  }
#pragma unroll 1
  for (int request = 0; request < requests; ++request) {
    int slot = request % LOOKAHEAD;
    float4 current = pending[slot];
    int future = request + LOOKAHEAD;
    // Source order deliberately requests the future texel before consuming
    // the current texel. SASS/resource reports are saved by the paper suite.
    if (future < requests)
      pending[slot] = issue_gather(texture, future, row, bin, qx, qy);
    int block = request * 8 + bin;
    if (block < blocks_per_row) {
      int offset = block * 16 + qy * 4 + qx;
      if (request & 1) consume(current, x, offset, scale, a10, a11);
      else consume(current, x, offset, scale, a00, a01);
    }
  }
  float accumulator = (a00 + a01) + (a10 + a11);
  for (int delta = 16; delta; delta >>= 1)
    accumulator += __shfl_down_sync(0xffffffff, accumulator, delta);
  if (lane == 0)
    y[row] = __float2half(accumulator + (bias ? __half2float(bias[row]) : 0.f));
}

// Multi-token prefill path. BC4 remains resident. Only a bounded row tile is
// reconstructed into temporary FP16, then immediately reused by cuBLAS across
// all prompt tokens. Single-token decode never enters this path.
__global__ void texelator_decode_tile(cudaTextureObject_t texture,
                       const float *scales, __half *dense, int row_start,
                       int rows, int K, int blocks_per_row) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  // One gather reconstructs a 2x2 quadrant: four adjacent logical weights.
  // Four quadrants therefore cover one 16-value BC4 block with four TLD4s.
  size_t groups_per_row = size_t(blocks_per_row) * 4;
  size_t count = size_t(rows) * groups_per_row;
  if (index >= count) return;
  int local_row = index / groups_per_row;
  int group = index - size_t(local_row) * groups_per_row;
  int block = group >> 2;
  int quadrant = group & 3;
  int qx = (quadrant & 1) * 2;
  int qy = (quadrant >> 1) * 2;
  int row = row_start + local_row;
  float4 values = tex2Dgather<float4>(
      texture, block * 4 + qx + 1.f, row * 4 + qy + 1.f, 0);
  float scale = scales[row];
  size_t offset = size_t(local_row) * K + block * 16 + qy * 4 + qx;
  dense[offset] = __float2half(values.w * scale);
  dense[offset + 1] = __float2half(values.z * scale);
  dense[offset + 4] = __float2half(values.x * scale);
  dense[offset + 5] = __float2half(values.y * scale);
}

__global__ void texelator_add_bias(__half *output, const __half *bias,
                                   size_t values, int M) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < values)
    output[index] = __hadd(output[index], bias[index % M]);
}

at::Tensor texelator_prefill(Handle *handle, at::Tensor x, int tokens,
                             const std::vector<int64_t> &shape,
                             cudaStream_t stream) {
  auto y = at::empty(shape, x.options());
  size_t free_bytes = 0, total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t row_bytes = size_t(handle->K) * sizeof(__half);
  size_t budget = free_bytes > (64ull << 20) ? (free_bytes - (64ull << 20)) / 4 : 0;
  int tile_rows = int(std::min<size_t>(handle->prefill_tile_rows,
      row_bytes ? budget / row_bytes : 0));
  tile_rows = std::min(tile_rows, handle->M);
  if (tile_rows >= 16) tile_rows = (tile_rows / 16) * 16;
  TORCH_CHECK(tile_rows >= 1,
      "insufficient CUDA memory for Texelator prefill workspace; free bytes=",
      free_bytes, ", bytes per dense row=", row_bytes);
  auto dense = at::empty({tile_rows, handle->K}, x.options());
  cublasHandle_t blas = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_CHECK(cublasSetStream(blas, stream));
  const float alpha = 1.f, beta = 0.f;
  auto *x_ptr = static_cast<const __half *>(x.data_ptr());
  auto *y_ptr = static_cast<__half *>(y.data_ptr());
  auto *dense_ptr = static_cast<__half *>(dense.data_ptr());
  for (int row_start = 0; row_start < handle->M; row_start += tile_rows) {
    int rows = std::min(tile_rows, handle->M - row_start);
    size_t gather_groups = size_t(rows) * handle->blocks_per_row * 4;
    texelator_decode_tile<<<(gather_groups + 255) / 256, 256, 0, stream>>>(
        handle->texture, handle->scales, dense_ptr, row_start, rows,
        handle->K, handle->blocks_per_row);
    CUDA_CHECK(cudaGetLastError());
    // Row-major Y = X W^T is column-major Y^T = W X^T. ldc remains the
    // complete output width, so every row tile is written directly into Y.
    CUBLAS_CHECK(cublasGemmEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N,
        rows, tokens, handle->K,
        &alpha,
        dense_ptr, CUDA_R_16F, handle->K,
        x_ptr, CUDA_R_16F, handle->K,
        &beta,
        y_ptr + row_start, CUDA_R_16F, handle->M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (handle->bias) {
    size_t values = size_t(tokens) * handle->M;
    texelator_add_bias<<<(values + 255) / 256, 256, 0, stream>>>(
        y_ptr, handle->bias, values, handle->M);
    CUDA_CHECK(cudaGetLastError());
  }
  return y;
}

uint64_t texelator_pack_encoded(at::Tensor blocks, at::Tensor scales,
                             c10::optional<at::Tensor> bias) {
  TORCH_CHECK(!blocks.is_cuda() && blocks.scalar_type() == at::kByte && blocks.is_contiguous(),
              "blocks must be contiguous CPU uint8");
  TORCH_CHECK(!scales.is_cuda() && scales.scalar_type() == at::kFloat && scales.is_contiguous(),
              "scales must be contiguous CPU float32");
  int M = scales.numel();
  TORCH_CHECK(M > 0 && blocks.numel() % (size_t(M) * 8) == 0, "invalid encoded size");
  int blocks_per_row = blocks.numel() / (size_t(M) * 8);
  int K = blocks_per_row * 16;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  auto *handle = new Handle{M, K, blocks_per_row, device, 2, 16, 1024};
  cudaChannelFormatDesc description = cudaCreateChannelDesc<uint2>();
  CUDA_CHECK(cudaMallocArray(&handle->array, &description, blocks_per_row, M));
  CUDA_CHECK(cudaMemcpy2DToArray(handle->array, 0, 0, blocks.data_ptr(), size_t(blocks_per_row) * 8,
                                size_t(blocks_per_row) * 8, M, cudaMemcpyHostToDevice));
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = handle->array;
  cudaResourceViewDesc view{};
  view.format = cudaResViewFormatSignedBlockCompressed4;
  view.width = blocks_per_row * 4;
  view.height = M * 4;
  cudaTextureDesc texture{};
  texture.addressMode[0] = cudaAddressModeClamp;
  texture.addressMode[1] = cudaAddressModeClamp;
  texture.filterMode = cudaFilterModePoint;
  texture.readMode = cudaReadModeElementType;
  CUDA_CHECK(cudaCreateTextureObject(&handle->texture, &resource, &texture, &view));
  CUDA_CHECK(cudaMalloc(&handle->scales, size_t(M) * 4));
  CUDA_CHECK(cudaMemcpy(handle->scales, scales.data_ptr(), size_t(M) * 4, cudaMemcpyHostToDevice));
  if (bias.has_value() && bias->defined() && bias->numel()) {
    auto contiguous = bias->contiguous();
    TORCH_CHECK(contiguous.is_cuda() && contiguous.scalar_type() == at::kHalf && contiguous.numel() == M,
                "bias must be CUDA FP16 [M]");
    CUDA_CHECK(cudaMalloc(&handle->bias, size_t(M) * 2));
    CUDA_CHECK(cudaMemcpy(handle->bias, contiguous.data_ptr(), size_t(M) * 2, cudaMemcpyDeviceToDevice));
  }
  return reinterpret_cast<uint64_t>(handle);
}

at::Tensor texelator_linear(uint64_t value, at::Tensor x) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous(),
              "input must be contiguous CUDA FP16");
  TORCH_CHECK(x.size(-1) == handle->K && x.get_device() == handle->device, "input shape/device mismatch");
  int tokens = x.numel() / handle->K;
  auto shape = x.sizes().vec();
  shape.back() = handle->M;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(handle->device);
  if (tokens >= handle->prefill_threshold)
    return texelator_prefill(handle, x, tokens, shape, stream);
  TORCH_CHECK(tokens <= 65535, "token grid exceeds CUDA grid.y limit");
  auto y = at::empty(shape, x.options());
  dim3 grid((handle->M + 7) / 8, tokens);
#define LAUNCH_K(KVALUE) texelator_bc4_kernel<KVALUE><<<grid, 256, 0, stream>>>(handle->texture, \
      static_cast<const __half *>(x.data_ptr()), handle->scales, handle->bias, \
      static_cast<__half *>(y.data_ptr()), handle->M, handle->K, handle->blocks_per_row, tokens)
  switch (handle->lookahead) {
    case 1: LAUNCH_K(1); break;
    case 2: LAUNCH_K(2); break;
    case 3: LAUNCH_K(3); break;
    case 4: LAUNCH_K(4); break;
    case 6: LAUNCH_K(6); break;
    case 8: LAUNCH_K(8); break;
    default: TORCH_CHECK(false, "unsupported lookahead");
  }
#undef LAUNCH_K
  CUDA_CHECK(cudaGetLastError());
  return y;
}

void texelator_set_lookahead(uint64_t value, int lookahead) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  TORCH_CHECK(lookahead == 1 || lookahead == 2 || lookahead == 3 ||
              lookahead == 4 || lookahead == 6 || lookahead == 8,
              "lookahead must be one of 1,2,3,4,6,8");
  handle->lookahead = lookahead;
}

void texelator_set_prefill(uint64_t value, int threshold, int tile_rows) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  TORCH_CHECK(threshold >= 1, "prefill threshold must be positive");
  TORCH_CHECK(tile_rows >= 16 && tile_rows % 16 == 0,
              "prefill tile rows must be a positive multiple of 16");
  handle->prefill_threshold = threshold;
  handle->prefill_tile_rows = tile_rows;
}

void texelator_free(uint64_t value) { delete reinterpret_cast<Handle *>(value); }

struct ProbeBlock { uint8_t bytes[8]; };

__global__ void texelator_dump_palette(cudaTextureObject_t texture, float *output, int pairs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pairs) return;
  int x = index % 255, y = index / 255;
  for (int selector = 0; selector < 8; ++selector)
    output[index * 8 + selector] = tex2D<float>(
        texture, x * 4 + (selector & 3) + .5f, y * 4 + (selector >> 2) + .5f);
}

at::Tensor texelator_palette_probe() {
  constexpr int pairs = 255 * 255;
  std::vector<ProbeBlock> host(pairs);
  uint64_t selectors = 0;
  for (int i = 0; i < 16; ++i) selectors |= uint64_t(i & 7) << (3 * i);
  for (int y = 0; y < 255; ++y) {
    for (int x = 0; x < 255; ++x) {
      ProbeBlock &block = host[y * 255 + x];
      block.bytes[0] = static_cast<uint8_t>(static_cast<int8_t>(x - 127));
      block.bytes[1] = static_cast<uint8_t>(static_cast<int8_t>(y - 127));
      for (int byte = 0; byte < 6; ++byte)
        block.bytes[byte + 2] = (selectors >> (8 * byte)) & 255;
    }
  }
  cudaArray_t array = nullptr;
  cudaTextureObject_t texture = 0;
  float *device_output = nullptr;
  cudaChannelFormatDesc description = cudaCreateChannelDesc<uint2>();
  CUDA_CHECK(cudaMallocArray(&array, &description, 255, 255));
  CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, host.data(), 255 * 8, 255 * 8, 255,
                                cudaMemcpyHostToDevice));
  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = array;
  cudaResourceViewDesc view{};
  view.format = cudaResViewFormatSignedBlockCompressed4;
  view.width = 1020;
  view.height = 1020;
  cudaTextureDesc texture_description{};
  texture_description.addressMode[0] = cudaAddressModeClamp;
  texture_description.addressMode[1] = cudaAddressModeClamp;
  texture_description.filterMode = cudaFilterModePoint;
  texture_description.readMode = cudaReadModeElementType;
  CUDA_CHECK(cudaCreateTextureObject(&texture, &resource, &texture_description, &view));
  CUDA_CHECK(cudaMalloc(&device_output, size_t(pairs) * 8 * sizeof(float)));
  texelator_dump_palette<<<(pairs + 255) / 256, 256>>>(texture, device_output, pairs);
  CUDA_CHECK(cudaGetLastError());
  auto output = at::empty({pairs, 8}, at::TensorOptions().dtype(at::kFloat));
  CUDA_CHECK(cudaMemcpy(output.data_ptr(), device_output, size_t(pairs) * 8 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaDestroyTextureObject(texture));
  CUDA_CHECK(cudaFreeArray(array));
  return output;
}
