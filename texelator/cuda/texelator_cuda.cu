#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include <vector>

at::Tensor texelator_cutlass_texture_bf16(
    uint64_t texture, float const *scales, void const *bias,
    at::Tensor x, int rows, int columns, int blocks_per_row,
    std::vector<int64_t> const &shape, cudaStream_t stream);
std::vector<int64_t> texelator_cutlass_texture_kernel_info();
at::Tensor texelator_cutlass_t1_bf16(
    uint64_t, float const *, void const *, at::Tensor, int, int, int,
    std::vector<int64_t> const &, cudaStream_t);
at::Tensor texelator_cutlass_t2_bf16(
    uint64_t, float const *, void const *, at::Tensor, int, int, int,
    std::vector<int64_t> const &, cudaStream_t);
at::Tensor texelator_cutlass_t3_bf16(
    uint64_t, float const *, void const *, at::Tensor, int, int, int,
    std::vector<int64_t> const &, cudaStream_t);
std::vector<int64_t> texelator_cutlass_t1_kernel_info();
std::vector<int64_t> texelator_cutlass_t2_kernel_info();
std::vector<int64_t> texelator_cutlass_t3_kernel_info();
at::Tensor texelator_cutlass_texture_smalln_bf16(
    uint64_t, float const *, void const *, at::Tensor, int, int, int,
    std::vector<int64_t> const &, cudaStream_t);
std::vector<int64_t> texelator_cutlass_texture_smalln_kernel_info();

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
  // 0: original direct texture/WMMA.
  // 1: texture decode to BF16 global scratch + cuBLAS.
  // 2: large-tile direct texture producer + Tensor Core MMA (no global scratch).
  // 3: M64/N256/K64 double-buffered texture + Tensor Core pipeline.
  // 4: graphics-style M64/N128/K64 pipeline, 48 KiB shared, two CTA target.
  // 5: CUTLASS MmaPipelined with a BC4 Texture Unit IteratorB producer.
  // 6: T1 M128/N128/K32, warp M64/N32/K32.
  // 7: T2 M128/N64/K32, warp M64/N32/K32.
  // 8: T3 M64/N128/K32, warp M32/N32/K32.
  int prefill_backend = 0;
  bool pipelined_supported = false, graphics_supported = false;
  cudaArray_t array = nullptr;
  cudaTextureObject_t texture = 0;
  float *scales = nullptr;
  void *bias = nullptr;
  at::ScalarType bias_type = at::kHalf;
  ~Handle() {
    cudaSetDevice(device);
    if (texture) cudaDestroyTextureObject(texture);
    if (array) cudaFreeArray(array);
    if (scales) cudaFree(scales);
    if (bias) cudaFree(bias);
  }
};

using namespace nvcuda;

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

__device__ __forceinline__ void consume_bf16(
    float4 values, const __nv_bfloat16 *x, int k, float scale,
    float &a0, float &a1) {
  float w0 = __bfloat162float(__float2bfloat16(values.w * scale));
  float w1 = __bfloat162float(__float2bfloat16(values.z * scale));
  float w4 = __bfloat162float(__float2bfloat16(values.x * scale));
  float w5 = __bfloat162float(__float2bfloat16(values.y * scale));
  a0 = fmaf(w0, __bfloat162float(x[k]), a0);
  a1 = fmaf(w1, __bfloat162float(x[k + 1]), a1);
  a0 = fmaf(w4, __bfloat162float(x[k + 4]), a0);
  a1 = fmaf(w5, __bfloat162float(x[k + 5]), a1);
}

template<int LOOKAHEAD>
__global__ void texelator_bc4_bf16_kernel(
    cudaTextureObject_t texture, const __nv_bfloat16 *x,
    const float *scales, const __nv_bfloat16 *bias, __nv_bfloat16 *y,
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
    if (future < requests)
      pending[slot] = issue_gather(texture, future, row, bin, qx, qy);
    int block = request * 8 + bin;
    if (block < blocks_per_row) {
      int offset = block * 16 + qy * 4 + qx;
      if (request & 1) consume_bf16(current, x, offset, scale, a10, a11);
      else consume_bf16(current, x, offset, scale, a00, a01);
    }
  }
  float accumulator = (a00 + a01) + (a10 + a11);
  for (int delta = 16; delta; delta >>= 1)
    accumulator += __shfl_down_sync(0xffffffff, accumulator, delta);
  if (lane == 0)
    y[row] = __float2bfloat16(
        accumulator + (bias ? __bfloat162float(bias[row]) : 0.f));
}

// BF16 prefill: a CTA reconstructs a 16x64 BC4 weight tile once and sixteen
// warps reuse it for 256 prompt tokens. No full dense matrix is materialized in
// global memory. Partial final token tiles use one shared, zero-padded B tile.
constexpr int BF16_PREFILL_WARPS = 16;
constexpr int BF16_PREFILL_THREADS = BF16_PREFILL_WARPS * 32;
constexpr int BF16_PREFILL_TOKEN_TILE = BF16_PREFILL_WARPS * 16;
constexpr int BF16_PREFILL_K_TILE = 64;

struct Bf16PrefillShared {
  __nv_bfloat16 weight[16 * BF16_PREFILL_K_TILE];
  __nv_bfloat16 boundary_b[16 * 16];
  float output[BF16_PREFILL_WARPS][16 * 16];
};

__global__ __launch_bounds__(BF16_PREFILL_THREADS, 1)
void texelator_bf16_prefill_kernel(
    cudaTextureObject_t texture, const __nv_bfloat16 *x,
    const float *scales, const __nv_bfloat16 *bias, __nv_bfloat16 *y,
    int M, int K, int blocks_per_row, int tokens) {
  __shared__ Bf16PrefillShared shared;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int row0 = blockIdx.x * 16;
  int token0 = blockIdx.y * BF16_PREFILL_TOKEN_TILE + warp * 16;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int k0 = 0; k0 < K; k0 += BF16_PREFILL_K_TILE) {
    if (tid < 256) {
      int local_row = tid >> 4;
      int group = tid & 15;
      int block_in_tile = group >> 2;
      int quadrant = group & 3;
      int qx = (quadrant & 1) * 2;
      int qy = (quadrant >> 1) * 2;
      int row = row0 + local_row;
      int block = k0 / 16 + block_in_tile;
      float4 values = make_float4(0, 0, 0, 0);
      float scale = 0.0f;
      if (row < M && block < blocks_per_row) {
        values = tex2Dgather<float4>(
            texture, block * 4 + qx + 1.f, row * 4 + qy + 1.f, 0);
        scale = scales[row];
      }
      int offset = local_row * BF16_PREFILL_K_TILE + block_in_tile * 16 + qy * 4 + qx;
      shared.weight[offset] = __float2bfloat16(values.w * scale);
      shared.weight[offset + 1] = __float2bfloat16(values.z * scale);
      shared.weight[offset + 4] = __float2bfloat16(values.x * scale);
      shared.weight[offset + 5] = __float2bfloat16(values.y * scale);
    }
    __syncthreads();

#pragma unroll
    for (int sub = 0; sub < BF16_PREFILL_K_TILE; sub += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16,
                     __nv_bfloat16, wmma::row_major> weight_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16,
                     __nv_bfloat16, wmma::col_major> activation_fragment;
      wmma::load_matrix_sync(
          weight_fragment, shared.weight + sub, BF16_PREFILL_K_TILE);
      if (token0 + 15 < tokens) {
        wmma::load_matrix_sync(
            activation_fragment, x + size_t(token0) * K + k0 + sub, K);
      } else if (token0 < tokens) {
        for (int index = lane; index < 256; index += 32) {
          int column = index >> 4;
          int inner = index & 15;
          int token = token0 + column;
          shared.boundary_b[index] = token < tokens
              ? x[size_t(token) * K + k0 + sub + inner]
              : __float2bfloat16(0.0f);
        }
        __syncwarp();
        wmma::load_matrix_sync(
            activation_fragment, shared.boundary_b, 16);
      }
      if (token0 < tokens)
        wmma::mma_sync(accumulator, weight_fragment, activation_fragment, accumulator);
    }
    __syncthreads();
  }

  if (token0 < tokens) {
    wmma::store_matrix_sync(
        shared.output[warp], accumulator, 16, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < 256; index += 32) {
      int local_row = index >> 4;
      int column = index & 15;
      int row = row0 + local_row;
      int token = token0 + column;
      if (row < M && token < tokens) {
        float value = shared.output[warp][index];
        if (bias) value += __bfloat162float(bias[row]);
        y[size_t(token) * M + row] = __float2bfloat16(value);
      }
    }
  }
}

// Large-prompt direct path.  The arithmetic consumer remains a regular BF16
// Tensor Core GEMM tile; only its A-tile producer is Texelator-specific.  Each
// of the 1024 threads issues exactly one TLD4 per K tile, reconstructing a
// 16x256 hardware-exact BC4 weight tile into shared memory.  All 32 warps then
// reuse that tile for the complete 512-token prompt.  Compared with the
// original 16-warp/64-K kernel this removes the second row-tile CTA, cuts CTA
// barriers by four, and never writes a dense BF16 weight matrix to global
// memory.
constexpr int BF16_DIRECT_WARPS = 32;
constexpr int BF16_DIRECT_THREADS = BF16_DIRECT_WARPS * 32;
constexpr int BF16_DIRECT_TOKENS = BF16_DIRECT_WARPS * 16;
constexpr int BF16_DIRECT_K_TILE = 256;
constexpr int BF16_DIRECT_GROUPS_PER_ROW = (BF16_DIRECT_K_TILE / 16) * 4;

struct Bf16DirectShared {
  __nv_bfloat16 weight[16 * BF16_DIRECT_K_TILE];
  float output[BF16_DIRECT_WARPS][16 * 16];
};

__global__ __launch_bounds__(BF16_DIRECT_THREADS, 1)
void texelator_bf16_direct_512_kernel(
    cudaTextureObject_t texture, const __nv_bfloat16 *x,
    const float *scales, const __nv_bfloat16 *bias, __nv_bfloat16 *y,
    int M, int K, int blocks_per_row) {
  __shared__ Bf16DirectShared shared;
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  int row0 = blockIdx.x * 16;
  int token0 = warp * 16;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int k0 = 0; k0 < K; k0 += BF16_DIRECT_K_TILE) {
    // 16 rows * 64 gather groups/row = 1024 independent texture requests.
    int local_row = tid / BF16_DIRECT_GROUPS_PER_ROW;
    int group = tid - local_row * BF16_DIRECT_GROUPS_PER_ROW;
    int block_in_tile = group >> 2;
    int quadrant = group & 3;
    int qx = (quadrant & 1) * 2;
    int qy = (quadrant >> 1) * 2;
    int row = row0 + local_row;
    int block = k0 / 16 + block_in_tile;
    float4 values = make_float4(0, 0, 0, 0);
    float scale = 0.0f;
    if (row < M && block < blocks_per_row) {
      values = tex2Dgather<float4>(
          texture, block * 4 + qx + 1.f, row * 4 + qy + 1.f, 0);
      scale = scales[row];
    }
    int offset = local_row * BF16_DIRECT_K_TILE +
                 block_in_tile * 16 + qy * 4 + qx;
    shared.weight[offset] = __float2bfloat16(values.w * scale);
    shared.weight[offset + 1] = __float2bfloat16(values.z * scale);
    shared.weight[offset + 4] = __float2bfloat16(values.x * scale);
    shared.weight[offset + 5] = __float2bfloat16(values.y * scale);
    __syncthreads();

#pragma unroll
    for (int sub = 0; sub < BF16_DIRECT_K_TILE; sub += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16,
                     __nv_bfloat16, wmma::row_major> weight_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16,
                     __nv_bfloat16, wmma::col_major> activation_fragment;
      wmma::load_matrix_sync(
          weight_fragment, shared.weight + sub, BF16_DIRECT_K_TILE);
      wmma::load_matrix_sync(
          activation_fragment, x + size_t(token0) * K + k0 + sub, K);
      wmma::mma_sync(
          accumulator, weight_fragment, activation_fragment, accumulator);
    }
    __syncthreads();
  }

  wmma::store_matrix_sync(
      shared.output[warp], accumulator, 16, wmma::mem_row_major);
  __syncwarp();
  for (int index = lane; index < 256; index += 32) {
    int local_row = index >> 4;
    int column = index & 15;
    int row = row0 + local_row;
    int token = token0 + column;
    if (row < M) {
      float value = shared.output[warp][index];
      if (bias) value += __bfloat162float(bias[row]);
      y[size_t(token) * M + row] = __float2bfloat16(value);
    }
  }
}

at::Tensor texelator_prefill_bf16_direct_512(
    Handle *handle, at::Tensor x, const std::vector<int64_t> &shape,
    cudaStream_t stream) {
  TORCH_CHECK(handle->K % BF16_DIRECT_K_TILE == 0,
              "direct MMA prefill requires K divisible by 256");
  TORCH_CHECK(!handle->bias || handle->bias_type == at::kBFloat16,
              "direct MMA prefill requires a BF16 bias");
  auto y = at::empty(shape, x.options());
  dim3 grid((handle->M + 15) / 16);
  texelator_bf16_direct_512_kernel<<<
      grid, BF16_DIRECT_THREADS, 0, stream>>>(
      handle->texture,
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      handle->scales,
      static_cast<const __nv_bfloat16 *>(handle->bias),
      static_cast<__nv_bfloat16 *>(y.data_ptr()),
      handle->M, handle->K, handle->blocks_per_row);
  CUDA_CHECK(cudaGetLastError());
  return y;
}

// Pipelined prefill gate.  A CTA computes C[64,256], so every activation tile
// is reused by four output-row warp groups.  Two shared-memory stages hold
// hardware-exact BC4 weights and BF16 activations.  The next stage's TLD4 and
// global activation loads are issued before the current stage's MMA work; the
// returned registers are committed to the alternate shared buffer afterwards.
constexpr int BF16_PIPE_WARPS = 32;
constexpr int BF16_PIPE_THREADS = BF16_PIPE_WARPS * 32;
constexpr int BF16_PIPE_M = 64;
constexpr int BF16_PIPE_N = 256;
constexpr int BF16_PIPE_K = 64;

struct __align__(16) PipelineActivationRegisters {
  uint4 first;
  uint4 second;
};

struct __align__(16) Bf16PipelineShared {
  __nv_bfloat16 weight[2][BF16_PIPE_M * BF16_PIPE_K];
  __nv_bfloat16 activation[2][BF16_PIPE_N * BF16_PIPE_K];
};

__device__ __forceinline__ float4 pipeline_issue_weight(
    cudaTextureObject_t texture, int tid, int row0, int k0,
    int M, int blocks_per_row, const float *scales, float &scale) {
  int local_row = tid >> 4;
  int group = tid & 15;
  int block_in_tile = group >> 2;
  int quadrant = group & 3;
  int qx = (quadrant & 1) * 2;
  int qy = (quadrant >> 1) * 2;
  int row = row0 + local_row;
  int block = k0 / 16 + block_in_tile;
  if (row < M && block < blocks_per_row) {
    scale = scales[row];
    return tex2Dgather<float4>(
        texture, block * 4 + qx + 1.f, row * 4 + qy + 1.f, 0);
  }
  scale = 0.0f;
  return make_float4(0, 0, 0, 0);
}

__device__ __forceinline__ void pipeline_store_weight(
    __nv_bfloat16 *destination, int tid, float4 values, float scale) {
  int local_row = tid >> 4;
  int group = tid & 15;
  int block_in_tile = group >> 2;
  int quadrant = group & 3;
  int qx = (quadrant & 1) * 2;
  int qy = (quadrant >> 1) * 2;
  int offset = local_row * BF16_PIPE_K +
               block_in_tile * 16 + qy * 4 + qx;
  destination[offset] = __float2bfloat16(values.w * scale);
  destination[offset + 1] = __float2bfloat16(values.z * scale);
  destination[offset + 4] = __float2bfloat16(values.x * scale);
  destination[offset + 5] = __float2bfloat16(values.y * scale);
}

__device__ __forceinline__ PipelineActivationRegisters pipeline_issue_activation(
    const __nv_bfloat16 *x, int tid, int token0, int k0, int K) {
  int linear = tid * 16;
  int local_token = linear / BF16_PIPE_K;
  int inner = linear - local_token * BF16_PIPE_K;
  const uint4 *source = reinterpret_cast<const uint4 *>(
      x + size_t(token0 + local_token) * K + k0 + inner);
  PipelineActivationRegisters value;
  value.first = source[0];
  value.second = source[1];
  return value;
}

__device__ __forceinline__ void pipeline_store_activation(
    __nv_bfloat16 *destination, int tid,
    const PipelineActivationRegisters &value) {
  uint4 *target = reinterpret_cast<uint4 *>(destination + tid * 16);
  target[0] = value.first;
  target[1] = value.second;
}

__global__ __launch_bounds__(BF16_PIPE_THREADS, 1)
void texelator_bf16_pipelined_512_kernel(
    cudaTextureObject_t texture, const __nv_bfloat16 *x,
    const float *scales, __nv_bfloat16 *y,
    int M, int K, int blocks_per_row) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto &shared = *reinterpret_cast<Bf16PipelineShared *>(storage);
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int warp_m = warp >> 3;
  int warp_n = warp & 7;
  int row0 = blockIdx.x * BF16_PIPE_M;
  int token0 = blockIdx.y * BF16_PIPE_N;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
  wmma::fill_fragment(accumulator0, 0.0f);
  wmma::fill_fragment(accumulator1, 0.0f);

  float initial_scale;
  float4 initial_weight = pipeline_issue_weight(
      texture, tid, row0, 0, M, blocks_per_row, scales, initial_scale);
  PipelineActivationRegisters initial_activation =
      pipeline_issue_activation(x, tid, token0, 0, K);
  pipeline_store_weight(shared.weight[0], tid, initial_weight, initial_scale);
  pipeline_store_activation(
      shared.activation[0], tid, initial_activation);
  __syncthreads();

  int current = 0;
  for (int k0 = 0; k0 < K; k0 += BF16_PIPE_K) {
    int next_k = k0 + BF16_PIPE_K;
    float next_scale = 0.0f;
    float4 next_weight = make_float4(0, 0, 0, 0);
    PipelineActivationRegisters next_activation{};
    if (next_k < K) {
      // These independent memory requests are intentionally issued before the
      // current stage is consumed, giving TEX/LDG latency useful MMA work to
      // overlap with. ptxas/SASS output is retained in the run log.
      next_weight = pipeline_issue_weight(
          texture, tid, row0, next_k, M, blocks_per_row,
          scales, next_scale);
      next_activation = pipeline_issue_activation(
          x, tid, token0, next_k, K);
    }

#pragma unroll
    for (int sub = 0; sub < BF16_PIPE_K; sub += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16,
                     __nv_bfloat16, wmma::row_major> weight_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16,
                     __nv_bfloat16, wmma::col_major> activation_fragment0;
      wmma::fragment<wmma::matrix_b, 16, 16, 16,
                     __nv_bfloat16, wmma::col_major> activation_fragment1;
      wmma::load_matrix_sync(
          weight_fragment,
          shared.weight[current] + warp_m * 16 * BF16_PIPE_K + sub,
          BF16_PIPE_K);
      wmma::load_matrix_sync(
          activation_fragment0,
          shared.activation[current] + warp_n * 16 * BF16_PIPE_K + sub,
          BF16_PIPE_K);
      wmma::load_matrix_sync(
          activation_fragment1,
          shared.activation[current] +
              (warp_n * 16 + 128) * BF16_PIPE_K + sub,
          BF16_PIPE_K);
      wmma::mma_sync(
          accumulator0, weight_fragment, activation_fragment0, accumulator0);
      wmma::mma_sync(
          accumulator1, weight_fragment, activation_fragment1, accumulator1);
    }

    if (next_k < K) {
      int next = current ^ 1;
      pipeline_store_weight(
          shared.weight[next], tid, next_weight, next_scale);
      pipeline_store_activation(
          shared.activation[next], tid, next_activation);
      __syncthreads();
      current = next;
    }
  }

  // FP32 WMMA accumulators cannot be stored directly to BF16. The pipeline
  // stages are dead now, so reuse their 80 KiB allocation as per-warp FP32
  // conversion scratch instead of reserving another shared-memory region.
  __syncthreads();
  float *output_scratch = reinterpret_cast<float *>(storage);
  float *warp_output = output_scratch + warp * 512;
  wmma::store_matrix_sync(
      warp_output, accumulator0, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(
      warp_output + 256, accumulator1, 16, wmma::mem_row_major);
  __syncwarp();
  for (int index = (threadIdx.x & 31); index < 256; index += 32) {
    int local_row = index >> 4;
    int column = index & 15;
    int output_row = row0 + warp_m * 16 + local_row;
    int output_token0 = token0 + warp_n * 16 + column;
    int output_token1 = output_token0 + 128;
    if (output_row < M) {
      y[size_t(output_token0) * M + output_row] =
          __float2bfloat16(warp_output[index]);
      y[size_t(output_token1) * M + output_row] =
          __float2bfloat16(warp_output[256 + index]);
    }
  }
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

// Decode each hardware-exact BC4 texel directly into a temporary BF16 weight
// matrix. The scratch buffer is reused immediately by a Tensor Core GEMM and
// never becomes part of the persistent model representation.
__global__ void texelator_decode_tile_bf16(
    cudaTextureObject_t texture, const float *scales,
    __nv_bfloat16 *dense, int row_start, int rows, int K,
    int blocks_per_row) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
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
  dense[offset] = __float2bfloat16(values.w * scale);
  dense[offset + 1] = __float2bfloat16(values.z * scale);
  dense[offset + 4] = __float2bfloat16(values.x * scale);
  dense[offset + 5] = __float2bfloat16(values.y * scale);
}

__global__ void texelator_add_bias_bf16(
    __nv_bfloat16 *output, const __nv_bfloat16 *bias,
    size_t values, int M) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < values) {
    float value = __bfloat162float(output[index]);
    value += __bfloat162float(bias[index % M]);
    output[index] = __float2bfloat16(value);
  }
}

at::Tensor texelator_prefill_bf16_pipelined_512(
    Handle *handle, at::Tensor x, const std::vector<int64_t> &shape,
    cudaStream_t stream) {
  TORCH_CHECK(handle->pipelined_supported,
              "pipelined prefill requires 80 KiB opt-in shared memory");
  TORCH_CHECK(handle->M % 16 == 0,
              "pipelined prefill requires M divisible by 16");
  TORCH_CHECK(handle->K % BF16_PIPE_K == 0,
              "pipelined prefill requires K divisible by 64");
  auto y = at::empty(shape, x.options());
  dim3 grid((handle->M + BF16_PIPE_M - 1) / BF16_PIPE_M,
            512 / BF16_PIPE_N);
  texelator_bf16_pipelined_512_kernel<<<
      grid, BF16_PIPE_THREADS, sizeof(Bf16PipelineShared), stream>>>(
      handle->texture,
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      handle->scales,
      static_cast<__nv_bfloat16 *>(y.data_ptr()),
      handle->M, handle->K, handle->blocks_per_row);
  CUDA_CHECK(cudaGetLastError());
  if (handle->bias) {
    TORCH_CHECK(handle->bias_type == at::kBFloat16,
                "pipelined prefill requires a BF16 bias");
    size_t values = size_t(512) * handle->M;
    texelator_add_bias_bf16<<<(values + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(y.data_ptr()),
        static_cast<const __nv_bfloat16 *>(handle->bias),
        values, handle->M);
    CUDA_CHECK(cudaGetLastError());
  }
  return y;
}

// Graphics-style tile-local consumer.  The 80-KiB pipeline above can keep
// only one CTA resident and assigns 32 warps to a block.  This variant halves
// the N tile to 128, uses 16 warps, and needs exactly 48 KiB for two stages.
// Each thread issues two independent TLD4 requests before current-stage MMA,
// then commits both results to the alternate shared-memory tile.  The BC4
// weight is never materialized in global memory.
constexpr int BF16_GRAPHICS_WARPS = 16;
constexpr int BF16_GRAPHICS_THREADS = BF16_GRAPHICS_WARPS * 32;
constexpr int BF16_GRAPHICS_M = 64;
constexpr int BF16_GRAPHICS_N = 128;
constexpr int BF16_GRAPHICS_K = 64;

struct __align__(16) Bf16GraphicsShared {
  __nv_bfloat16 weight[2][BF16_GRAPHICS_M * BF16_GRAPHICS_K];
  __nv_bfloat16 activation[2][BF16_GRAPHICS_N * BF16_GRAPHICS_K];
};

struct GraphicsWeightRegisters {
  float4 first;
  float4 second;
  float first_scale;
  float second_scale;
};

__device__ __forceinline__ float4 graphics_issue_weight_group(
    cudaTextureObject_t texture, int linear_group, int row0, int k0,
    int M, int blocks_per_row, const float *scales, float &scale) {
  int local_row = linear_group >> 4;
  int group = linear_group & 15;
  int block_in_tile = group >> 2;
  int quadrant = group & 3;
  int qx = (quadrant & 1) * 2;
  int qy = (quadrant >> 1) * 2;
  int row = row0 + local_row;
  int block = k0 / 16 + block_in_tile;
  if (row < M && block < blocks_per_row) {
    scale = scales[row];
    return tex2Dgather<float4>(
        texture, block * 4 + qx + 1.f, row * 4 + qy + 1.f, 0);
  }
  scale = 0.0f;
  return make_float4(0, 0, 0, 0);
}

__device__ __forceinline__ GraphicsWeightRegisters graphics_issue_weights(
    cudaTextureObject_t texture, int tid, int row0, int k0,
    int M, int blocks_per_row, const float *scales) {
  GraphicsWeightRegisters result;
  // Source order is deliberate: both independent TLD4 requests precede any
  // shared-memory stores or current-stage WMMA consumption.
  result.first = graphics_issue_weight_group(
      texture, tid, row0, k0, M, blocks_per_row,
      scales, result.first_scale);
  result.second = graphics_issue_weight_group(
      texture, tid + BF16_GRAPHICS_THREADS, row0, k0, M, blocks_per_row,
      scales, result.second_scale);
  return result;
}

__device__ __forceinline__ void graphics_store_weight_group(
    __nv_bfloat16 *destination, int linear_group,
    float4 values, float scale) {
  int local_row = linear_group >> 4;
  int group = linear_group & 15;
  int block_in_tile = group >> 2;
  int quadrant = group & 3;
  int qx = (quadrant & 1) * 2;
  int qy = (quadrant >> 1) * 2;
  int offset = local_row * BF16_GRAPHICS_K +
               block_in_tile * 16 + qy * 4 + qx;
  destination[offset] = __float2bfloat16(values.w * scale);
  destination[offset + 1] = __float2bfloat16(values.z * scale);
  destination[offset + 4] = __float2bfloat16(values.x * scale);
  destination[offset + 5] = __float2bfloat16(values.y * scale);
}

__device__ __forceinline__ void graphics_store_weights(
    __nv_bfloat16 *destination, int tid,
    const GraphicsWeightRegisters &values) {
  graphics_store_weight_group(
      destination, tid, values.first, values.first_scale);
  graphics_store_weight_group(
      destination, tid + BF16_GRAPHICS_THREADS,
      values.second, values.second_scale);
}

__device__ __forceinline__ PipelineActivationRegisters graphics_issue_activation(
    const __nv_bfloat16 *x, int tid, int token0, int k0, int K) {
  int linear = tid * 16;
  int local_token = linear / BF16_GRAPHICS_K;
  int inner = linear - local_token * BF16_GRAPHICS_K;
  const uint4 *source = reinterpret_cast<const uint4 *>(
      x + size_t(token0 + local_token) * K + k0 + inner);
  PipelineActivationRegisters value;
  value.first = source[0];
  value.second = source[1];
  return value;
}

__global__ __launch_bounds__(BF16_GRAPHICS_THREADS, 2)
void texelator_bf16_graphics_512_kernel(
    cudaTextureObject_t texture, const __nv_bfloat16 *x,
    const float *scales, __nv_bfloat16 *y,
    int M, int K, int blocks_per_row) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto &shared = *reinterpret_cast<Bf16GraphicsShared *>(storage);
  int tid = threadIdx.x;
  int warp = tid >> 5;
  int warp_m = warp >> 2;
  int warp_n = warp & 3;
  int row0 = blockIdx.x * BF16_GRAPHICS_M;
  int token0 = blockIdx.y * BF16_GRAPHICS_N;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator0;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator1;
  wmma::fill_fragment(accumulator0, 0.0f);
  wmma::fill_fragment(accumulator1, 0.0f);

  GraphicsWeightRegisters initial_weight = graphics_issue_weights(
      texture, tid, row0, 0, M, blocks_per_row, scales);
  PipelineActivationRegisters initial_activation =
      graphics_issue_activation(x, tid, token0, 0, K);
  graphics_store_weights(shared.weight[0], tid, initial_weight);
  pipeline_store_activation(shared.activation[0], tid, initial_activation);
  __syncthreads();

  int current = 0;
  for (int k0 = 0; k0 < K; k0 += BF16_GRAPHICS_K) {
    int next_k = k0 + BF16_GRAPHICS_K;
    GraphicsWeightRegisters next_weight{};
    PipelineActivationRegisters next_activation{};
    if (next_k < K) {
      next_weight = graphics_issue_weights(
          texture, tid, row0, next_k, M, blocks_per_row, scales);
      next_activation = graphics_issue_activation(
          x, tid, token0, next_k, K);
    }

#pragma unroll
    for (int sub = 0; sub < BF16_GRAPHICS_K; sub += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16,
                     __nv_bfloat16, wmma::row_major> weight_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16,
                     __nv_bfloat16, wmma::col_major> activation_fragment0;
      wmma::fragment<wmma::matrix_b, 16, 16, 16,
                     __nv_bfloat16, wmma::col_major> activation_fragment1;
      wmma::load_matrix_sync(
          weight_fragment,
          shared.weight[current] + warp_m * 16 * BF16_GRAPHICS_K + sub,
          BF16_GRAPHICS_K);
      wmma::load_matrix_sync(
          activation_fragment0,
          shared.activation[current] + warp_n * 16 * BF16_GRAPHICS_K + sub,
          BF16_GRAPHICS_K);
      wmma::load_matrix_sync(
          activation_fragment1,
          shared.activation[current] +
              (warp_n * 16 + 64) * BF16_GRAPHICS_K + sub,
          BF16_GRAPHICS_K);
      wmma::mma_sync(
          accumulator0, weight_fragment, activation_fragment0, accumulator0);
      wmma::mma_sync(
          accumulator1, weight_fragment, activation_fragment1, accumulator1);
    }

    if (next_k < K) {
      int next = current ^ 1;
      graphics_store_weights(shared.weight[next], tid, next_weight);
      pipeline_store_activation(
          shared.activation[next], tid, next_activation);
      __syncthreads();
      current = next;
    }
  }

  __syncthreads();
  float *output_scratch = reinterpret_cast<float *>(storage);
  float *warp_output = output_scratch + warp * 512;
  wmma::store_matrix_sync(
      warp_output, accumulator0, 16, wmma::mem_row_major);
  wmma::store_matrix_sync(
      warp_output + 256, accumulator1, 16, wmma::mem_row_major);
  __syncwarp();
  for (int index = (threadIdx.x & 31); index < 256; index += 32) {
    int local_row = index >> 4;
    int column = index & 15;
    int output_row = row0 + warp_m * 16 + local_row;
    int output_token0 = token0 + warp_n * 16 + column;
    int output_token1 = output_token0 + 64;
    if (output_row < M) {
      y[size_t(output_token0) * M + output_row] =
          __float2bfloat16(warp_output[index]);
      y[size_t(output_token1) * M + output_row] =
          __float2bfloat16(warp_output[256 + index]);
    }
  }
}

at::Tensor texelator_prefill_bf16_graphics_512(
    Handle *handle, at::Tensor x, const std::vector<int64_t> &shape,
    cudaStream_t stream) {
  TORCH_CHECK(handle->M % 16 == 0,
              "graphics prefill requires M divisible by 16");
  TORCH_CHECK(handle->K % BF16_GRAPHICS_K == 0,
              "graphics prefill requires K divisible by 64");
  auto y = at::empty(shape, x.options());
  dim3 grid((handle->M + BF16_GRAPHICS_M - 1) / BF16_GRAPHICS_M,
            512 / BF16_GRAPHICS_N);
  texelator_bf16_graphics_512_kernel<<<
      grid, BF16_GRAPHICS_THREADS, sizeof(Bf16GraphicsShared), stream>>>(
      handle->texture,
      static_cast<const __nv_bfloat16 *>(x.data_ptr()),
      handle->scales,
      static_cast<__nv_bfloat16 *>(y.data_ptr()),
      handle->M, handle->K, handle->blocks_per_row);
  CUDA_CHECK(cudaGetLastError());
  if (handle->bias) {
    TORCH_CHECK(handle->bias_type == at::kBFloat16,
                "graphics prefill requires a BF16 bias");
    size_t values = size_t(512) * handle->M;
    texelator_add_bias_bf16<<<(values + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(y.data_ptr()),
        static_cast<const __nv_bfloat16 *>(handle->bias),
        values, handle->M);
    CUDA_CHECK(cudaGetLastError());
  }
  return y;
}

at::Tensor texelator_prefill_bf16_cublas(
    Handle *handle, at::Tensor x, int tokens,
    const std::vector<int64_t> &shape, cudaStream_t stream) {
  auto y = at::empty(shape, x.options());
  size_t free_bytes = 0, total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t row_bytes = size_t(handle->K) * sizeof(__nv_bfloat16);
  // Leave a conservative reserve for attention/KV temporaries. The remainder
  // permits a whole matrix for current 27B shapes, avoiding many small GEMMs.
  size_t reserve = 256ull << 20;
  size_t budget = free_bytes > reserve ? (free_bytes - reserve) / 2 : 0;
  int tile_rows = int(std::min<size_t>(
      handle->prefill_tile_rows, row_bytes ? budget / row_bytes : 0));
  tile_rows = std::min(tile_rows, handle->M);
  if (tile_rows >= 16) tile_rows = (tile_rows / 16) * 16;
  TORCH_CHECK(tile_rows >= 1,
      "insufficient CUDA memory for BF16 cuBLAS prefill workspace; free bytes=",
      free_bytes, ", bytes per dense row=", row_bytes);

  auto dense = at::empty({tile_rows, handle->K}, x.options());
  auto *dense_ptr = static_cast<__nv_bfloat16 *>(dense.data_ptr());
  auto *x_ptr = static_cast<const __nv_bfloat16 *>(x.data_ptr());
  auto *y_ptr = static_cast<__nv_bfloat16 *>(y.data_ptr());
  cublasHandle_t blas = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_CHECK(cublasSetStream(blas, stream));
  const float alpha = 1.f, beta = 0.f;

  for (int row_start = 0; row_start < handle->M; row_start += tile_rows) {
    int rows = std::min(tile_rows, handle->M - row_start);
    size_t gather_groups = size_t(rows) * handle->blocks_per_row * 4;
    texelator_decode_tile_bf16<<<
        (gather_groups + 255) / 256, 256, 0, stream>>>(
        handle->texture, handle->scales, dense_ptr, row_start, rows,
        handle->K, handle->blocks_per_row);
    CUDA_CHECK(cudaGetLastError());
    // Row-major Y = X W^T is column-major Y^T = W X^T.
    CUBLAS_CHECK(cublasGemmEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N,
        rows, tokens, handle->K,
        &alpha,
        dense_ptr, CUDA_R_16BF, handle->K,
        x_ptr, CUDA_R_16BF, handle->K,
        &beta,
        y_ptr + row_start, CUDA_R_16BF, handle->M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (handle->bias) {
    TORCH_CHECK(handle->bias_type == at::kBFloat16,
                "BF16 cuBLAS prefill requires a BF16 bias");
    size_t values = size_t(tokens) * handle->M;
    texelator_add_bias_bf16<<<(values + 255) / 256, 256, 0, stream>>>(
        y_ptr, static_cast<const __nv_bfloat16 *>(handle->bias),
        values, handle->M);
    CUDA_CHECK(cudaGetLastError());
  }
  return y;
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
    TORCH_CHECK(handle->bias_type == at::kHalf,
                "FP16 prefill requires an FP16 bias");
    size_t values = size_t(tokens) * handle->M;
    texelator_add_bias<<<(values + 255) / 256, 256, 0, stream>>>(
        y_ptr, static_cast<const __half *>(handle->bias), values, handle->M);
    CUDA_CHECK(cudaGetLastError());
  }
  return y;
}

// Map a finite FP32 value to the nearest E2M1 magnitude. The sign occupies
// bit 3 and the positive magnitudes 0..7 encode
// {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
__device__ __forceinline__ uint8_t texelator_e2m1(float value) {
  float magnitude = fabsf(value);
  uint8_t code;
  if (magnitude < 0.25f) code = 0;
  else if (magnitude < 0.75f) code = 1;
  else if (magnitude < 1.25f) code = 2;
  else if (magnitude < 1.75f) code = 3;
  else if (magnitude < 2.5f) code = 4;
  else if (magnitude < 3.5f) code = 5;
  else if (magnitude < 5.0f) code = 6;
  else code = 7;
  return code | (signbit(value) ? 8 : 0);
}

__device__ __forceinline__ size_t texelator_blocked_scale_index(
    int row, int group, int groups_padded) {
  int row_block = row >> 7;
  int row_in_block = row & 127;
  int group_block = group >> 2;
  int group_in_block = group & 3;
  int group_blocks = groups_padded >> 2;
  return (((size_t(row_block) * group_blocks + group_block) * 32 +
           (row_in_block & 31)) * 16 +
          (row_in_block >> 5) * 4 + group_in_block);
}

__device__ __forceinline__ uint8_t texelator_fp8_e4m3(float value) {
  return static_cast<uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

// Runtime activation quantization used only by the FP4 speed gate. One CUDA
// thread owns one 16-value scaling group and writes both packed E2M1 values
// and the scale in Blackwell's blocked scale-factor layout.
__global__ void texelator_activation_to_fp4_kernel(
    const __nv_bfloat16 *input, uint8_t *packed, uint8_t *blocked_scales,
    int rows, int columns, int groups, int groups_padded) {
  size_t group_index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t count = size_t(rows) * groups;
  if (group_index >= count) return;
  int row = int(group_index / groups);
  int group = int(group_index - size_t(row) * groups);
  const __nv_bfloat16 *source = input + size_t(row) * columns + group * 16;
  float values[16];
  float maximum = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] = __bfloat162float(source[i]);
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  float scale = maximum > 0.f ? maximum * (1.f / 6.f) : 1.f;
  float inverse = maximum > 0.f ? 6.f / maximum : 0.f;
  uint8_t *destination = packed + size_t(row) * (columns / 2) + group * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t low = texelator_e2m1(values[i * 2] * inverse);
    uint8_t high = texelator_e2m1(values[i * 2 + 1] * inverse);
    destination[i] = low | (high << 4);
  }
  blocked_scales[texelator_blocked_scale_index(
      row, group, groups_padded)] = texelator_fp8_e4m3(scale);
}

// Decode one actual hardware BC4 block through four independent TLD4
// requests, then immediately quantize its 16 reconstructed values into an
// NVFP4-compatible packed scratch matrix. This is deliberately staged: the
// scratch write and native FP4 GEMM are both included in the speed gate.
__global__ void texelator_bc4_to_fp4_kernel(
    cudaTextureObject_t texture, const float *row_scales,
    uint8_t *packed, uint8_t *blocked_scales,
    int rows, int columns, int blocks_per_row, int groups_padded) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t count = size_t(rows) * blocks_per_row;
  if (index >= count) return;
  int row = int(index / blocks_per_row);
  int block = int(index - size_t(row) * blocks_per_row);
  float4 q0 = tex2Dgather<float4>(texture, block * 4 + 1.f, row * 4 + 1.f, 0);
  float4 q1 = tex2Dgather<float4>(texture, block * 4 + 3.f, row * 4 + 1.f, 0);
  float4 q2 = tex2Dgather<float4>(texture, block * 4 + 1.f, row * 4 + 3.f, 0);
  float4 q3 = tex2Dgather<float4>(texture, block * 4 + 3.f, row * 4 + 3.f, 0);
  float multiplier = row_scales[row];
  float values[16] = {
      q0.w, q0.z, q1.w, q1.z, q0.x, q0.y, q1.x, q1.y,
      q2.w, q2.z, q3.w, q3.z, q2.x, q2.y, q3.x, q3.y};
  float maximum = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    values[i] *= multiplier;
    maximum = fmaxf(maximum, fabsf(values[i]));
  }
  float scale = maximum > 0.f ? maximum * (1.f / 6.f) : 1.f;
  float inverse = maximum > 0.f ? 6.f / maximum : 0.f;
  uint8_t *destination = packed + size_t(row) * (columns / 2) + block * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint8_t low = texelator_e2m1(values[i * 2] * inverse);
    uint8_t high = texelator_e2m1(values[i * 2 + 1] * inverse);
    destination[i] = low | (high << 4);
  }
  blocked_scales[texelator_blocked_scale_index(
      row, block, groups_padded)] = texelator_fp8_e4m3(scale);
}

std::vector<at::Tensor> texelator_quantize_activation_fp4(at::Tensor input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
              input.scalar_type() == at::kBFloat16 && input.dim() == 2,
              "activation must be contiguous CUDA BF16 [tokens,K]");
  int rows = int(input.size(0));
  int columns = int(input.size(1));
  TORCH_CHECK(columns % 16 == 0, "activation K must be divisible by 16");
  int groups = columns / 16;
  int groups_padded = ((groups + 3) / 4) * 4;
  int rows_padded = ((rows + 127) / 128) * 128;
  auto byte_options = input.options().dtype(at::kByte);
  auto packed = at::empty({rows, columns / 2}, byte_options);
  auto scales = at::zeros({int64_t(rows_padded) * groups_padded}, byte_options);
  size_t count = size_t(rows) * groups;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  texelator_activation_to_fp4_kernel<<<
      (count + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input.data_ptr()),
      static_cast<uint8_t *>(packed.data_ptr()),
      static_cast<uint8_t *>(scales.data_ptr()), rows, columns,
      groups, groups_padded);
  CUDA_CHECK(cudaGetLastError());
  return {packed, scales};
}

std::vector<at::Tensor> texelator_decode_weight_fp4(uint64_t value) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  CUDA_CHECK(cudaSetDevice(handle->device));
  int groups = handle->blocks_per_row;
  int groups_padded = ((groups + 3) / 4) * 4;
  int rows_padded = ((handle->M + 127) / 128) * 128;
  auto byte_options = at::TensorOptions().dtype(at::kByte).device(
      at::kCUDA, handle->device);
  auto packed = at::empty({handle->M, handle->K / 2}, byte_options);
  auto scales = at::zeros({int64_t(rows_padded) * groups_padded}, byte_options);
  size_t count = size_t(handle->M) * handle->blocks_per_row;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(handle->device);
  texelator_bc4_to_fp4_kernel<<<
      (count + 255) / 256, 256, 0, stream>>>(
      handle->texture, handle->scales,
      static_cast<uint8_t *>(packed.data_ptr()),
      static_cast<uint8_t *>(scales.data_ptr()), handle->M, handle->K,
      handle->blocks_per_row, groups_padded);
  CUDA_CHECK(cudaGetLastError());
  return {packed, scales};
}

void texelator_quantize_activation_fp4_out(
    at::Tensor input, at::Tensor packed, at::Tensor scales) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
              input.scalar_type() == at::kBFloat16 && input.dim() == 2,
              "activation must be contiguous CUDA BF16 [tokens,K]");
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() &&
              packed.scalar_type() == at::kByte,
              "packed activation must be contiguous CUDA uint8");
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() &&
              scales.scalar_type() == at::kByte,
              "activation scales must be contiguous CUDA uint8");
  int rows = int(input.size(0));
  int columns = int(input.size(1));
  TORCH_CHECK(columns % 16 == 0 && packed.size(0) == rows &&
              packed.size(1) == columns / 2,
              "invalid activation FP4 output shape");
  int groups = columns / 16;
  int groups_padded = ((groups + 3) / 4) * 4;
  int rows_padded = ((rows + 127) / 128) * 128;
  TORCH_CHECK(scales.numel() >= int64_t(rows_padded) * groups_padded,
              "activation scale output is too small");
  size_t count = size_t(rows) * groups;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  texelator_activation_to_fp4_kernel<<<
      (count + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(input.data_ptr()),
      static_cast<uint8_t *>(packed.data_ptr()),
      static_cast<uint8_t *>(scales.data_ptr()), rows, columns,
      groups, groups_padded);
  CUDA_CHECK(cudaGetLastError());
}

void texelator_decode_weight_fp4_out(
    uint64_t value, at::Tensor packed, at::Tensor scales) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() &&
              packed.scalar_type() == at::kByte &&
              packed.size(0) == handle->M && packed.size(1) == handle->K / 2,
              "invalid packed weight FP4 output");
  TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() &&
              scales.scalar_type() == at::kByte,
              "weight scales must be contiguous CUDA uint8");
  int groups_padded = ((handle->blocks_per_row + 3) / 4) * 4;
  int rows_padded = ((handle->M + 127) / 128) * 128;
  TORCH_CHECK(scales.numel() >= int64_t(rows_padded) * groups_padded,
              "weight scale output is too small");
  size_t count = size_t(handle->M) * handle->blocks_per_row;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(handle->device);
  texelator_bc4_to_fp4_kernel<<<
      (count + 255) / 256, 256, 0, stream>>>(
      handle->texture, handle->scales,
      static_cast<uint8_t *>(packed.data_ptr()),
      static_cast<uint8_t *>(scales.data_ptr()), handle->M, handle->K,
      handle->blocks_per_row, groups_padded);
  CUDA_CHECK(cudaGetLastError());
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
  auto *handle = new Handle;
  handle->M = M;
  handle->K = K;
  handle->blocks_per_row = blocks_per_row;
  handle->device = device;
  int maximum_optin_shared = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &maximum_optin_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (maximum_optin_shared >= int(sizeof(Bf16PipelineShared))) {
    cudaError_t attribute_status = cudaFuncSetAttribute(
        texelator_bf16_pipelined_512_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(sizeof(Bf16PipelineShared)));
    handle->pipelined_supported = attribute_status == cudaSuccess;
    if (attribute_status != cudaSuccess) cudaGetLastError();
  }
  if (maximum_optin_shared >= int(sizeof(Bf16GraphicsShared))) {
    cudaError_t attribute_status = cudaFuncSetAttribute(
        texelator_bf16_graphics_512_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(sizeof(Bf16GraphicsShared)));
    handle->graphics_supported = attribute_status == cudaSuccess;
    if (attribute_status != cudaSuccess) cudaGetLastError();
  }
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
    TORCH_CHECK(contiguous.is_cuda() &&
                (contiguous.scalar_type() == at::kHalf ||
                 contiguous.scalar_type() == at::kBFloat16) &&
                contiguous.numel() == M,
                "bias must be CUDA FP16 or BF16 [M]");
    CUDA_CHECK(cudaMalloc(&handle->bias, size_t(M) * 2));
    CUDA_CHECK(cudaMemcpy(handle->bias, contiguous.data_ptr(), size_t(M) * 2, cudaMemcpyDeviceToDevice));
    handle->bias_type = contiguous.scalar_type();
  }
  return reinterpret_cast<uint64_t>(handle);
}

at::Tensor texelator_linear(uint64_t value, at::Tensor x) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  TORCH_CHECK(x.is_cuda() &&
              (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
              x.is_contiguous(),
              "input must be contiguous CUDA FP16 or BF16");
  TORCH_CHECK(x.size(-1) == handle->K && x.get_device() == handle->device, "input shape/device mismatch");
  int tokens = x.numel() / handle->K;
  auto shape = x.sizes().vec();
  shape.back() = handle->M;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(handle->device);
  if (tokens >= handle->prefill_threshold) {
    if (x.scalar_type() == at::kBFloat16) {
      if (handle->prefill_backend == 9) {
        if (handle->M % 8 == 0 && handle->K % 64 == 0) {
          return texelator_cutlass_texture_smalln_bf16(
              handle->texture, handle->scales, handle->bias, x,
              handle->M, handle->K, handle->blocks_per_row, shape, stream);
        }
        return texelator_prefill_bf16_cublas(
            handle, x, tokens, shape, stream);
      }
      if (handle->prefill_backend == 5) {
        if (handle->M % 8 == 0 && handle->K % 32 == 0) {
          return texelator_cutlass_texture_bf16(
              handle->texture, handle->scales, handle->bias, x,
              handle->M, handle->K, handle->blocks_per_row, shape, stream);
        }
        return texelator_prefill_bf16_cublas(
            handle, x, tokens, shape, stream);
      }
      if (handle->prefill_backend >= 6 && handle->prefill_backend <= 8) {
        if (handle->M % 8 == 0 && handle->K % 32 == 0) {
          if (handle->prefill_backend == 6) {
            return texelator_cutlass_t1_bf16(
                handle->texture, handle->scales, handle->bias, x,
                handle->M, handle->K, handle->blocks_per_row, shape, stream);
          }
          if (handle->prefill_backend == 7) {
            return texelator_cutlass_t2_bf16(
                handle->texture, handle->scales, handle->bias, x,
                handle->M, handle->K, handle->blocks_per_row, shape, stream);
          }
          if (handle->prefill_backend == 8) {
            return texelator_cutlass_t3_bf16(
                handle->texture, handle->scales, handle->bias, x,
                handle->M, handle->K, handle->blocks_per_row, shape, stream);
          }
          TORCH_CHECK(false, "unreachable CUTLASS tile backend");
        }
        return texelator_prefill_bf16_cublas(
            handle, x, tokens, shape, stream);
      }
      if (handle->prefill_backend == 4) {
        if (tokens == 512 && handle->graphics_supported &&
            handle->M % 16 == 0 &&
            handle->K % BF16_GRAPHICS_K == 0) {
          return texelator_prefill_bf16_graphics_512(
              handle, x, shape, stream);
        }
        return texelator_prefill_bf16_cublas(
            handle, x, tokens, shape, stream);
      }
      if (handle->prefill_backend == 3) {
        if (tokens == 512 && handle->pipelined_supported &&
            handle->M % 16 == 0 &&
            handle->K % BF16_PIPE_K == 0) {
          return texelator_prefill_bf16_pipelined_512(
              handle, x, shape, stream);
        }
        return texelator_prefill_bf16_cublas(
            handle, x, tokens, shape, stream);
      }
      if (handle->prefill_backend == 2) {
        // The direct kernel is deliberately specialized to the paper's frozen
        // 512-token prompt. Other prompt lengths retain the correctness-tested
        // cuBLAS path until a production GEMM mainloop is integrated.
        if (tokens == BF16_DIRECT_TOKENS &&
            handle->K % BF16_DIRECT_K_TILE == 0) {
          return texelator_prefill_bf16_direct_512(
              handle, x, shape, stream);
        }
        return texelator_prefill_bf16_cublas(
            handle, x, tokens, shape, stream);
      }
      if (handle->prefill_backend == 1) {
        return texelator_prefill_bf16_cublas(
            handle, x, tokens, shape, stream);
      }
      TORCH_CHECK(handle->K % BF16_PREFILL_K_TILE == 0,
                  "BF16 fused prefill requires K divisible by 64");
      TORCH_CHECK(!handle->bias || handle->bias_type == at::kBFloat16,
                  "BF16 prefill requires a BF16 bias");
      auto y = at::empty(shape, x.options());
      dim3 grid((handle->M + 15) / 16,
                (tokens + BF16_PREFILL_TOKEN_TILE - 1) / BF16_PREFILL_TOKEN_TILE);
      texelator_bf16_prefill_kernel<<<grid, BF16_PREFILL_THREADS, 0, stream>>>(
          handle->texture,
          static_cast<const __nv_bfloat16 *>(x.data_ptr()),
          handle->scales,
          static_cast<const __nv_bfloat16 *>(handle->bias),
          static_cast<__nv_bfloat16 *>(y.data_ptr()),
          handle->M, handle->K, handle->blocks_per_row, tokens);
      CUDA_CHECK(cudaGetLastError());
      return y;
    }
    return texelator_prefill(handle, x, tokens, shape, stream);
  }
  TORCH_CHECK(tokens <= 65535, "token grid exceeds CUDA grid.y limit");
  auto y = at::empty(shape, x.options());
  dim3 grid((handle->M + 7) / 8, tokens);
#define LAUNCH_FP16(KVALUE) texelator_bc4_kernel<KVALUE><<<grid, 256, 0, stream>>>(handle->texture, \
      static_cast<const __half *>(x.data_ptr()), handle->scales, \
      static_cast<const __half *>(handle->bias), static_cast<__half *>(y.data_ptr()), \
      handle->M, handle->K, handle->blocks_per_row, tokens)
#define LAUNCH_BF16(KVALUE) texelator_bc4_bf16_kernel<KVALUE><<<grid, 256, 0, stream>>>(handle->texture, \
      static_cast<const __nv_bfloat16 *>(x.data_ptr()), handle->scales, \
      static_cast<const __nv_bfloat16 *>(handle->bias), static_cast<__nv_bfloat16 *>(y.data_ptr()), \
      handle->M, handle->K, handle->blocks_per_row, tokens)
  TORCH_CHECK(!handle->bias || handle->bias_type == x.scalar_type(),
              "input and bias dtypes must match");
#define LAUNCH_K(KVALUE) do { \
  if (x.scalar_type() == at::kBFloat16) LAUNCH_BF16(KVALUE); \
  else LAUNCH_FP16(KVALUE); \
} while (0)
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
#undef LAUNCH_BF16
#undef LAUNCH_FP16
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

void texelator_set_prefill_backend(uint64_t value, int backend) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  TORCH_CHECK(backend == 0 || backend == 1 || backend == 2 || backend == 3 ||
              backend == 4 || backend == 5 || backend == 6 ||
              backend == 7 || backend == 8 || backend == 9,
              "prefill backend must be 0 (original fused), 1 (BF16 cuBLAS), "
              "2 (direct texture-to-MMA), 3 (pipelined texture-to-MMA), "
              "4 (graphics-style tile-local texture-to-MMA), or "
              "5 (CUTLASS baseline), 6 (T1 M128/N128), "
              "7 (T2 M128/N64), 8 (T3 M64/N128), or "
              "9 (small-output M32/N32/K64)");
  handle->prefill_backend = backend;
}

std::vector<int64_t> texelator_handle_info(uint64_t value) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  return {handle->M, handle->K, handle->prefill_threshold,
          handle->prefill_tile_rows, handle->prefill_backend,
          handle->pipelined_supported ? 1 : 0,
          handle->graphics_supported ? 1 : 0};
}

std::vector<int64_t> texelator_pipeline_kernel_info() {
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(
      &attributes, texelator_bf16_pipelined_512_kernel));
  return {attributes.numRegs,
          int64_t(attributes.sharedSizeBytes),
          int64_t(attributes.localSizeBytes),
          attributes.maxThreadsPerBlock,
          attributes.maxDynamicSharedSizeBytes,
          int64_t(sizeof(Bf16PipelineShared))};
}

std::vector<int64_t> texelator_graphics_kernel_info() {
  cudaFuncAttributes attributes{};
  CUDA_CHECK(cudaFuncGetAttributes(
      &attributes, texelator_bf16_graphics_512_kernel));
  return {attributes.numRegs,
          int64_t(attributes.sharedSizeBytes),
          int64_t(attributes.localSizeBytes),
          attributes.maxThreadsPerBlock,
          attributes.maxDynamicSharedSizeBytes,
          int64_t(sizeof(Bf16GraphicsShared))};
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
