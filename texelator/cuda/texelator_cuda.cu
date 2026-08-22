#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CUDA_CHECK(call) do { \
  cudaError_t error = (call); \
  TORCH_CHECK(error == cudaSuccess, "CUDA: ", cudaGetErrorString(error)); \
} while (0)

struct Handle {
  int M, K, blocks_per_row, device, lookahead = 2;
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

uint64_t texelator_pack_encoded(torch::Tensor blocks, torch::Tensor scales,
                             c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(!blocks.is_cuda() && blocks.scalar_type() == torch::kUInt8 && blocks.is_contiguous(),
              "blocks must be contiguous CPU uint8");
  TORCH_CHECK(!scales.is_cuda() && scales.scalar_type() == torch::kFloat32 && scales.is_contiguous(),
              "scales must be contiguous CPU float32");
  int M = scales.numel();
  TORCH_CHECK(M > 0 && blocks.numel() % (size_t(M) * 8) == 0, "invalid encoded size");
  int blocks_per_row = blocks.numel() / (size_t(M) * 8);
  int K = blocks_per_row * 16;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  auto *handle = new Handle{M, K, blocks_per_row, device, 2};
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
    TORCH_CHECK(contiguous.is_cuda() && contiguous.scalar_type() == torch::kFloat16 && contiguous.numel() == M,
                "bias must be CUDA FP16 [M]");
    CUDA_CHECK(cudaMalloc(&handle->bias, size_t(M) * 2));
    CUDA_CHECK(cudaMemcpy(handle->bias, contiguous.data_ptr(), size_t(M) * 2, cudaMemcpyDeviceToDevice));
  }
  return reinterpret_cast<uint64_t>(handle);
}

torch::Tensor texelator_linear(uint64_t value, torch::Tensor x) {
  auto *handle = reinterpret_cast<Handle *>(value);
  TORCH_CHECK(handle, "null Texelator handle");
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous(),
              "input must be contiguous CUDA FP16");
  TORCH_CHECK(x.size(-1) == handle->K && x.get_device() == handle->device, "input shape/device mismatch");
  int tokens = x.numel() / handle->K;
  TORCH_CHECK(tokens <= 65535, "token grid exceeds CUDA grid.y limit");
  auto shape = x.sizes().vec();
  shape.back() = handle->M;
  auto y = torch::empty(shape, x.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(handle->device);
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

void texelator_free(uint64_t value) { delete reinterpret_cast<Handle *>(value); }
