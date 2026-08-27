#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/matrix.h"

#include <cstdint>
#include <vector>

#define CUDA_CHECK_MAINLOOP(call) do { \
  cudaError_t error = (call); \
  TORCH_CHECK(error == cudaSuccess, "CUDA: ", cudaGetErrorString(error)); \
} while (0)

namespace texelator_cutlass_t1 {

using Element = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using OutputOp = cutlass::epilogue::thread::LinearCombination<
    Element, 128 / cutlass::sizeof_bits<Element>::value,
    ElementAccumulator, ElementAccumulator>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using Default = cutlass::gemm::kernel::DefaultGemm<
    Element, LayoutA, 8,
    Element, LayoutB, 8,
    Element, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    OutputOp,
    Swizzle,
    2,
    false,
    cutlass::arch::OpMultiplyAdd>;

using BaseMma = typename Default::Mma;
using ThreadMapB = typename BaseMma::IteratorB::ThreadMap;

// ReadableTileIterator compatible with CUTLASS MmaPipelined. Its Fragment is
// exactly the fragment expected by the production shared-memory iterator, but
// load() reconstructs the B tile from BC4 through TEX instead of global BF16.
class TextureIteratorB {
 public:
  using Shape = cutlass::MatrixShape<BaseMma::Shape::kK, BaseMma::Shape::kN>;
  using Element = texelator_cutlass_t1::Element;
  using Layout = LayoutB;
  using ThreadMap = ThreadMapB;
  using Index = int;
  using LongIndex = int64_t;
  using TensorCoord = cutlass::MatrixCoord;
  using Pointer = Element const *;
  using TensorRef = cutlass::TensorRef<Element const, Layout>;
  using AccessType = cutlass::AlignedArray<Element, 8>;
  using Fragment = cutlass::Array<
      Element, ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  struct Params {
    uint64_t texture = 0;
    float const *scales = nullptr;
    int rows = 0;
    int columns = 0;
    int blocks_per_row = 0;
    CUTLASS_HOST_DEVICE Params() = default;
    CUTLASS_HOST_DEVICE explicit Params(Layout const &) {}
  };

 private:
  Params params_;
  int base_k_ = 0;
  int base_n_ = 0;
  int extent_k_ = 0;
  int extent_n_ = 0;
  bool mask_ = true;

 public:
  CUTLASS_HOST_DEVICE TextureIteratorB() = default;

  CUTLASS_HOST_DEVICE TextureIteratorB(
      Params const &params, Pointer, TensorCoord extent, int thread_id,
      TensorCoord const &threadblock_offset, int const * = nullptr)
      : params_(params), extent_k_(extent.row()), extent_n_(extent.column()) {
    auto offset = ThreadMap::initial_offset(thread_id);
    base_k_ = threadblock_offset.row() + offset.contiguous();
    base_n_ = threadblock_offset.column() + offset.strided();
  }

  CUTLASS_HOST_DEVICE TextureIteratorB &operator++() {
    base_k_ += Shape::kRow;
    return *this;
  }

  CUTLASS_DEVICE void clear_mask(bool clear = true) {
    if (clear) mask_ = false;
  }

  CUTLASS_DEVICE void load(Fragment &fragment) const {
    constexpr int kContiguous = ThreadMap::Iterations::kContiguous;
    constexpr int kStrided = ThreadMap::Iterations::kStrided;
    constexpr int kAccesses = ThreadMap::Iterations::kCount;
    static_assert(ThreadMap::kElementsPerAccess == 8,
                  "texture iterator requires eight contiguous BF16 elements");

    float4 first[kAccesses];
    float4 second[kAccesses];
    float scale[kAccesses];
    int start_k[kAccesses];
    int row[kAccesses];
    bool valid[kAccesses];

    // Issue every independent texture request before consuming any result.
#pragma unroll
    for (int s = 0; s < kStrided; ++s) {
#pragma unroll
      for (int c = 0; c < kContiguous; ++c) {
        int index = c + s * kContiguous;
        int k = base_k_ + c * ThreadMap::Delta::kContiguous;
        int n = base_n_ + s * ThreadMap::Delta::kStrided;
        start_k[index] = k;
        row[index] = n;
        valid[index] = mask_ && k >= 0 && k + 7 < extent_k_ &&
                       n >= 0 && n < extent_n_;
        first[index] = make_float4(0, 0, 0, 0);
        second[index] = make_float4(0, 0, 0, 0);
        scale[index] = 0.0f;
        if (valid[index]) {
          int block = k >> 4;
          int qy = ((k & 15) >> 2) & 2;
          first[index] = tex2Dgather<float4>(
              params_.texture, block * 4 + 1.f,
              n * 4 + qy + 1.f, 0);
          second[index] = tex2Dgather<float4>(
              params_.texture, block * 4 + 3.f,
              n * 4 + qy + 1.f, 0);
          scale[index] = params_.scales[n];
        }
      }
    }

    AccessType *access = reinterpret_cast<AccessType *>(&fragment);
#pragma unroll
    for (int index = 0; index < kAccesses; ++index) {
      if (valid[index] && (start_k[index] & 7) == 0) {
        float value[8] = {
            first[index].w, first[index].z,
            second[index].w, second[index].z,
            first[index].x, first[index].y,
            second[index].x, second[index].y};
#pragma unroll
        for (int element = 0; element < 8; ++element)
          access[index][element] = Element(value[element] * scale[index]);
      } else {
        access[index].clear();
      }
    }
  }
};

using TextureMma = cutlass::gemm::threadblock::MmaPipelined<
    typename BaseMma::Shape,
    typename BaseMma::IteratorA,
    typename BaseMma::SmemIteratorA,
    TextureIteratorB,
    typename BaseMma::SmemIteratorB,
    ElementAccumulator,
    LayoutC,
    typename BaseMma::Policy>;

using Kernel = cutlass::gemm::kernel::Gemm<
    TextureMma, typename Default::Epilogue, Swizzle, false>;

__global__ void add_bias(
    __nv_bfloat16 *output, __nv_bfloat16 const *bias,
    size_t count, int columns) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    float value = __bfloat162float(output[index]);
    value += __bfloat162float(bias[index % columns]);
    output[index] = __float2bfloat16(value);
  }
}

}  // namespace texelator_cutlass_t1

at::Tensor texelator_cutlass_t1_bf16(
    uint64_t texture, float const *scales, void const *bias,
    at::Tensor x, int rows, int columns, int blocks_per_row,
    std::vector<int64_t> const &shape, cudaStream_t stream) {
  using namespace texelator_cutlass_t1;
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_cuda() && x.is_contiguous(),
              "CUTLASS texture mainloop requires contiguous CUDA BF16 input");
  int tokens = x.numel() / columns;
  TORCH_CHECK(columns % ThreadblockShape::kK == 0,
              "K must be divisible by the CUTLASS K tile");
  auto output = at::empty(shape, x.options());

  cutlass::gemm::GemmCoord problem(tokens, rows, columns);
  Swizzle swizzle;
  cutlass::gemm::GemmCoord threadblock_shape(
      ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK);
  auto tiled = swizzle.get_tiled_shape(problem, threadblock_shape, 1);
  dim3 grid = swizzle.get_grid_shape(tiled);

  using RefA = typename TextureMma::IteratorA::TensorRef;
  using RefB = typename TextureIteratorB::TensorRef;
  using RefC = typename Default::Epilogue::OutputTileIterator::TensorRef;
  RefA ref_a(reinterpret_cast<Element *>(x.data_ptr()), LayoutA(columns));
  RefB ref_b(nullptr, LayoutB(columns));
  RefC ref_c(reinterpret_cast<Element *>(output.data_ptr()), LayoutC(rows));
  RefC ref_d(reinterpret_cast<Element *>(output.data_ptr()), LayoutC(rows));
  typename OutputOp::Params output_op(1.0f, 0.0f);
  typename Kernel::Params params(
      problem, tiled, ref_a, ref_b, ref_c, ref_d, output_op);
  params.params_B.texture = texture;
  params.params_B.scales = scales;
  params.params_B.rows = rows;
  params.params_B.columns = columns;
  params.params_B.blocks_per_row = blocks_per_row;

  cutlass::Kernel<Kernel><<<
      grid, Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage), stream>>>(params);
  CUDA_CHECK_MAINLOOP(cudaGetLastError());
  if (bias) {
    size_t count = size_t(tokens) * rows;
    add_bias<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(output.data_ptr()),
        static_cast<__nv_bfloat16 const *>(bias), count, rows);
    CUDA_CHECK_MAINLOOP(cudaGetLastError());
  }
  return output;
}

std::vector<int64_t> texelator_cutlass_t1_kernel_info() {
  using namespace texelator_cutlass_t1;
  cudaFuncAttributes attributes{};
  CUDA_CHECK_MAINLOOP(cudaFuncGetAttributes(
      &attributes, cutlass::Kernel<Kernel>));
  int active_blocks = 0;
  CUDA_CHECK_MAINLOOP(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreadCount,
      sizeof(typename Kernel::SharedStorage)));
  return {attributes.numRegs,
          int64_t(attributes.sharedSizeBytes),
          int64_t(attributes.localSizeBytes),
          attributes.maxThreadsPerBlock,
          int64_t(sizeof(typename Kernel::SharedStorage)),
          ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK,
          active_blocks};
}
