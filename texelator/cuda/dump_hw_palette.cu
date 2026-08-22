#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::cerr << "CUDA: " << cudaGetErrorString(e) << "\n"; return 1; } } while (0)

struct Block { uint8_t bytes[8]; };

static Block make_block(int8_t endpoint0, int8_t endpoint1) {
  Block block{};
  block.bytes[0] = static_cast<uint8_t>(endpoint0);
  block.bytes[1] = static_cast<uint8_t>(endpoint1);
  uint64_t selectors = 0;
  for (int i = 0; i < 16; ++i) selectors |= uint64_t(i & 7) << (3 * i);
  for (int i = 0; i < 6; ++i) block.bytes[i + 2] = (selectors >> (8 * i)) & 255;
  return block;
}

__global__ void dump(cudaTextureObject_t texture, float *output, int pairs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pairs) return;
  int x = index % 255, y = index / 255;
  for (int selector = 0; selector < 8; ++selector)
    output[index * 8 + selector] = tex2D<float>(
        texture, x * 4 + (selector & 3) + .5f,
        y * 4 + (selector >> 2) + .5f);
}

int main(int argc, char **argv) {
  if (argc != 2) { std::cerr << "usage: dump_hw_palette OUTPUT_DIRECTORY\n"; return 2; }
  fs::path output_directory = argv[1];
  fs::create_directories(output_directory);
  cudaDeviceProp properties{};
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  constexpr int pairs = 255 * 255;
  std::vector<Block> host(pairs);
  for (int y = 0; y < 255; ++y)
    for (int x = 0; x < 255; ++x)
      host[y * 255 + x] = make_block(int8_t(x - 127), int8_t(y - 127));
  cudaArray_t array;
  cudaChannelFormatDesc description = cudaCreateChannelDesc<uint2>();
  CUDA_CHECK(cudaMallocArray(&array, &description, 255, 255));
  CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, host.data(), 255 * 8, 255 * 8, 255, cudaMemcpyHostToDevice));
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
  cudaTextureObject_t texture;
  CUDA_CHECK(cudaCreateTextureObject(&texture, &resource, &texture_description, &view));
  float *device_output;
  CUDA_CHECK(cudaMalloc(&device_output, size_t(pairs) * 8 * sizeof(float)));
  dump<<<(pairs + 255) / 256, 256>>>(texture, device_output, pairs);
  CUDA_CHECK(cudaGetLastError());
  std::vector<float> values(size_t(pairs) * 8);
  CUDA_CHECK(cudaMemcpy(values.data(), device_output, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
  std::ofstream binary(output_directory / "palette.bin", std::ios::binary);
  binary.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
  std::ofstream metadata(output_directory / "palette.json");
  metadata << "{\n"
           << "  \"device\": \"" << properties.name << "\",\n"
           << "  \"compute_capability\": \"" << properties.major << "." << properties.minor << "\",\n"
           << "  \"endpoint_pairs\": " << pairs << ",\n"
           << "  \"values_per_pair\": 8,\n"
           << "  \"bytes\": " << values.size() * sizeof(float) << "\n"
           << "}\n";
  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaDestroyTextureObject(texture));
  CUDA_CHECK(cudaFreeArray(array));
  std::cout << "wrote " << (output_directory / "palette.bin") << " for " << properties.name << "\n";
  return 0;
}
