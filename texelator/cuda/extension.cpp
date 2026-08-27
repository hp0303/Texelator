#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>

uint64_t texelator_pack_encoded(at::Tensor, at::Tensor, c10::optional<at::Tensor>);
at::Tensor texelator_linear(uint64_t, at::Tensor);
void texelator_free(uint64_t);
void texelator_set_lookahead(uint64_t, int);
void texelator_set_prefill(uint64_t, int, int);
void texelator_set_prefill_backend(uint64_t, int);
std::vector<int64_t> texelator_handle_info(uint64_t);
std::vector<int64_t> texelator_pipeline_kernel_info();
std::vector<int64_t> texelator_graphics_kernel_info();
std::vector<int64_t> texelator_cutlass_texture_kernel_info();
std::vector<int64_t> texelator_cutlass_t1_kernel_info();
std::vector<int64_t> texelator_cutlass_t2_kernel_info();
std::vector<int64_t> texelator_cutlass_t3_kernel_info();
std::vector<int64_t> texelator_cutlass_texture_smalln_kernel_info();
at::Tensor texelator_palette_probe();
std::vector<at::Tensor> texelator_quantize_activation_fp4(at::Tensor);
std::vector<at::Tensor> texelator_decode_weight_fp4(uint64_t);
void texelator_quantize_activation_fp4_out(at::Tensor, at::Tensor, at::Tensor);
void texelator_decode_weight_fp4_out(uint64_t, at::Tensor, at::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack_encoded", &texelator_pack_encoded, pybind11::arg("blocks"),
             pybind11::arg("scales"), pybind11::arg("bias") = c10::nullopt);
  module.def("linear", &texelator_linear);
  module.def("set_lookahead", &texelator_set_lookahead);
  module.def("set_prefill", &texelator_set_prefill);
  module.def("set_prefill_backend", &texelator_set_prefill_backend);
  module.def("handle_info", &texelator_handle_info);
  module.def("pipeline_kernel_info", &texelator_pipeline_kernel_info);
  module.def("graphics_kernel_info", &texelator_graphics_kernel_info);
  module.def("cutlass_texture_kernel_info", &texelator_cutlass_texture_kernel_info);
  module.def("cutlass_t1_kernel_info", &texelator_cutlass_t1_kernel_info);
  module.def("cutlass_t2_kernel_info", &texelator_cutlass_t2_kernel_info);
  module.def("cutlass_t3_kernel_info", &texelator_cutlass_t3_kernel_info);
  module.def("cutlass_smalln_kernel_info", &texelator_cutlass_texture_smalln_kernel_info);
  module.def("free", &texelator_free);
  module.def("palette_probe", &texelator_palette_probe);
  module.def("quantize_activation_fp4", &texelator_quantize_activation_fp4);
  module.def("decode_weight_fp4", &texelator_decode_weight_fp4);
  module.def("quantize_activation_fp4_out", &texelator_quantize_activation_fp4_out);
  module.def("decode_weight_fp4_out", &texelator_decode_weight_fp4_out);
}
