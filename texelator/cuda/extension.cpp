#include <torch/extension.h>
#include <c10/util/Optional.h>

uint64_t texelator_pack_encoded(at::Tensor, at::Tensor, c10::optional<at::Tensor>);
at::Tensor texelator_linear(uint64_t, at::Tensor);
void texelator_free(uint64_t);
void texelator_set_lookahead(uint64_t, int);
void texelator_set_prefill(uint64_t, int, int);
at::Tensor texelator_palette_probe();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack_encoded", &texelator_pack_encoded, pybind11::arg("blocks"),
             pybind11::arg("scales"), pybind11::arg("bias") = c10::nullopt);
  module.def("linear", &texelator_linear);
  module.def("set_lookahead", &texelator_set_lookahead);
  module.def("set_prefill", &texelator_set_prefill);
  module.def("free", &texelator_free);
  module.def("palette_probe", &texelator_palette_probe);
}
