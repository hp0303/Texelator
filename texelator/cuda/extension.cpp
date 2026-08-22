#include <torch/extension.h>
#include <c10/util/Optional.h>

uint64_t texelator_pack_encoded(torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>);
torch::Tensor texelator_linear(uint64_t, torch::Tensor);
void texelator_free(uint64_t);
void texelator_set_lookahead(uint64_t, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack_encoded", &texelator_pack_encoded, pybind11::arg("blocks"),
             pybind11::arg("scales"), pybind11::arg("bias") = c10::nullopt);
  module.def("linear", &texelator_linear);
  module.def("set_lookahead", &texelator_set_lookahead);
  module.def("free", &texelator_free);
}
