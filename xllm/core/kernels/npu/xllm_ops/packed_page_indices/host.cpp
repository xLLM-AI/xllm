/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */

#include <torch/extension.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace {

class HostPlan final {
 public:
  explicit HostPlan(torch::Tensor like,
                    std::vector<int64_t> shape,
                    int64_t layers,
                    int64_t blocks)
      : like_(std::move(like)),
        shape_(std::move(shape)),
        layers_(layers),
        blocks_(blocks) {}

  torch::Tensor allocate() const {
    return torch::empty(shape_, like_.options());
  }

  std::pair<torch::Tensor, torch::Tensor> allocate_workspace() const {
    torch::Tensor pages = torch::empty({shape_.back()}, like_.options());
    const int64_t padded_blocks = ((blocks_ + 32) / 32) * 32;
    torch::Tensor padded =
        torch::empty({layers_, padded_blocks}, like_.options());
    return {std::move(pages), std::move(padded)};
  }

  torch::Tensor gather(const torch::Tensor& mapping,
                       const torch::Tensor& pages) const {
    torch::Tensor output = mapping.index_select(/*dim=*/1, pages);
    return shape_.size() == 1 ? output.reshape(shape_) : output;
  }

 private:
  torch::Tensor like_;
  std::vector<int64_t> shape_;
  int64_t layers_;
  int64_t blocks_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<HostPlan>(module, "HostPlan")
      .def(pybind11::
               init<torch::Tensor, std::vector<int64_t>, int64_t, int64_t>())
      .def("allocate", &HostPlan::allocate)
      .def("allocate_workspace", &HostPlan::allocate_workspace)
      .def("gather", &HostPlan::gather);
}
