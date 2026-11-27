/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <torch/extension.h>

#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "gather",
      [](const torch::Tensor& mapping,
         const torch::Tensor& indices,
         const torch::Tensor& invalid,
         const std::vector<int64_t>& shape) -> torch::Tensor {
        const int64_t blocks = mapping.size(-1);
        auto result =
            torch::index_select(mapping.reshape({-1, blocks}), 1, indices);
        result.masked_fill_(invalid.unsqueeze(0), -1);
        return result.reshape(shape);
      },
      pybind11::call_guard<pybind11::gil_scoped_release>(),
      "Native NPU gather and invalid-position fill");
}
