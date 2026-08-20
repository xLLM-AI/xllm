/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "rms_norm_loader.h"

namespace xllm {
namespace layer {

RMSNormLoader::RMSNormLoader(uint64_t weight_count,
                             const ModelContext& context,
                             LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  if (load_to_host()) {
    working_tensors()[0] = torch::zeros({1});
  } else {
    working_tensors()[0] = torch::zeros({1}).to(options);
  }
}

void RMSNormLoader::load_state_dict(const StateDict& state_dict) {
  set_weight(state_dict, "weight", 0, load_to_host());
  working_tensors()[0] = working_tensors()[0].to(dtype_);
}

void RMSNormLoader::verify_loaded_weights(const std::string& weight_str) const {
  CHECK(working_tensors()[0].sizes() != std::vector<int64_t>({1}))
      << "final norm weight is not loaded for " << weight_str;
}

}  // namespace layer
}  // namespace xllm
