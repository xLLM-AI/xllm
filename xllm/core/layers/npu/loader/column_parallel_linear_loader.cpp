/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "column_parallel_linear_loader.h"

namespace xllm {
namespace layer {

ColumParallelLinearLoader::ColumParallelLinearLoader(
    uint64_t weight_count,
    const ModelContext& context,
    LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  if (load_to_host()) {
    working_tensors()[0] = torch::zeros(
        {1}, torch::TensorOptions().dtype(dtype_).device(torch::kCPU));
  } else {
    working_tensors()[0] = torch::zeros({1}).to(options);
  }
}

void ColumParallelLinearLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  if (dp_size_ > 1 || cp_size_ > 1) {
    set_weight(state_dict,
               "weight",
               0,
               0,
               dp_local_tp_rank_,
               dp_local_tp_size_,
               to_host);
  } else {
    set_weight(state_dict, "weight", 0, 0, to_host);
  }
  working_tensors()[0] = working_tensors()[0].to(dtype_);
}

void ColumParallelLinearLoader::verify_loaded_weights() const {
  if (mode() == LoadMode::kManual) {
    verify_loaded_weights("column_parallel_linear");
  }
}

void ColumParallelLinearLoader::verify_loaded_weights(
    const std::string& weight_str) const {
  CHECK(working_tensors()[0].sizes() != std::vector<int64_t>({1}))
      << "weight is not loaded for " << weight_str;
}

void ColumParallelLinearLoader::fuse_eagle3_quarot_input_rotation(
    torch::Tensor global_rotation) {
  auto& weight = working_tensors()[0];
  CHECK(weight.defined()) << "Eagle3 fc weight is not loaded";
  CHECK(weight.sizes() != std::vector<int64_t>({1}))
      << "Eagle3 fc weight is not loaded";
  CHECK_EQ(weight.dim(), 2) << "Eagle3 fc weight must be 2D, got "
                            << weight.sizes();
  CHECK_EQ(global_rotation.dim(), 2)
      << "QuaRot global_rotation must be a 2D tensor";
  CHECK_EQ(global_rotation.size(0), global_rotation.size(1))
      << "QuaRot global_rotation must be square";
  CHECK_EQ(weight.size(1) % 3, 0)
      << "Eagle3 fc input dim must be 3 * target hidden size, got "
      << weight.sizes();

  const int64_t hidden_size = weight.size(1) / 3;
  CHECK_EQ(global_rotation.size(0), hidden_size)
      << "QuaRot global_rotation hidden size mismatch, expected "
      << hidden_size << ", got " << global_rotation.size(0);

  const auto weight_device = weight.device();
  const auto weight_dtype = weight.scalar_type();
  auto weight_cpu = weight.to(torch::kCPU).to(torch::kFloat32).contiguous();
  auto rotation_cpu =
      global_rotation.to(torch::kCPU).to(torch::kFloat32).contiguous();

  std::vector<torch::Tensor> fused_chunks;
  fused_chunks.reserve(3);
  for (int64_t i = 0; i < 3; ++i) {
    auto chunk =
        weight_cpu.slice(/*dim=*/1, i * hidden_size, (i + 1) * hidden_size);
    fused_chunks.emplace_back(torch::matmul(chunk, rotation_cpu));
  }

  auto fused_weight = torch::cat(fused_chunks, /*dim=*/1)
                          .to(torch::TensorOptions()
                                  .dtype(weight_dtype)
                                  .device(weight_device))
                          .contiguous();
  weight = fused_weight;
}

}  // namespace layer
}  // namespace xllm
