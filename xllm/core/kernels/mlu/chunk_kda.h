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
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

namespace xllm::kernel::mlu {

// Returns the chunk size used by the KDA prefill kernel. The default is 64;
// XLLM_MLU_KDA_CHUNK_SIZE may select 16.
int64_t kda_prefill_chunk_size();

// Inference-only, variable-length KDA prefill. The implementation follows the
// chunk/WY decomposition used by Flash Linear Attention while keeping KDA's
// per-key log gate and V-first recurrent-state layout explicit.
class ChunkKDAImpl final : public torch::nn::Module {
 public:
  static constexpr int64_t kDefaultChunkSize = 64;

  explicit ChunkKDAImpl(int64_t num_heads);
  ~ChunkKDAImpl() override = default;

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& q,
      const torch::Tensor& k,
      const torch::Tensor& v,
      const torch::Tensor& log_gate,
      const torch::Tensor& beta,
      const torch::Tensor& initial_state,
      const torch::Tensor& cu_seqlens,
      const torch::Tensor& chunk_indices,
      bool output_final_state,
      bool use_qk_l2norm);

 private:
  int64_t total_core_num_ = 0;
  int64_t num_heads_ = 0;
};

TORCH_MODULE(ChunkKDA);

}  // namespace xllm::kernel::mlu
