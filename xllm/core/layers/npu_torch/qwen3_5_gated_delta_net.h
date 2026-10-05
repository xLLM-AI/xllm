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

#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "qwen3_next_gated_delta_net.h"

namespace xllm {
namespace layer {

namespace qwen3_5_gdn_internal {

const MegaGdnPrefillIndicesCache& get_or_build_prefill_indices(
    const AttentionMetadata& attn_metadata,
    const std::vector<int32_t>& live_slots,
    const std::vector<int64_t>& validity_mask,
    int64_t checkpoint_stride,
    int64_t num_slots,
    const torch::Device& device,
    const std::vector<LinearStateCacheOp>& cache_ops = {});

}  // namespace qwen3_5_gdn_internal

class Qwen3_5GatedDeltaNetImpl : public Qwen3NextGatedDeltaNetImpl {
 public:
  Qwen3_5GatedDeltaNetImpl() = default;
  Qwen3_5GatedDeltaNetImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params) override;

 protected:
  std::optional<
      std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
  project_split_inputs(const torch::Tensor& hidden_states,
                       const AttentionMetadata& attn_metadata) override;
  bool use_fla_ssm_state_layout() const override { return true; }

  void load_projection_state_dict(const StateDict& state_dict) override;
  void verify_projection_weights(const std::string& prefix) const override;

 private:
  ColumnParallelLinear in_proj_qkv_{nullptr};
  ColumnParallelLinear in_proj_z_{nullptr};
  ColumnParallelLinear in_proj_b_{nullptr};
  ColumnParallelLinear in_proj_a_{nullptr};
};
TORCH_MODULE(Qwen3_5GatedDeltaNet);

}  // namespace layer
}  // namespace xllm
