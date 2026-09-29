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

#include <string>
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/mlu/chunk_kda.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm_gated.h"

namespace xllm {
namespace layer {

// GLM-5-Next's bounded, per-key log-space gate. The result is FP32 and lies
// in [lower_bound, 0].
torch::Tensor glm5_next_safe_gate(const torch::Tensor& raw_gate,
                                  const torch::Tensor& A_log,
                                  const torch::Tensor& dt_bias,
                                  float lower_bound);

// Correctness-first KDA recurrence used by the MLU prefill fallback. State is
// laid out as [batch, heads, value_dim, key_dim].
std::tuple<torch::Tensor, torch::Tensor> glm5_next_kda_eager_recurrence(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& log_gate,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state,
    bool l2norm_qk);

// GLM-5-Next uses a sigmoid output gate after RMSNorm. This intentionally
// differs from Qwen's SiLU-gated RMSNorm.
torch::Tensor glm5_next_kda_apply_output_gate(const torch::Tensor& normalized,
                                              const torch::Tensor& output_gate);

class Glm5NextKDAImpl final : public torch::nn::Module {
 public:
  Glm5NextKDAImpl() = default;
  Glm5NextKDAImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options,
                  float gate_lower_bound = -5.0f);
  explicit Glm5NextKDAImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix = "") const;

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 private:
  torch::Tensor get_linear_state_indices(const ModelInputParams& input_params,
                                         const torch::Device& device) const;
  int64_t get_checkpoint_stride(const KVCache& kv_cache) const;
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> split_mixed_qkv(
      const torch::Tensor& mixed_qkv) const;

  int64_t num_heads_ = 0;
  int64_t local_num_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t projection_size_ = 0;
  int64_t local_projection_size_ = 0;
  int64_t tp_size_ = 1;
  int64_t rank_ = 0;
  int32_t conv_kernel_size_ = 0;
  float gate_lower_bound_ = -5.0f;

  ColumnParallelLinear q_proj_{nullptr};
  ColumnParallelLinear k_proj_{nullptr};
  ColumnParallelLinear v_proj_{nullptr};
  ColumnParallelLinear b_proj_{nullptr};
  ReplicatedLinear f_a_proj_{nullptr};
  ReplicatedLinear g_a_proj_{nullptr};
  ColumnParallelLinear f_b_proj_{nullptr};
  ColumnParallelLinear g_b_proj_{nullptr};
  ColumnParallelLinear q_conv1d_{nullptr};
  ColumnParallelLinear k_conv1d_{nullptr};
  ColumnParallelLinear v_conv1d_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  RmsNormGated o_norm_{nullptr};
  kernel::mlu::ChunkKDA chunk_kda_{nullptr};

  bool f_a_is_loaded_ = false;
  bool g_a_is_loaded_ = false;
  bool o_norm_is_loaded_ = false;

  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);
};

TORCH_MODULE(Glm5NextKDA);

}  // namespace layer
}  // namespace xllm
