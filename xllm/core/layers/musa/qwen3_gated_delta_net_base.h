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

#pragma once

#include <torch/torch.h>

#include <string>
#include <tuple>
#include <utility>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/attention.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm_gated.h"

namespace xllm {
namespace layer {

class Qwen3GatedDeltaNetBaseImpl : public torch::nn::Module {
 public:
  Qwen3GatedDeltaNetBaseImpl() = default;
  Qwen3GatedDeltaNetBaseImpl(const ModelArgs& args,
                             const QuantArgs& quant_args,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options);

  virtual void load_state_dict(const StateDict& state_dict) = 0;
  virtual void verify_loaded_weights(const std::string& prefix) const = 0;

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params);

 protected:
  virtual std::pair<torch::Tensor, torch::Tensor> project_decode_inputs(
      const torch::Tensor& hidden_states) = 0;
  virtual std::pair<torch::Tensor, torch::Tensor> project_flat_inputs(
      const torch::Tensor& hidden_states) = 0;
  virtual bool use_fla_ssm_state_layout() const { return false; }

  void load_common_state_dict(const StateDict& state_dict);
  void verify_common_loaded_weights(const std::string& prefix) const;

  torch::Tensor get_linear_state_indices(const ModelInputParams& input_params,
                                         const torch::Device& device) const;

  std::pair<torch::Tensor, torch::Tensor> project_padded_inputs(
      const torch::Tensor& hidden_states,
      const AttentionMetadata& attn_metadata);

  torch::Tensor reshape_qkvz_unpad(const AttentionMetadata& attn_metadata,
                                   const torch::Tensor& padded_qkvz) const;

  torch::Tensor reshape_qkvz_with_pad(const AttentionMetadata& attn_metadata,
                                      const torch::Tensor& qkvz) const;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> process_mixed_qkv(
      torch::Tensor& mixed_qkv) const;

  int64_t num_k_heads_ = 0;
  int64_t num_v_heads_ = 0;
  int64_t head_k_dim_ = 0;
  int64_t head_v_dim_ = 0;
  int64_t k_size_ = 0;
  int64_t v_size_ = 0;
  int64_t tp_size_ = 1;
  int64_t rank_ = 0;
  int32_t conv_kernel_size_ = 0;

  ColumnParallelLinear conv1d_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  RmsNormGated norm_{nullptr};

  DEFINE_WEIGHT(dt_bias);
  DEFINE_WEIGHT(A_log);

#if defined(USE_CUDA) || defined(USE_MUSA)
  // Persistent output buffers consumed by xllm::kernel::
  // fused_qkvzba_split_reshape_cat in lieu of the libtorch
  // `reshape().contiguous() ... torch::cat()` chain. Same lazy / grow-only
  // pattern used by ColumnParallelLinearImpl::output_buf_ and
  // AttentionImpl::output_buf_: sized on the first forward (during graph
  // warmup), reused on every replay via narrow() views. Eager calls also
  // benefit (one allocation per process instead of per-step).
  mutable torch::Tensor mixed_qkv_out_buf_;
  mutable torch::Tensor z_out_buf_;
  mutable torch::Tensor b_out_buf_;
  mutable torch::Tensor a_out_buf_;

  // Persistent output buffer for the decode-path causal_conv1d_update call.
  // Without this, the kernel falls through to its libtorch slow path
  // (`weight.to(fp32)` / `x.to(fp32)` / `torch::empty_like(x_f32)`) which
  // triggers EmptyStridedMUSA -> MUSA stream-capture abort. Providing a
  // pre-allocated buffer unlocks the in-house `causal_conv1d_decode_fused`
  // fast path (see gdn_ops.cpp::causal_conv1d_update fast-path guard).
  mutable torch::Tensor conv1d_decode_out_buf_;

  // Persistent output buffer for the in-house fused_gated_delta_rule_decode
  // kernel. Wired into `MateGatedDeltaRuleDecodeParams::decode_output` so
  // the kernel skips its `torch::empty({B, Hv, V}, ...)` fallback (which
  // hits EmptyMUSA mid-capture) and writes directly into pre-allocated
  // storage. Reused across replays; same lazy / grow-only contract as the
  // other graph-safe buffers above.
  mutable torch::Tensor fused_gdn_decode_out_buf_;
#endif
};

}  // namespace layer
}  // namespace xllm
