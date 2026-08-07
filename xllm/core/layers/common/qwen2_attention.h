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

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/rms_norm.h"
#include "linear.h"
#include "lora/lora_qkv_parallel_linear.h"
#include "lora/lora_row_parallel_linear.h"
#include "rotary_embedding.h"

namespace xllm {
namespace layer {

class Qwen2AttentionImpl : public torch::nn::Module {
 public:
  Qwen2AttentionImpl() = default;
  Qwen2AttentionImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

  // Get FP8 input scale from qkv_proj for fused RMSNorm+FP8 quantization
  std::optional<torch::Tensor> get_fp8_input_scale() const;

 private:
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t num_kv_head_replicas_;
  int64_t head_dim_;
  int64_t q_size_;
  int64_t kv_size_;
  float scaling_;
  bool is_qwen3_style_;
  bool can_use_fused_qk_norm_rope_;

  LoRAQKVParallelLinear qkv_proj_{nullptr};  // LoRA-wrapped drop-in
  LoRARowParallelLinear o_proj_{nullptr};    // LoRA-wrapped drop-in
  RMSNorm q_norm_{nullptr};
  RMSNorm k_norm_{nullptr};
  Attention attn_{nullptr};
  MRotaryEmbedding rotary_emb_{nullptr};
};
TORCH_MODULE(Qwen2Attention);

}  // namespace layer
}  // namespace xllm
