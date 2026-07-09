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

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"
#include "layers/common/partial_rotary_embedding.h"
#include "layers/common/qwen3_next_rms_norm.h"
#include "layers/common/rotary_embedding.h"
#include "layers/mlu/attention.h"

namespace xllm {
namespace layer {

class Qwen3_5AttentionImpl : public torch::nn::Module {
 public:
  Qwen3_5AttentionImpl() = default;
  Qwen3_5AttentionImpl(const ModelArgs& args,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options,
                       int32_t layer_id);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);
  void rotary_emb_forward(torch::Tensor& q,
                          torch::Tensor& k,
                          const torch::Tensor& positions,
                          const AttentionMetadata& attn_metadata);

 private:
  int64_t num_heads_;
  int64_t num_kv_heads_;
  int64_t num_kv_head_replicas_;
  int64_t head_dim_;
  int64_t q_size_;
  int64_t kv_size_;
  float scaling_;
  bool attn_output_gate_;
  int32_t layer_id_;
  int32_t rank_;
  bool use_split_qkv_ = false;

  QKVParallelLinear qkv_proj_{nullptr};
  ColumnParallelLinear q_proj_{nullptr};
  ColumnParallelLinear k_proj_{nullptr};
  ColumnParallelLinear v_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};

  Qwen3NextRMSNorm q_norm_{nullptr};
  Qwen3NextRMSNorm k_norm_{nullptr};

  Attention attn_{nullptr};
  MRotaryEmbedding rotary_emb_{nullptr};
  torch::Tensor mrope_cu_seq_lens_;
};
TORCH_MODULE(Qwen3_5Attention);

}  // namespace layer
}  // namespace xllm
