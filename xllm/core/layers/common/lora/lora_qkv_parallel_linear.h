/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// Composition-based LoRA wrapper around QKVParallelLinear.
//
// Design decisions (from Spike Day 2 code reading):
//   Base linear forward is non-virtual + private weight members, so we wrap
//   by composition rather than inheritance.
//   Wrapper owns a base QKVParallelLinear registered as "base" plus the
//   LoRA A/B tensors. Base weight loading path is unchanged; adapter A/B
//   are set via set_lora_weights (Spike phase) or load_lora_state_dict
//   (real adapter manager, later phases).
//   Delta path: y = base(x); delta = B(A(x)) * scaling; return y + delta.
//   Fused Q/K/V is handled by treating lora_A as a single [rank, hidden]
//   projection and lora_B as [q_size + 2*kv_size_local, rank] (col-shard
//   under TP, matching base's fused output layout).

#pragma once

#include <torch/torch.h>

#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"

namespace xllm {
namespace layer {

class LoRAQKVParallelLinearImpl : public torch::nn::Module {
 public:
  LoRAQKVParallelLinearImpl() = default;

  // Signature mirrors QKVParallelLinearImpl exactly so we can drop-in
  // replace the member type in Qwen2Attention with zero call-site changes.
  //
  // q_has_gate: when true, the base linear was constructed with
  //   num_heads_effective = num_heads_attn * 2 (Qwen3-Next attn_output_gate
  //   fuses q + gate into the q lane). The wrapper then sizes q_size_local
  //   as num_heads_effective * head_size to match base's fused output, and
  //   the PEFT adapter is expected to have been trained on the same fused
  //   lane (B_q shape [q_size_fused, rank]).
  LoRAQKVParallelLinearImpl(int64_t hidden_size,
                            int64_t num_heads,
                            int64_t num_kv_heads,
                            int64_t head_size,
                            int64_t num_kv_head_replicas,
                            bool bias,
                            bool gather_output,
                            const ParallelArgs& parallel_args,
                            const torch::TensorOptions& options,
                            const QuantArgs& quant_args = QuantArgs{},
                            bool q_has_gate = false);

  // Same signature as base: y = base(x); delta = B(A(x)) * scaling.
  torch::Tensor forward(torch::Tensor input);

  // Base weight loading -- direct passthrough.
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);
  void load_state_dict(const StateDict& state_dict);

  // Passthrough accessors -- keeps wrapper drop-in compatible with
  // QKVParallelLinearImpl for callers that peek at the base state.
  void pretty_print(std::ostream& stream) const {
    stream << name()
           << " (LoRA-wrapped)  base_weight=" << base_->weight().sizes()
           << "  lora_active=" << (lora_active_ ? "true" : "false");
  }
  torch::Tensor weight() const { return base_->weight(); }
  std::optional<torch::Tensor> get_input_scale() const {
    return base_->get_input_scale();
  }

  // ---- LoRA-specific interface (Spike + future adapter manager) ----

  // Set LoRA weights directly (Spike hardcoded flow).
  //   lora_a: [rank, hidden_size]                     replicated under TP
  //   lora_b: [q_size + 2 * kv_size_local, rank]      col-sharded under TP
  //   scaling: alpha / rank (typically pre-multiplied into B for real loads)
  void set_lora_weights(const torch::Tensor& lora_a,
                        const torch::Tensor& lora_b,
                        double scaling);

  // Disable LoRA delta (return base output only).
  void clear_lora() { lora_active_ = false; }

  bool has_active_lora() const { return lora_active_; }
  int64_t lora_rank() const { return lora_rank_; }

 private:
  // Base linear held via ModuleHolder (shared_ptr semantics) but NOT
  // register_module'd on this wrapper. Rationale: keeping the base out of
  // this wrapper's module tree preserves the original state_dict key
  // layout — a checkpoint written for vanilla QKVParallelLinear (path
  // "qkv_proj.weight") continues to load unchanged. If we registered the
  // base as a submodule, keys would become "qkv_proj.base.weight" and
  // every existing checkpoint would break.
  //
  // Base construction happens inside the wrapper's ctor; forward() and
  // load_state_dict() delegate directly.
  QKVParallelLinear base_{nullptr};

  // LoRA delta parameters. Held as raw tensors (not registered params) so
  // they can be swapped/cleared at runtime without perturbing the module
  // tree. Real adapter manager will register a slot pool separately.
  torch::Tensor lora_a_;  // [rank, hidden]                          replicated
  torch::Tensor lora_b_;  // [q_size + 2 * kv_size_local, rank]      col-shard
  double lora_scaling_ = 1.0;
  int64_t lora_rank_ = 0;
  bool lora_active_ = false;

  // Cached shape info for TP-aware LoRA slicing (populated in ctor).
  int64_t hidden_size_ = 0;
  int64_t q_size_local_ = 0;    // q_size / tp_world_size
  int64_t kv_size_local_ = 0;   // kv_size / tp_world_size (with replication)
  int64_t out_size_local_ = 0;  // q_size_local + 2 * kv_size_local
  int64_t tp_rank_ = 0;
  int64_t tp_world_size_ = 1;

  // True for Qwen3-Next-style attn_output_gate: q_size_local includes the
  // fused gate lane, and the PEFT adapter's B_q is expected at fused width.
  // Currently informational — sizing already uses fused num_heads.
  bool q_has_gate_ = false;
};
TORCH_MODULE(LoRAQKVParallelLinear);

}  // namespace layer
}  // namespace xllm
