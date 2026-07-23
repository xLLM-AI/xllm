/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0.
==============================================================================*/

// Composition-based LoRA wrapper around ColumnParallelLinear.
//
// Used for gate_up_proj (fused gate+up in Qwen family). LoRA A/B come from
// PEFT under two distinct target modules: "gate_proj" and "up_proj". This
// wrapper reads both per-proj deltas from LoRARuntime and concatenates them
// along the output dim to match the base's fused output.
//
// LoRA math (fused gate_up):
//   gate_delta = B_gate @ A_gate @ x     A: [r, hidden]  B: [inter, r]
//   up_delta   = B_up   @ A_up   @ x
//   y = base(x)  # shape [T, 2*inter_local]
//   y += cat([gate_delta, up_delta], -1) * scaling

#pragma once

#include <torch/torch.h>

#include <string>

#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"

namespace xllm {
namespace layer {

class LoRAColumnParallelLinearImpl : public torch::nn::Module {
 public:
  LoRAColumnParallelLinearImpl() = default;

  // Signature mirrors ColumnParallelLinearImpl (with LinearExtraArgs).
  // Extra parameter:
  //   fused: "gate_up_proj" -> lookup A/B for "gate_proj" + "up_proj",
  //          concat delta along dim=-1.
  //   single proj name (e.g. "o_proj" not applicable here since that's Row)
  //     - reserved for future use.
  LoRAColumnParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      bool gather_output,
      const QuantArgs& quant_args,
      ProcessGroup* process_group,
      const torch::TensorOptions& options,
      const std::string& proj_name,
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs());

  torch::Tensor forward(torch::Tensor input);

  void load_state_dict(const StateDict& state_dict);
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);
  void load_state_dict(const StateDict& state_dict,
                       int32_t shard_tensor_count,
                       const std::vector<int64_t>& shard_sizes);

  torch::Tensor weight() const { return base_->weight(); }
  std::optional<torch::Tensor> get_input_scale() const {
    return base_->get_input_scale();
  }

  void pretty_print(std::ostream& stream) const {
    stream << name() << " (LoRA-wrapped/" << proj_name_
           << ")  base_weight=" << base_->weight().sizes();
  }

 private:
  ColumnParallelLinear base_{nullptr};
  std::string proj_name_;  // "gate_up_proj" for fused, or single
  int64_t in_features_ = 0;
  int64_t out_features_ = 0;      // per-rank local, total (fused = 2*inter)
  int64_t out_size_local_ = 0;    // per-rank, matches base
  int64_t inter_size_local_ = 0;  // per-rank, half of out_size_local_ if fused
  bool is_fused_gate_up_ = false;
  int64_t tp_rank_ = 0;
  int64_t tp_world_size_ = 1;
};
TORCH_MODULE(LoRAColumnParallelLinear);

}  // namespace layer
}  // namespace xllm
