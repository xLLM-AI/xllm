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

#include "dense_mlp.h"

#include <glog/logging.h>

#include "common/flash_comm1_context.h"
#include "kernels/ops_api.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

DenseMLPImpl::DenseMLPImpl(int64_t hidden_size,
                           int64_t intermediate_size,
                           bool is_gated,
                           bool has_bias,
                           const std::string& hidden_act,
                           bool enable_result_reduction,
                           const QuantArgs& quant_args,
                           ProcessGroup* process_group,
                           const torch::TensorOptions& options,
                           const std::string& module_prefix,
                           double swiglu_limit)
    : is_gated_(is_gated),
      intermediate_size_(intermediate_size),
      process_group_(process_group),
      hidden_act_(hidden_act),
      swiglu_limit_(swiglu_limit) {
  // Check if using w8a8 smoothquant quantization
  is_smoothquant_ = quant_args.quant_method() == kQuantMethodSmoothquant;

  if (is_smoothquant_) {
    // Safety check: only w8a8 smoothquant is supported
    if (quant_args.bits() != 8 || !quant_args.activation_dynamic()) {
      LOG(FATAL)
          << "DenseMLP w8a8 mode only supports w8a8 smoothquant quantization. "
          << "Got bits=" << quant_args.bits()
          << ", activation_dynamic=" << quant_args.activation_dynamic();
    }
  }

  // Determine extra args based on quantization mode
  LinearExtraArgs gate_up_proj_extra_args("none", false);
  LinearExtraArgs down_proj_extra_args("none", false);
  if (is_smoothquant_) {
    // For per-token smoothquant, use specific args
    down_proj_extra_args = LinearExtraArgs(hidden_act_, is_gated_);
  }

  // 1. gate + up
  int64_t out_feature = is_gated_ ? intermediate_size_ * 2 : intermediate_size_;
  gate_up_proj_ =
      register_module("gate_up_proj",
                      ColumnParallelLinear(hidden_size,
                                           out_feature,
                                           /*bias=*/has_bias,
                                           /*gather_output=*/false,
                                           quant_args,
                                           process_group_,
                                           options,
                                           gate_up_proj_extra_args));

  act_ =
      register_module("act", Activation(hidden_act_, is_gated_, swiglu_limit_));

  // 2. down
  const auto down_proj_quant_args =
      module_prefix.empty()
          ? quant_args
          : quant_args.for_module(module_prefix + ".down_proj");
  down_proj_ = register_module("down_proj",
                               RowParallelLinear(intermediate_size_,
                                                 hidden_size,
                                                 /*bias=*/has_bias,
                                                 /*input_is_parallelized=*/true,
                                                 enable_result_reduction,
                                                 down_proj_quant_args,
                                                 process_group_,
                                                 options,
                                                 down_proj_extra_args));
}

torch::Tensor DenseMLPImpl::forward(const torch::Tensor& hidden_states) {
  const FlashComm1Context* fc1_ctx = get_current_flash_comm1_context();
  torch::Tensor h = hidden_states;

  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    h = gather_sequence(hidden_states, *fc1_ctx);
  }

  auto gate_up = gate_up_proj_->forward(h);

  if (is_smoothquant_) {
    if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
      return down_proj_->forward(gate_up,
                                 row_parallel_reduce_mode_for_fc1(*fc1_ctx));
    }
    return down_proj_->forward(gate_up);
  }

  torch::Tensor output;
  if (!Platform::is_npu()) {
    const int64_t batch_size = gate_up.sizes()[0];
    output = torch::empty(
        {batch_size, intermediate_size_ / process_group_->world_size()},
        gate_up.options());
  }

  act_->forward(gate_up, output);

  if (fc1_ctx && is_sequence_sharded(*fc1_ctx)) {
    return down_proj_->forward(output,
                               row_parallel_reduce_mode_for_fc1(*fc1_ctx));
  }
  return down_proj_->forward(output);
}

void DenseMLPImpl::load_state_dict(const StateDict& state_dict) {
  gate_up_proj_->load_state_dict(state_dict, {"gate_proj.", "up_proj."});
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix("down_proj."));
}

void DenseMLPImpl::load_state_dict(const StateDict& state_dict,
                                   const std::vector<std::string>& gate_up_name,
                                   const std::string& down_name) {
  if (is_gated_) {
    CHECK_EQ(gate_up_name.size(), 2);
    gate_up_proj_->load_state_dict(state_dict, gate_up_name);
  } else {
    CHECK_EQ(gate_up_name.size(), 1);
    gate_up_proj_->load_state_dict(
        state_dict.get_dict_with_prefix(gate_up_name[0]));
  }
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix(down_name));
}

std::optional<torch::Tensor> DenseMLPImpl::get_fp8_input_scale() const {
  if (gate_up_proj_) {
    return gate_up_proj_->get_input_scale();
  }
  return std::nullopt;
}

}  // namespace layer
}  // namespace xllm
