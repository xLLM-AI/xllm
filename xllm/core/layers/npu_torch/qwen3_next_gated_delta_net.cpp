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

#include "qwen3_next_gated_delta_net.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

Qwen3NextGatedDeltaNetImpl::Qwen3NextGatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : Qwen3NextGatedDeltaNetImpl(args,
                                 quant_args,
                                 parallel_args,
                                 options,
                                 /*init_projections=*/true) {}

Qwen3NextGatedDeltaNetImpl::Qwen3NextGatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    bool init_projections)
    : Qwen3GatedDeltaNetBaseImpl(args, quant_args, parallel_args, options) {
  if (init_projections) {
    init_next_projections(args, quant_args, parallel_args, options);
  }
}

void Qwen3NextGatedDeltaNetImpl::init_next_projections(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  // QKVZ projection used by Qwen3-Next linear attention.
  qkvz_proj_ = register_module("in_proj_qkvz",
                               ColumnParallelLinear(args.hidden_size(),
                                                    k_size_ * 2 + v_size_ * 2,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  // BA projection used to derive gating and beta terms.
  ba_proj_ = register_module("in_proj_ba",
                             ColumnParallelLinear(args.hidden_size(),
                                                  num_v_heads_ * 2,
                                                  /*bias=*/false,
                                                  /*gather_output=*/false,
                                                  quant_args,
                                                  parallel_args.tp_group_,
                                                  options));
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3NextGatedDeltaNetImpl::project_decode_inputs(
    const torch::Tensor& hidden_states) {
  auto qkvz = qkvz_proj_->forward(hidden_states);
  auto ba = ba_proj_->forward(hidden_states);
  return {qkvz.view({qkvz.size(0), -1, qkvz.size(-1)}),
          ba.view({ba.size(0), -1, ba.size(-1)})};
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3NextGatedDeltaNetImpl::project_flat_inputs(
    const torch::Tensor& hidden_states) {
  return {qkvz_proj_->forward(hidden_states), ba_proj_->forward(hidden_states)};
}

void Qwen3NextGatedDeltaNetImpl::load_state_dict(const StateDict& state_dict) {
  load_projection_state_dict(state_dict);
  load_common_state_dict(state_dict);
}

void Qwen3NextGatedDeltaNetImpl::load_projection_state_dict(
    const StateDict& state_dict) {
  auto qkvz_state_dict = state_dict.get_dict_with_prefix("in_proj_qkvz.");
  if (qkvz_state_dict.size() > 0 && !qkvz_proj_->is_weight_loaded()) {
    qkvz_proj_->load_state_dict(qkvz_state_dict);
  }

  auto ba_state_dict = state_dict.get_dict_with_prefix("in_proj_ba.");
  if (ba_state_dict.size() > 0 && !ba_proj_->is_weight_loaded()) {
    ba_proj_->load_state_dict(ba_state_dict);
  }
}

void Qwen3NextGatedDeltaNetImpl::verify_loaded_weights(
    const std::string& prefix) const {
  verify_projection_weights(prefix);
  verify_common_loaded_weights(prefix);
}

void Qwen3NextGatedDeltaNetImpl::verify_projection_weights(
    const std::string& prefix) const {
  CHECK(qkvz_proj_ && qkvz_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkvz.weight";
  CHECK(ba_proj_ && ba_proj_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_ba.weight";
}

}  // namespace layer
}  // namespace xllm
