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

#include "qwen3_5_fused_moe.h"

#include <glog/logging.h>

#include "framework/parallel_state/parallel_state.h"
#include "framework/state_dict/utils.h"
#include "layers/common/moe_weight_loader_helper.h"

namespace xllm {
namespace layer {
namespace {
torch::Tensor get_tensor_with_weight_suffix(const StateDict& state_dict,
                                            const std::string& tensor_name) {
  torch::Tensor tensor = state_dict.get_tensor(tensor_name);
  if (!tensor.defined()) {
    tensor = state_dict.get_tensor(tensor_name + ".weight");
  }
  return tensor;
}

bool load_fused_gate_up_fallback(const StateDict& state_dict,
                                 int64_t rank,
                                 int64_t world_size,
                                 int64_t start_expert_id,
                                 int64_t num_experts_per_rank,
                                 torch::Tensor& w13) {
  torch::Tensor fused_gate_up =
      get_tensor_with_weight_suffix(state_dict, "gate_up_proj");
  if (!fused_gate_up.defined()) {
    return false;
  }

  torch::Tensor gate_up_slice = moe_weight::slice_experts(
      moe_weight::shard_gate_up(fused_gate_up, rank, world_size),
      start_expert_id,
      num_experts_per_rank);
  CHECK_EQ(w13.sizes(), gate_up_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.gate_up_proj";
  w13.copy_(gate_up_slice);
  return true;
}

bool load_fused_down_fallback(const StateDict& state_dict,
                              int64_t rank,
                              int64_t world_size,
                              int64_t start_expert_id,
                              int64_t num_experts_per_rank,
                              torch::Tensor& w2) {
  torch::Tensor fused_down =
      get_tensor_with_weight_suffix(state_dict, "down_proj");
  if (!fused_down.defined()) {
    return false;
  }

  torch::Tensor down_slice = moe_weight::slice_experts(
      moe_weight::shard_last_dim(fused_down, rank, world_size, w2.size(-1)),
      start_expert_id,
      num_experts_per_rank);
  CHECK_EQ(w2.sizes(), down_slice.sizes())
      << "weight size mismatch for " << state_dict.prefix()
      << "experts.down_proj";
  w2.copy_(down_slice);
  return true;
}
}  // namespace

Qwen3_5FusedMoEImpl::Qwen3_5FusedMoEImpl(const ModelArgs& model_args,
                                         const FusedMoEArgs& moe_args,
                                         const QuantArgs& quant_args,
                                         const ParallelArgs& parallel_args,
                                         const torch::TensorOptions& options)
    : FusedMoEImpl(model_args, moe_args, quant_args, parallel_args, options) {
  if (n_shared_experts_ > 0) {
    shared_expert_gate_ = register_module(
        "shared_expert_gate",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, 1).bias(false)));
    shared_expert_gate_->weight.set_data(
        shared_expert_gate_->weight.to(options));
  }
}

void Qwen3_5FusedMoEImpl::load_experts(const StateDict& state_dict) {
  FusedMoEImpl::load_experts(state_dict);

  if (is_smoothquant_) {
    const bool gate_up_loaded =
        w13_is_loaded_ && w13_scale_is_loaded_ && input_smooth_is_loaded_;
    if (!gate_up_loaded) {
      CHECK(moe_weight::try_load_gate_up_sq(state_dict,
                                            tp_pg_->rank(),
                                            tp_pg_->world_size(),
                                            start_expert_id_,
                                            num_experts_per_rank_,
                                            w13_,
                                            w13_scale_,
                                            input_smooth_,
                                            w13_is_loaded_,
                                            w13_scale_is_loaded_,
                                            input_smooth_is_loaded_))
          << "failed to load gate_up smoothquant weights for "
          << state_dict.prefix();
    }

    const bool down_loaded =
        w2_is_loaded_ && w2_scale_is_loaded_ && act_smooth_is_loaded_;
    if (!down_loaded) {
      CHECK(moe_weight::try_load_down_sq(state_dict,
                                         tp_pg_->rank(),
                                         tp_pg_->world_size(),
                                         start_expert_id_,
                                         num_experts_per_rank_,
                                         w2_,
                                         w2_scale_,
                                         act_smooth_,
                                         w2_is_loaded_,
                                         w2_scale_is_loaded_,
                                         act_smooth_is_loaded_))
          << "failed to load down smoothquant weights for "
          << state_dict.prefix();
    }
  } else {
    if (!w13_is_loaded_) {
      w13_is_loaded_ = load_fused_gate_up_fallback(state_dict,
                                                   tp_pg_->rank(),
                                                   tp_pg_->world_size(),
                                                   start_expert_id_,
                                                   num_experts_per_rank_,
                                                   w13_);
    }

    if (!w2_is_loaded_) {
      w2_is_loaded_ = load_fused_down_fallback(state_dict,
                                               tp_pg_->rank(),
                                               tp_pg_->world_size(),
                                               start_expert_id_,
                                               num_experts_per_rank_,
                                               w2_);
    }
  }
}

void Qwen3_5FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  if (n_shared_experts_ > 0) {
    shared_experts_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_expert."));
    auto weight = state_dict.get_tensor("shared_expert_gate.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(shared_expert_gate_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      shared_expert_gate_->weight.data().copy_(weight);
    }
  }
  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

void Qwen3_5FusedMoEImpl::final_comm_allreduce(
    torch::Tensor& final_hidden_states,
    const torch::Tensor& hidden_states,
    torch::Tensor& shared_expert_output) {
  auto current_stream = device_.current_stream();
  routed_stream_->wait_stream(*current_stream);
  {
    torch::StreamGuard stream_guard = routed_stream_->set_stream_guard();
    if (tp_pg_->world_size() > 1) {
      final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
    }
    if (parallel_args_.ep_size() > 1) {
      final_hidden_states = parallel_state::reduce(
          final_hidden_states, parallel_args_.moe_ep_group_);
    }
  }

  if (n_shared_experts_ > 0) {
    shared_stream_->wait_stream(*current_stream);
    torch::StreamGuard stream_guard = shared_stream_->set_stream_guard();
    shared_expert_output = shared_experts_(hidden_states);
    if (shared_expert_gate_) {
      auto gate = torch::sigmoid(shared_expert_gate_->forward(hidden_states));
      shared_expert_output = gate * shared_expert_output;
    }
    shared_expert_output =
        shared_expert_output.reshape({-1, shared_expert_output.size(-1)});
  }

  // join for parallelization
  current_stream->wait_stream(*routed_stream_);
  if (n_shared_experts_ > 0) {
    current_stream->wait_stream(*shared_stream_);
    final_hidden_states += shared_expert_output;
  }
}

}  // namespace layer
}  // namespace xllm
