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

#include "qwen3_5_gated_delta_net.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

Qwen3_5GatedDeltaNetImpl::Qwen3_5GatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : Qwen3NextGatedDeltaNetImpl(args,
                                 quant_args,
                                 parallel_args,
                                 options,
                                 /*init_projections=*/false) {
  in_proj_qkv_ = register_module("in_proj_qkv",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      k_size_ * 2 + v_size_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      parallel_args.tp_group_,
                                                      options));
  in_proj_z_ = register_module("in_proj_z",
                               ColumnParallelLinear(args.hidden_size(),
                                                    v_size_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_b_ = register_module("in_proj_b",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_a_ = register_module("in_proj_a",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_qkvz_from_split_activations(
    const torch::Tensor& qkv,
    const torch::Tensor& z) const {
  CHECK_EQ(qkv.dim(), 3) << "Expected qkv activation to be 3D, got "
                         << qkv.sizes();
  CHECK_EQ(z.dim(), 3) << "Expected z activation to be 3D, got " << z.sizes();
  CHECK_EQ(qkv.size(0), z.size(0)) << "qkv/z batch size mismatch.";
  CHECK_EQ(qkv.size(1), z.size(1)) << "qkv/z sequence size mismatch.";
  CHECK_EQ(qkv.size(2), (2 * k_size_ + v_size_) / tp_size_)
      << "Unexpected qkv hidden size for Qwen3.5.";
  CHECK_EQ(z.size(2), v_size_ / tp_size_)
      << "Unexpected z hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = qkv.size(0);
  const int64_t seqlen = qkv.size(1);
  const int64_t local_k_heads = num_k_heads_ / tp_size_;
  const int64_t local_v_heads = num_v_heads_ / tp_size_;
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;

  auto qkv_split = torch::split(
      qkv, {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_}, 2);
  auto q = qkv_split[0].view({bs, seqlen, local_k_heads, head_k_dim_});
  auto k = qkv_split[1].view({bs, seqlen, local_k_heads, head_k_dim_});
  auto v = qkv_split[2].view({bs, seqlen, local_v_heads, head_v_dim_});
  auto z_view = z.view({bs, seqlen, local_v_heads, head_v_dim_});

  v = v.view({bs, seqlen, local_k_heads, num_v_heads_per_k * head_v_dim_});
  z_view =
      z_view.view({bs, seqlen, local_k_heads, num_v_heads_per_k * head_v_dim_});

  return torch::cat({q, k, v, z_view}, -1).view({bs, seqlen, -1}).contiguous();
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::merge_ba_from_split_activations(
    const torch::Tensor& b,
    const torch::Tensor& a) const {
  CHECK_EQ(b.dim(), 3) << "Expected b activation to be 3D, got " << b.sizes();
  CHECK_EQ(a.dim(), 3) << "Expected a activation to be 3D, got " << a.sizes();
  CHECK_EQ(b.size(0), a.size(0)) << "b/a batch size mismatch.";
  CHECK_EQ(b.size(1), a.size(1)) << "b/a sequence size mismatch.";
  CHECK_EQ(b.size(2), num_v_heads_ / tp_size_)
      << "Unexpected b hidden size for Qwen3.5.";
  CHECK_EQ(a.size(2), num_v_heads_ / tp_size_)
      << "Unexpected a hidden size for Qwen3.5.";
  CHECK_GT(num_k_heads_, 0) << "linear_num_key_heads must be positive.";
  CHECK_EQ(num_v_heads_ % num_k_heads_, 0)
      << "linear_num_value_heads must be divisible by linear_num_key_heads.";

  const int64_t bs = b.size(0);
  const int64_t seqlen = b.size(1);
  const int64_t local_k_heads = num_k_heads_ / tp_size_;
  const int64_t num_v_heads_per_k = num_v_heads_ / num_k_heads_;

  auto b_view = b.view({bs, seqlen, local_k_heads, num_v_heads_per_k});
  auto a_view = a.view({bs, seqlen, local_k_heads, num_v_heads_per_k});
  return torch::cat({b_view, a_view}, -1).view({bs, seqlen, -1}).contiguous();
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_decode_inputs(
    const torch::Tensor& hidden_states) {
  const auto reshape_projection = [](const torch::Tensor& projection) {
    return projection.view({projection.size(0), -1, projection.size(-1)});
  };
  auto qkv = reshape_projection(in_proj_qkv_->forward(hidden_states));
  auto z_proj = reshape_projection(in_proj_z_->forward(hidden_states));
  auto b_proj = reshape_projection(in_proj_b_->forward(hidden_states));
  auto a_proj = reshape_projection(in_proj_a_->forward(hidden_states));
  return {merge_qkvz_from_split_activations(qkv, z_proj),
          merge_ba_from_split_activations(b_proj, a_proj)};
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3_5GatedDeltaNetImpl::project_flat_inputs(
    const torch::Tensor& hidden_states) {
  auto qkv = in_proj_qkv_->forward(hidden_states).unsqueeze(0);
  auto z_proj = in_proj_z_->forward(hidden_states).unsqueeze(0);
  auto b_proj = in_proj_b_->forward(hidden_states).unsqueeze(0);
  auto a_proj = in_proj_a_->forward(hidden_states).unsqueeze(0);
  auto qkvz = merge_qkvz_from_split_activations(qkv, z_proj);
  auto ba = merge_ba_from_split_activations(b_proj, a_proj);
  return {qkvz.view({hidden_states.size(0), qkvz.size(-1)}).contiguous(),
          ba.view({hidden_states.size(0), ba.size(-1)}).contiguous()};
}

void Qwen3_5GatedDeltaNetImpl::load_projection_state_dict(
    const StateDict& state_dict) {
  auto in_proj_qkv_state_dict = state_dict.get_dict_with_prefix("in_proj_qkv.");
  if (in_proj_qkv_state_dict.size() > 0 && !in_proj_qkv_->is_weight_loaded()) {
    in_proj_qkv_->load_state_dict(
        in_proj_qkv_state_dict,
        /*shard_tensor_count=*/3,
        /*shard_sizes=*/
        {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_});
  }

  auto in_proj_z_state_dict = state_dict.get_dict_with_prefix("in_proj_z.");
  if (in_proj_z_state_dict.size() > 0 && !in_proj_z_->is_weight_loaded()) {
    in_proj_z_->load_state_dict(in_proj_z_state_dict);
  }

  auto in_proj_b_state_dict = state_dict.get_dict_with_prefix("in_proj_b.");
  if (in_proj_b_state_dict.size() > 0 && !in_proj_b_->is_weight_loaded()) {
    in_proj_b_->load_state_dict(in_proj_b_state_dict);
  }

  auto in_proj_a_state_dict = state_dict.get_dict_with_prefix("in_proj_a.");
  if (in_proj_a_state_dict.size() > 0 && !in_proj_a_->is_weight_loaded()) {
    in_proj_a_->load_state_dict(in_proj_a_state_dict);
  }
}

void Qwen3_5GatedDeltaNetImpl::verify_projection_weights(
    const std::string& prefix) const {
  CHECK(in_proj_qkv_ && in_proj_qkv_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkv.weight";
  CHECK(in_proj_z_ && in_proj_z_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_z.weight";
  CHECK(in_proj_b_ && in_proj_b_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_b.weight";
  CHECK(in_proj_a_ && in_proj_a_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_a.weight";
}

}  // namespace layer
}  // namespace xllm
