/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {
namespace moe_weight {

inline torch::Tensor slice_experts(const torch::Tensor& tensor,
                                   int64_t start_expert_id,
                                   int64_t num_experts_per_rank) {
  return tensor
      .slice(/*dim=*/0, start_expert_id, start_expert_id + num_experts_per_rank)
      .contiguous();
}

inline torch::Tensor shard_gate_up(const torch::Tensor& tensor,
                                   int64_t rank,
                                   int64_t world_size) {
  if (world_size <= 1) {
    return tensor;
  }

  CHECK_GE(tensor.dim(), 2)
      << "gate_up_proj must have at least 2 dims, got " << tensor.sizes();
  CHECK_EQ(tensor.size(1) % 2, 0)
      << "gate_up_proj dim1 must be even, got " << tensor.size(1);
  const int64_t full_intermediate = tensor.size(1) / 2;
  CHECK_EQ(full_intermediate % world_size, 0)
      << "gate_up_proj intermediate dim is not divisible by world_size";
  const int64_t local_intermediate = full_intermediate / world_size;

  torch::Tensor gate_full = tensor.slice(/*dim=*/1, 0, full_intermediate);
  torch::Tensor up_full =
      tensor.slice(/*dim=*/1, full_intermediate, full_intermediate * 2);
  torch::Tensor gate_shard = gate_full.slice(
      /*dim=*/1, rank * local_intermediate, (rank + 1) * local_intermediate);
  torch::Tensor up_shard = up_full.slice(
      /*dim=*/1, rank * local_intermediate, (rank + 1) * local_intermediate);
  return torch::cat({gate_shard, up_shard}, /*dim=*/1);
}

inline torch::Tensor shard_last_dim(const torch::Tensor& tensor,
                                    int64_t rank,
                                    int64_t world_size,
                                    int64_t local_size) {
  if (world_size <= 1) {
    return tensor;
  }

  CHECK_GE(tensor.dim(), 1) << "tensor must have at least 1 dim";
  CHECK_EQ(tensor.size(-1), local_size * world_size)
      << "last dim is not divisible by world_size, tensor shape="
      << tensor.sizes() << ", local_size=" << local_size
      << ", world_size=" << world_size;
  return tensor.slice(
      tensor.dim() - 1, rank * local_size, (rank + 1) * local_size);
}

inline void copy_expand_vec(const torch::Tensor& src, torch::Tensor& dst) {
  CHECK_EQ(dst.size(-1), src.numel())
      << "smooth size mismatch, src=" << src.sizes() << ", dst=" << dst.sizes();
  torch::Tensor reshaped = src.reshape({1, -1}).expand(dst.sizes());
  dst.copy_(reshaped);
}

inline bool can_shard_gate_up(const torch::Tensor& tensor, int64_t world_size) {
  if (world_size <= 1) {
    return tensor.defined();
  }
  return tensor.defined() && tensor.dim() >= 2 && tensor.size(1) % 2 == 0 &&
         (tensor.size(1) / 2) % world_size == 0;
}

inline bool can_shard_last_dim(const torch::Tensor& tensor,
                               int64_t world_size,
                               int64_t local_size) {
  if (world_size <= 1) {
    return tensor.defined();
  }
  return tensor.defined() && tensor.dim() >= 1 &&
         tensor.size(-1) == local_size * world_size;
}

inline bool can_slice_experts(const torch::Tensor& tensor,
                              int64_t start_expert_id,
                              int64_t num_experts_per_rank) {
  return tensor.defined() && tensor.dim() >= 1 && start_expert_id >= 0 &&
         num_experts_per_rank >= 0 &&
         start_expert_id + num_experts_per_rank <= tensor.size(0);
}

inline bool can_expand_vec_to(const torch::Tensor& src,
                              const torch::Tensor& dst) {
  return src.defined() && dst.defined() && src.dim() >= 1 && dst.dim() >= 1 &&
         dst.size(-1) == src.numel();
}

inline bool try_load_gate_up_sq(const StateDict& state_dict,
                                int64_t rank,
                                int64_t world_size,
                                int64_t start_expert_id,
                                int64_t num_experts_per_rank,
                                torch::Tensor& w13,
                                torch::Tensor& w13_scale,
                                torch::Tensor& input_smooth,
                                bool& w13_is_loaded,
                                bool& w13_scale_is_loaded,
                                bool& input_smooth_is_loaded) {
  torch::Tensor fused_qweight = state_dict.get_tensor("gate_up_proj.qweight");
  torch::Tensor fused_scale =
      state_dict.get_tensor("gate_up_proj.per_channel_scale");
  torch::Tensor fused_smooth = state_dict.get_tensor("gate_up_proj.smooth");
  if (!fused_qweight.defined() || !fused_scale.defined() ||
      !fused_smooth.defined()) {
    return false;
  }
  if (!can_shard_gate_up(fused_qweight, world_size) ||
      !can_shard_gate_up(fused_scale, world_size) ||
      !can_slice_experts(
          fused_qweight, start_expert_id, num_experts_per_rank) ||
      !can_slice_experts(fused_scale, start_expert_id, num_experts_per_rank) ||
      !can_expand_vec_to(fused_smooth, input_smooth)) {
    return false;
  }

  torch::Tensor qweight_shard =
      slice_experts(shard_gate_up(fused_qweight, rank, world_size),
                    start_expert_id,
                    num_experts_per_rank);
  torch::Tensor scale_shard =
      slice_experts(shard_gate_up(fused_scale, rank, world_size),
                    start_expert_id,
                    num_experts_per_rank);
  if (w13.sizes() != qweight_shard.sizes() ||
      w13_scale.sizes() != scale_shard.sizes()) {
    return false;
  }

  w13.copy_(qweight_shard);
  w13_is_loaded = true;
  w13_scale.copy_(scale_shard);
  w13_scale_is_loaded = true;

  copy_expand_vec(fused_smooth, input_smooth);
  input_smooth_is_loaded = true;
  return true;
}

inline bool try_load_down_sq(const StateDict& state_dict,
                             int64_t rank,
                             int64_t world_size,
                             int64_t start_expert_id,
                             int64_t num_experts_per_rank,
                             torch::Tensor& w2,
                             torch::Tensor& w2_scale,
                             torch::Tensor& act_smooth,
                             bool& w2_is_loaded,
                             bool& w2_scale_is_loaded,
                             bool& act_smooth_is_loaded) {
  torch::Tensor fused_qweight = state_dict.get_tensor("down_proj.qweight");
  torch::Tensor fused_scale =
      state_dict.get_tensor("down_proj.per_channel_scale");
  torch::Tensor fused_smooth = state_dict.get_tensor("down_proj.smooth");
  if (!fused_qweight.defined() || !fused_scale.defined() ||
      !fused_smooth.defined()) {
    return false;
  }
  if (!can_shard_last_dim(fused_qweight, world_size, w2.size(-1)) ||
      !can_slice_experts(
          fused_qweight, start_expert_id, num_experts_per_rank) ||
      !can_slice_experts(fused_scale, start_expert_id, num_experts_per_rank) ||
      !can_shard_last_dim(fused_smooth, world_size, act_smooth.size(-1))) {
    return false;
  }

  torch::Tensor qweight_shard = slice_experts(
      shard_last_dim(fused_qweight, rank, world_size, w2.size(-1)),
      start_expert_id,
      num_experts_per_rank);
  torch::Tensor scale = fused_scale;
  if (fused_scale.dim() == w2_scale.dim() &&
      can_shard_last_dim(fused_scale, world_size, w2_scale.size(-1))) {
    scale = shard_last_dim(fused_scale, rank, world_size, w2_scale.size(-1));
  }
  torch::Tensor scale_shard =
      slice_experts(scale, start_expert_id, num_experts_per_rank);
  torch::Tensor smooth_shard =
      shard_last_dim(fused_smooth, rank, world_size, act_smooth.size(-1));
  if (w2.sizes() != qweight_shard.sizes() ||
      w2_scale.sizes() != scale_shard.sizes() ||
      !can_expand_vec_to(smooth_shard, act_smooth)) {
    return false;
  }

  w2.copy_(qweight_shard);
  w2_is_loaded = true;
  w2_scale.copy_(scale_shard);
  w2_scale_is_loaded = true;
  copy_expand_vec(smooth_shard, act_smooth);
  act_smooth_is_loaded = true;
  return true;
}

}  // namespace moe_weight
}  // namespace layer
}  // namespace xllm
