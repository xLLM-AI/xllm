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

#include "minimax_m2_tensor_parallel_rms_norm.h"

#include <glog/logging.h>

#include <vector>

namespace xllm {
namespace layer {

MiniMaxM2TensorParallelRMSNormImpl::MiniMaxM2TensorParallelRMSNormImpl(
    int64_t local_dim,
    int64_t global_dim,
    int64_t replica_factor,
    double eps,
    ProcessGroup* process_group,
    const torch::TensorOptions& options)
    : local_dim_(local_dim),
      global_dim_(global_dim),
      replica_factor_(replica_factor),
      eps_(eps),
      process_group_(process_group) {
  CHECK(process_group_ != nullptr)
      << "MiniMaxM2TensorParallelRMSNorm requires a TP process group.";
  CHECK_GT(replica_factor_, 0);
  CHECK_EQ(process_group_->world_size() % replica_factor_, 0)
      << "tp world size " << process_group_->world_size()
      << " must be divisible by replica factor " << replica_factor_;

  const int64_t effective_world_size =
      process_group_->world_size() / replica_factor_;
  CHECK_GT(effective_world_size, 0);
  CHECK_EQ(global_dim_ % effective_world_size, 0)
      << "global_dim " << global_dim_
      << " must be divisible by effective world size " << effective_world_size;
  CHECK_EQ(local_dim_, global_dim_ / effective_world_size)
      << "unexpected local shard size for MiniMax-M2 TP RMSNorm.";

  weight_ = register_parameter(
      "weight", torch::empty({local_dim_}, options), /*requires_grad=*/false);
}

torch::Tensor MiniMaxM2TensorParallelRMSNormImpl::forward(
    const torch::Tensor& input) {
  std::vector<int64_t> org_shape = input.sizes().vec();
  torch::Tensor input_2d = input.reshape({-1, local_dim_});
  torch::Tensor input_fp32 = input_2d.to(torch::kFloat32);
  torch::Tensor sq_sum =
      (input_fp32 * input_fp32).sum(/*dim=*/-1, /*keepdim=*/true);
  if (process_group_->world_size() > 1) {
    sq_sum = parallel_state::reduce(sq_sum, process_group_);
  }

  const float inv_global_dim =
      1.0f / static_cast<float>(global_dim_ * replica_factor_);
  torch::Tensor inv_rms = torch::rsqrt(sq_sum * inv_global_dim + eps_);
  torch::Tensor normalized = (input_fp32 * inv_rms).to(input_2d.scalar_type());
  torch::Tensor output = normalized * weight_.view({1, local_dim_});
  return output.view(org_shape);
}

void MiniMaxM2TensorParallelRMSNormImpl::load_state_dict(
    const StateDict& state_dict) {
  if (weight_is_loaded_) {
    return;
  }

  const int64_t rank = process_group_->rank() / replica_factor_;
  const int64_t world_size = process_group_->world_size() / replica_factor_;
  torch::Tensor tensor = state_dict.get_sharded_tensor(
      "weight", /*dim=*/0, /*rank=*/rank, /*world_size=*/world_size);
  if (!tensor.defined()) {
    return;
  }

  CHECK_EQ(weight_.sizes(), tensor.sizes())
      << "weight size mismatch for " << state_dict.prefix() << "weight.";
  weight_.copy_(tensor);
  weight_is_loaded_ = true;
}

std::tuple<torch::Tensor, torch::Tensor> forward_minimax_m2_qk_rms_norm(
    MiniMaxM2TensorParallelRMSNorm& q_norm,
    MiniMaxM2TensorParallelRMSNorm& k_norm,
    const torch::Tensor& query,
    const torch::Tensor& key) {
  CHECK(q_norm->process_group() != nullptr &&
        q_norm->process_group() == k_norm->process_group())
      << "MiniMax-M2 q/k RMSNorm requires q_norm and k_norm to share the same "
         "process group.";

  std::vector<int64_t> q_org_shape = query.sizes().vec();
  std::vector<int64_t> k_org_shape = key.sizes().vec();
  const torch::Tensor q_2d =
      query.reshape({-1, q_norm->local_dim()}).to(torch::kFloat32);
  const torch::Tensor k_2d =
      key.reshape({-1, k_norm->local_dim()}).to(torch::kFloat32);

  torch::Tensor q_var = (q_2d * q_2d).mean(/*dim=*/-1, /*keepdim=*/true);
  torch::Tensor k_var = (k_2d * k_2d).mean(/*dim=*/-1, /*keepdim=*/true);

  if (q_norm->process_group()->world_size() > 1) {
    torch::Tensor qk_var = torch::cat({q_var, k_var}, /*dim=*/-1);
    qk_var = parallel_state::reduce(qk_var, q_norm->process_group());
    std::vector<torch::Tensor> chunks = qk_var.chunk(/*chunks=*/2, /*dim=*/-1);
    q_var = chunks[0];
    k_var = chunks[1];
  }

  const int64_t q_effective_ws =
      q_norm->process_group()->world_size() / q_norm->replica_factor();
  const int64_t k_effective_ws =
      k_norm->process_group()->world_size() / k_norm->replica_factor();
  q_var = q_var / static_cast<double>(q_effective_ws);
  k_var = k_var / static_cast<double>(k_effective_ws);

  torch::Tensor q_out =
      (q_2d * torch::rsqrt(q_var + q_norm->eps())).to(query.scalar_type());
  torch::Tensor k_out =
      (k_2d * torch::rsqrt(k_var + k_norm->eps())).to(key.scalar_type());
  q_out = (q_out * q_norm->weight()).view(q_org_shape);
  k_out = (k_out * k_norm->weight()).view(k_org_shape);
  return std::make_tuple(std::move(q_out), std::move(k_out));
}

}  // namespace layer
}  // namespace xllm
