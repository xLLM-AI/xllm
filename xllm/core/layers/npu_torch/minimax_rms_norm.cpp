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

#include "minimax_rms_norm.h"

namespace xllm {
namespace layer {

MiniMaxTensorParallelRMSNormImpl::MiniMaxTensorParallelRMSNormImpl(
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
      << "MiniMaxTensorParallelRMSNorm requires tp process group";
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
      << "unexpected local shard size for TP RMSNorm";

  weight_ = register_parameter(
      "weight", torch::empty({local_dim_}, options), /*requires_grad=*/false);
}

torch::Tensor MiniMaxTensorParallelRMSNormImpl::forward(
    const torch::Tensor& input) {
  auto org_shape = input.sizes().vec();
  auto input_2d = input.reshape({-1, local_dim_});
  auto input_fp32 = input_2d.to(torch::kFloat32);
  auto sq_sum = (input_fp32 * input_fp32).sum(/*dim=*/-1, /*keepdim=*/true);
  if (process_group_->world_size() > 1) {
    sq_sum = parallel_state::reduce(sq_sum, process_group_);
  }

  const float inv_global_dim =
      1.0f / static_cast<float>(global_dim_ * replica_factor_);
  auto inv_rms = torch::rsqrt(sq_sum * inv_global_dim + eps_);
  auto normalized = (input_fp32 * inv_rms).to(input_2d.scalar_type());
  auto output = normalized * weight_.view({1, local_dim_});
  return output.view(org_shape);
}

void MiniMaxTensorParallelRMSNormImpl::load_state_dict(
    const StateDict& state_dict) {
  if (weight_is_loaded_) {
    return;
  }

  const int64_t rank = process_group_->rank() / replica_factor_;
  const int64_t world_size = process_group_->world_size() / replica_factor_;
  auto tensor = state_dict.get_sharded_tensor("weight", 0, rank, world_size);
  if (!tensor.defined()) {
    return;
  }

  CHECK_EQ(weight_.sizes(), tensor.sizes())
      << "weight size mismatch for " << state_dict.prefix() << "weight";
  weight_.copy_(tensor);
  weight_is_loaded_ = true;
}

std::tuple<torch::Tensor, torch::Tensor> forward_qk_rms_norm(
    MiniMaxTensorParallelRMSNorm& q_norm,
    MiniMaxTensorParallelRMSNorm& k_norm,
    const torch::Tensor& query,
    const torch::Tensor& key) {
  CHECK(q_norm->process_group() != nullptr &&
        q_norm->process_group() == k_norm->process_group())
      << "forward_qk requires q_norm and k_norm to share the same process "
         "group";

  auto q_org_shape = query.sizes().vec();
  auto k_org_shape = key.sizes().vec();
  const auto q_2d =
      query.reshape({-1, q_norm->local_dim()}).to(torch::kFloat32);
  const auto k_2d = key.reshape({-1, k_norm->local_dim()}).to(torch::kFloat32);

  auto q_var = (q_2d * q_2d).mean(/*dim=*/-1, /*keepdim=*/true);
  auto k_var = (k_2d * k_2d).mean(/*dim=*/-1, /*keepdim=*/true);

  if (q_norm->process_group()->world_size() > 1) {
    auto qk_var = torch::cat({q_var, k_var}, /*dim=*/-1);
    qk_var = parallel_state::reduce(qk_var, q_norm->process_group());
    auto chunks = qk_var.chunk(2, /*dim=*/-1);
    q_var = chunks[0];
    k_var = chunks[1];
  }

  const int64_t q_effective_ws =
      q_norm->process_group()->world_size() / q_norm->replica_factor();
  const int64_t k_effective_ws =
      k_norm->process_group()->world_size() / k_norm->replica_factor();
  q_var = q_var / static_cast<double>(q_effective_ws);
  k_var = k_var / static_cast<double>(k_effective_ws);

  auto q_out =
      (q_2d * torch::rsqrt(q_var + q_norm->eps())).to(query.scalar_type());
  auto k_out =
      (k_2d * torch::rsqrt(k_var + k_norm->eps())).to(key.scalar_type());
  q_out = (q_out * q_norm->weight()).view(q_org_shape);
  k_out = (k_out * k_norm->weight()).view(k_org_shape);
  return std::make_tuple(std::move(q_out), std::move(k_out));
}

}  // namespace layer
}  // namespace xllm
