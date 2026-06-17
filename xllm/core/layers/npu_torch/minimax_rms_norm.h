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

#pragma once

#include <torch/torch.h>

#include <tuple>

#include "core/framework/parallel_state/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

class MiniMaxTensorParallelRMSNormImpl : public torch::nn::Module {
 public:
  MiniMaxTensorParallelRMSNormImpl(int64_t local_dim,
                                   int64_t global_dim,
                                   int64_t replica_factor,
                                   double eps,
                                   ProcessGroup* process_group,
                                   const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& input);

  const torch::Tensor& weight() const { return weight_; }
  double eps() const { return eps_; }
  ProcessGroup* process_group() const { return process_group_; }
  int64_t local_dim() const { return local_dim_; }
  int64_t replica_factor() const { return replica_factor_; }

  void load_state_dict(const StateDict& state_dict);

 private:
  DEFINE_WEIGHT(weight);
  int64_t local_dim_ = 0;
  int64_t global_dim_ = 0;
  int64_t replica_factor_ = 1;
  double eps_ = 1e-6;
  ProcessGroup* process_group_ = nullptr;
};
TORCH_MODULE(MiniMaxTensorParallelRMSNorm);

std::tuple<torch::Tensor, torch::Tensor> forward_qk_rms_norm(
    MiniMaxTensorParallelRMSNorm& q_norm,
    MiniMaxTensorParallelRMSNorm& k_norm,
    const torch::Tensor& query,
    const torch::Tensor& key);

}  // namespace layer
}  // namespace xllm
