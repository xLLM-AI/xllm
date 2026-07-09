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

#include <cstdint>
#include <optional>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/fused_moe.h"

namespace xllm {
namespace layer {

class DeepseekV4SparseMoEBlockTestPeer;

class DeepseekV4SparseMoEBlockImpl final : public torch::nn::Module {
 public:
  DeepseekV4SparseMoEBlockImpl() = default;
  DeepseekV4SparseMoEBlockImpl(const ModelArgs& model_args,
                               const QuantArgs& quant_args,
                               const ParallelArgs& parallel_args,
                               const torch::TensorOptions& options,
                               bool use_hash);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  FusedMoEImpl::RouteInfo prep_route(
      torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);
  torch::Tensor forward_selected(const torch::Tensor& hidden_states,
                                 const torch::Tensor& topk_weights,
                                 const torch::Tensor& topk_ids,
                                 const ModelInputParams& input_params);

 private:
  bool need_gather() const;
  ProcessGroup* routed_pg() const;
  FusedMoEImpl::RouteInfo make_route(const torch::Tensor& topk_weights,
                                     const torch::Tensor& topk_ids,
                                     int64_t hidden_rows) const;
  std::vector<int32_t> get_row_dp_tokens(
      int64_t hidden_rows,
      const ModelInputParams& input_params) const;

  ParallelArgs parallel_args_;
  bool enable_deep_ep_ = false;
  FusedMoE moe_{nullptr};

  friend class DeepseekV4SparseMoEBlockTestPeer;
};

TORCH_MODULE(DeepseekV4SparseMoEBlock);

}  // namespace layer
}  // namespace xllm
