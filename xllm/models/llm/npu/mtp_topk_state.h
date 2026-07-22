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

#include <glog/logging.h>
#include <torch/torch.h>

#include <utility>

#include "core/framework/model/mtp_topk_state.h"

namespace xllm::npu::model {

// Adapts the NPU decoder's native top-k indices to the shared MTP draft-step
// state contract. The decoder continues to consume and produce the same tensor.
class NpuMtpTopkState final : public ::xllm::MtpTopkState {
 public:
  explicit NpuMtpTopkState(torch::Tensor topk_indices)
      : topk_indices_(std::move(topk_indices)) {
    CHECK(topk_indices_.defined())
        << "NPU MTP DSA top-k indices must be defined.";
    CHECK_GE(topk_indices_.dim(), 1)
        << "NPU MTP DSA top-k indices must have at least one dimension.";
  }

  const torch::Tensor& topk_indices() const { return topk_indices_; }

  int64_t num_rows() const override { return topk_indices_.size(0); }

  torch::Device device() const override { return topk_indices_.device(); }

  MtpTopkStatePtr to(const torch::Device& device) const override {
    return std::make_shared<NpuMtpTopkState>(topk_indices_.to(
        topk_indices_.options().device(device), /*non_blocking=*/true));
  }

  MtpTopkStatePtr index_select_rows(const torch::Tensor& index) const override {
    return std::make_shared<NpuMtpTopkState>(
        topk_indices_.index_select(/*dim=*/0, index));
  }

 private:
  torch::Tensor topk_indices_;
};

}  // namespace xllm::npu::model
