/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/types.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace xllm {

class MtpTopkState;
using MtpTopkStatePtr = std::shared_ptr<const MtpTopkState>;

// Backend-neutral state carried between MTP draft steps. Representations share
// a row axis; tensor-only consumers request as_tensor() without backend casts.
class MtpTopkState {
 public:
  virtual ~MtpTopkState() = default;

  // Share the caller's tensor storage, including final graph inputs.
  static MtpTopkStatePtr from_tensor(torch::Tensor topk_indices);

  // A layered payload is not interchangeable with a single index tensor.
  virtual std::optional<torch::Tensor> as_tensor() const = 0;
  virtual int64_t num_rows() const = 0;
  virtual torch::Device device() const = 0;
  virtual MtpTopkStatePtr to(const torch::Device& device) const = 0;
  virtual MtpTopkStatePtr index_select_rows(
      const torch::Tensor& index) const = 0;
};

class TensorMtpTopkState final : public MtpTopkState {
 public:
  explicit TensorMtpTopkState(torch::Tensor topk_indices)
      : topk_indices_(std::move(topk_indices)) {
    CHECK(topk_indices_.defined()) << "MTP top-k indices must be defined.";
    CHECK_GE(topk_indices_.dim(), 1)
        << "MTP top-k indices must have at least one dimension.";
  }

  const torch::Tensor& topk_indices() const { return topk_indices_; }

  std::optional<torch::Tensor> as_tensor() const override {
    return topk_indices_;
  }

  int64_t num_rows() const override { return topk_indices_.size(0); }

  torch::Device device() const override { return topk_indices_.device(); }

  MtpTopkStatePtr to(const torch::Device& device) const override {
    return from_tensor(topk_indices_.to(topk_indices_.options().device(device),
                                        /*non_blocking=*/true));
  }

  MtpTopkStatePtr index_select_rows(const torch::Tensor& index) const override {
    return from_tensor(topk_indices_.index_select(/*dim=*/0, index));
  }

 private:
  torch::Tensor topk_indices_;
};

inline MtpTopkStatePtr MtpTopkState::from_tensor(torch::Tensor topk_indices) {
  return std::make_shared<TensorMtpTopkState>(std::move(topk_indices));
}

}  // namespace xllm
