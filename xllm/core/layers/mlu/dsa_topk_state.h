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

namespace xllm::layer {

// Native MLU sparse-attention state shared between decoder layers or MTP
// draft steps. Absence is represented by std::optional at the call site so
// the paired tensors cannot be separated.
class DsaTopkState final {
 public:
  DsaTopkState(torch::Tensor block_tables, torch::Tensor context_lens)
      : block_tables_(std::move(block_tables)),
        context_lens_(std::move(context_lens)) {
    CHECK(block_tables_.defined()) << "DSA top-k block tables must be defined.";
    CHECK(context_lens_.defined()) << "DSA top-k context lens must be defined.";
    CHECK(block_tables_.dim() == 2 || block_tables_.dim() == 3)
        << "DSA top-k block tables must be a 2-D or 3-D tensor.";
    CHECK_EQ(context_lens_.dim(), 1)
        << "DSA top-k context lens must be a 1-D tensor.";
    CHECK(block_tables_.scalar_type() == torch::kInt &&
          context_lens_.scalar_type() == torch::kInt)
        << "DSA top-k sparse metadata must use int32.";
    CHECK(block_tables_.device() == context_lens_.device())
        << "DSA top-k sparse metadata must use the same device.";
    CHECK_GT(block_tables_.size(-1), 0)
        << "DSA top-k block tables must have at least one column.";
    const int64_t row_count =
        block_tables_.dim() == 3 ? block_tables_.size(0) * block_tables_.size(1)
                                 : block_tables_.size(0);
    CHECK_EQ(row_count, context_lens_.numel())
        << "DSA top-k block table row count must match context lens.";
  }

  const torch::Tensor& block_tables() const { return block_tables_; }
  const torch::Tensor& context_lens() const { return context_lens_; }

  DsaTopkState flattened() const {
    if (block_tables_.dim() == 2) {
      return *this;
    }
    CHECK(block_tables_.is_contiguous())
        << "MLU MTP DSA top-k block tables must be contiguous before flatten.";
    return DsaTopkState(block_tables_.view({-1, block_tables_.size(-1)}),
                        context_lens_);
  }

  DsaTopkState to(const torch::Device& device) const {
    torch::Tensor block_tables = block_tables_.to(
        block_tables_.options().device(device), /*non_blocking=*/true);
    torch::Tensor context_lens = context_lens_.to(
        context_lens_.options().device(device), /*non_blocking=*/true);
    return DsaTopkState(std::move(block_tables), std::move(context_lens));
  }

 private:
  torch::Tensor block_tables_;
  torch::Tensor context_lens_;
};

}  // namespace xllm::layer
