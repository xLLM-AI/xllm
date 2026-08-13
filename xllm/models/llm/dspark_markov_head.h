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

#include <cstdint>
#include <string>

#include "core/framework/state_dict/state_dict.h"

namespace xllm {

// Low-rank Markov head shared by the Qwen3 and DeepSeek-V4 DSpark drafts.
// Both matrices are kept replicated because every sequential proposal step
// consumes a full-vocabulary bias.
class DSparkMarkovHead final {
 public:
  DSparkMarkovHead(const torch::TensorOptions& options, int64_t markov_rank)
      : tensor_options_(options), markov_rank_(markov_rank) {
    CHECK_GT(markov_rank_, 0) << "DSpark requires markov_rank > 0.";
  }

  void load_state_dict(const StateDict& state_dict) {
    torch::Tensor w1 = state_dict.get_tensor("markov_w1.weight");
    torch::Tensor w2 = state_dict.get_tensor("markov_w2.weight");
    if (w1.defined()) {
      markov_w1_ = w1.to(tensor_options_);
    }
    if (w2.defined()) {
      markov_w2_ = w2.to(tensor_options_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(markov_w1_.defined())
        << "Failed to find DSpark " << prefix << "markov_w1.weight.";
    CHECK(markov_w2_.defined())
        << "Failed to find DSpark " << prefix << "markov_w2.weight.";
    CHECK_EQ(markov_w1_.dim(), 2)
        << "DSpark markov_w1 must be two-dimensional.";
    CHECK_EQ(markov_w2_.dim(), 2)
        << "DSpark markov_w2 must be two-dimensional.";
    CHECK_EQ(markov_w1_.size(1), markov_rank_)
        << "DSpark markov_w1 rank mismatch.";
    CHECK_EQ(markov_w2_.size(1), markov_rank_)
        << "DSpark markov_w2 rank mismatch.";
    CHECK_EQ(markov_w1_.size(0), markov_w2_.size(0))
        << "DSpark reduced-vocabulary drafts need draft-to-target remapping, "
           "which is not implemented.";
  }

  torch::Tensor bias(const torch::Tensor& previous_token_ids) const {
    CHECK(markov_w1_.defined() && markov_w2_.defined())
        << "DSpark Markov head weights are not initialized.";
    namespace F = torch::nn::functional;
    torch::Tensor markov_embedding =
        F::embedding(previous_token_ids, markov_w1_);
    return F::linear(markov_embedding, markov_w2_);
  }

 private:
  torch::Tensor markov_w1_;
  torch::Tensor markov_w2_;
  torch::TensorOptions tensor_options_;
  int64_t markov_rank_ = 0;
};

}  // namespace xllm
