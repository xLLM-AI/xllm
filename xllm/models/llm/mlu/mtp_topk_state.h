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
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/model/mtp_topk_state.h"
#include "core/layers/mlu/dsa_topk_state.h"

namespace xllm::mlu::model {

// Model-level MLU adapter for sparse-attention state emitted by each MTP
// decoder layer. Runtime sees only the shared row-oriented MtpTopkState API.
class MluMtpTopkState final : public ::xllm::MtpTopkState {
 public:
  using LayerStates = std::vector<std::optional<layer::DsaTopkState>>;

  explicit MluMtpTopkState(LayerStates layer_states)
      : layer_states_(std::move(layer_states)) {
    CHECK(!layer_states_.empty())
        << "MLU MTP top-k state must contain decoder-layer entries.";
    bool found_state = false;
    for (std::optional<layer::DsaTopkState>& state : layer_states_) {
      if (!state.has_value()) {
        continue;
      }
      state = state->flattened();
      if (!found_state) {
        num_rows_ = state->block_tables().size(0);
        device_ = state->block_tables().device();
        found_state = true;
        continue;
      }
      CHECK_EQ(state->block_tables().size(0), num_rows_)
          << "MLU per-layer MTP top-k states must have equal row counts.";
      CHECK(state->block_tables().device() == device_)
          << "MLU per-layer MTP top-k states must use the same device.";
    }
    CHECK(found_state)
        << "MLU MTP top-k state must contain at least one populated layer.";
  }

  const LayerStates& layer_states() const { return layer_states_; }

  int64_t num_rows() const override { return num_rows_; }

  torch::Device device() const override { return device_; }

  MtpTopkStatePtr to(const torch::Device& device) const override {
    LayerStates moved_states;
    moved_states.reserve(layer_states_.size());
    for (const std::optional<layer::DsaTopkState>& state : layer_states_) {
      if (!state.has_value()) {
        moved_states.emplace_back(std::nullopt);
        continue;
      }
      moved_states.emplace_back(state->to(device));
    }
    return std::make_shared<MluMtpTopkState>(std::move(moved_states));
  }

  MtpTopkStatePtr index_select_rows(const torch::Tensor& index) const override {
    LayerStates selected_states;
    selected_states.reserve(layer_states_.size());
    for (const std::optional<layer::DsaTopkState>& state : layer_states_) {
      if (!state.has_value()) {
        selected_states.emplace_back(std::nullopt);
        continue;
      }
      selected_states.emplace_back(layer::DsaTopkState(
          state->block_tables().index_select(/*dim=*/0, index),
          state->context_lens().index_select(/*dim=*/0, index)));
    }
    return std::make_shared<MluMtpTopkState>(std::move(selected_states));
  }

 private:
  LayerStates layer_states_;
  int64_t num_rows_ = 0;
  torch::Device device_{torch::kCPU};
};

}  // namespace xllm::mlu::model
