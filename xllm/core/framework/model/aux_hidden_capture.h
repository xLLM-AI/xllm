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
#include <unordered_map>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"

namespace xllm {

// Buffers the residual stream of selected layers into a
// [tokens, hidden * num_captured] tensor for a spec draft (Eagle3,
// DFlash/DSpark) to consume. A non-empty layers_to_capture is the sole capture
// signal.
class AuxHiddenCapture final {
 public:
  void init(const ModelArgs& model_args,
            const torch::TensorOptions& options,
            int64_t max_tokens_per_batch) {
    if (model_args.layers_to_capture().empty()) {
      return;
    }
    const auto& layers = model_args.layers_to_capture();
    num_captured_ = static_cast<int64_t>(layers.size());
    slot_by_layer_.reserve(layers.size());
    for (int32_t slot = 0; slot < static_cast<int32_t>(layers.size()); ++slot) {
      slot_by_layer_.emplace(layers[slot], slot);
    }
    const int64_t aux_dim = model_args.hidden_size() * num_captured_;
    buffer_ = torch::empty({max_tokens_per_batch, aux_dim}, options);
  }

  // Pass residual when the caller keeps `h` and residual as separate tensors
  // (intralayer add-norm); pass std::nullopt when `h` already carries the sum.
  void capture_layer(int32_t layer_idx,
                     const torch::Tensor& h,
                     const std::optional<torch::Tensor>& residual) {
    const auto it = slot_by_layer_.find(layer_idx);
    if (it == slot_by_layer_.end()) {
      return;
    }
    const int64_t num_tokens = h.size(0);
    const int64_t hidden_size = h.size(-1);
    CHECK_LE(num_tokens, buffer_.size(0))
        << "Auxiliary hidden capture exceeds its configured token capacity.";
    const int64_t slot_idx = it->second;
    // add_out fuses the residual add into the preallocated slice, avoiding a
    // fresh [tokens, hidden] sum tensor per captured layer.
    torch::Tensor slot =
        buffer_.slice(0, 0, num_tokens)
            .slice(1, slot_idx * hidden_size, (slot_idx + 1) * hidden_size);
    torch::Tensor h_2d = h.reshape({num_tokens, hidden_size});
    if (residual.has_value()) {
      torch::add_out(
          slot, h_2d, residual.value().reshape({num_tokens, hidden_size}));
    } else {
      slot.copy_(h_2d);
    }
    capture_idx_++;
  }

  // Call before each forward's layer loop so capture_layer() refills from
  // column 0.
  void reset_capture_index() { capture_idx_ = 0; }

  bool enabled() const { return num_captured_ > 0; }

  bool should_capture(int32_t layer_idx) const {
    return slot_by_layer_.count(layer_idx) > 0;
  }

  ModelOutput finalize(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& residual = std::nullopt) const {
    ModelOutput output(hidden_states, residual);
    if (!enabled()) {
      return output;
    }
    CHECK_EQ(capture_idx_, num_captured_)
        << "Captured aux hidden layer count mismatch.";
    output.aux_hidden_states = buffer_.slice(0, 0, hidden_states.size(0));
    return output;
  }

 private:
  std::unordered_map<int32_t, int32_t> slot_by_layer_;
  int64_t num_captured_ = 0;
  torch::Tensor buffer_;
  int64_t capture_idx_ = 0;
};

}  // namespace xllm
