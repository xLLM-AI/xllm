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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
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
  AuxHiddenCapture(const ModelArgs& model_args,
                   const torch::TensorOptions& options,
                   int64_t max_tokens_per_batch) {
    if (model_args.layers_to_capture().empty()) {
      return;
    }
    layers_to_capture_ = model_args.layers_to_capture();
    const int64_t num_captured =
        static_cast<int64_t>(layers_to_capture_.size());
    const int64_t aux_dim = model_args.hidden_size() * num_captured;
    buffer_ = torch::empty({max_tokens_per_batch, aux_dim}, options);
  }

  // Pass residual when the caller keeps `h` and residual as separate tensors
  // (intralayer add-norm); pass std::nullopt when `h` already carries the sum.
  void capture_layer(int32_t layer_idx,
                     const torch::Tensor& h,
                     const std::optional<torch::Tensor>& residual) {
    const auto it = std::find(
        layers_to_capture_.begin(), layers_to_capture_.end(), layer_idx);
    if (it == layers_to_capture_.end()) {
      return;
    }
    const int64_t num_tokens = h.size(0);
    const int64_t hidden_size = h.size(-1);
    const int64_t slot_idx =
        static_cast<int64_t>(std::distance(layers_to_capture_.begin(), it));
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
  }

  bool enabled() const { return !layers_to_capture_.empty(); }

  // Typical k <= 5, so a cache-line linear scan beats any hashed lookup.
  bool should_capture(int32_t layer_idx) const {
    return std::find(layers_to_capture_.begin(),
                     layers_to_capture_.end(),
                     layer_idx) != layers_to_capture_.end();
  }

  ModelOutput finalize(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& residual = std::nullopt) const {
    ModelOutput output(hidden_states, residual);
    if (enabled()) {
      output.aux_hidden_states = buffer_.slice(0, 0, hidden_states.size(0));
    }
    return output;
  }

 private:
  // Layer ids to capture; non-empty iff capture is enabled.
  std::vector<int32_t> layers_to_capture_;
  torch::Tensor buffer_;
};

}  // namespace xllm
