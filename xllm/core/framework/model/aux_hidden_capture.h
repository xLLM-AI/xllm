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

#include <torch/types.h>

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
  // EAGLE-3 low/mid/high default capture layers (0-based post-layer indices),
  // used when the target config omits the capture list.
  static std::vector<int32_t> legacy_default_capture_layer_ids(
      int32_t num_layers) {
    return {1, num_layers / 2 - 1, num_layers - 4};
  }

  // Number of aux hidden-state chunks the target produces and the draft
  // consumes. An omitted capture list falls back to the EAGLE-3
  // low/mid/high default (three layers), matching the speculators None->3
  // fallback and legacy_default_capture_layer_ids.
  static int64_t num_aux_layers(const ModelArgs& model_args) {
    const auto& capture_ids = model_args.layers_to_capture();
    return capture_ids.empty()
               ? static_cast<int64_t>(
                     legacy_default_capture_layer_ids(model_args.num_layers())
                         .size())
               : static_cast<int64_t>(capture_ids.size());
  }
  // Width of the concatenated aux hidden-state tensor the target produces and
  // the draft consumes.
  static int64_t aux_hidden_dim(const ModelArgs& model_args) {
    return num_aux_layers(model_args) * model_args.hidden_size();
  }

  AuxHiddenCapture(const ModelArgs& model_args,
                   const torch::TensorOptions& options,
                   int64_t max_tokens_per_batch) {
    if (model_args.layers_to_capture().empty()) {
      return;
    }
    layers_to_capture_ = model_args.layers_to_capture();
    buffer_ = torch::empty({max_tokens_per_batch, aux_hidden_dim(model_args)},
                           options);
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

  // Only the sharded rows are written before CP restores global order; never
  // slice with the global row count.
  torch::Tensor sharded_aux_hidden_states(int64_t num_tokens) const {
    CHECK_LE(num_tokens, buffer_.size(0));
    return buffer_.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens);
  }

  // Typical k <= 5, so a cache-line linear scan beats any hashed lookup.
  bool should_capture(int32_t layer_idx) const {
    return std::find(layers_to_capture_.begin(),
                     layers_to_capture_.end(),
                     layer_idx) != layers_to_capture_.end();
  }

  // `aux_hidden_states` carries CP-gathered global-order rows when set.
  ModelOutput finalize(
      const torch::Tensor& hidden_states,
      const std::optional<torch::Tensor>& residual = std::nullopt,
      const std::optional<torch::Tensor>& aux_hidden_states =
          std::nullopt) const {
    ModelOutput output(hidden_states, residual);
    if (enabled()) {
      if (aux_hidden_states.has_value()) {
        CHECK_EQ(aux_hidden_states->size(0), hidden_states.size(0))
            << "CP-restored aux rows must match the hidden state rows";
        output.aux_hidden_states = *aux_hidden_states;
      } else {
        output.aux_hidden_states =
            sharded_aux_hidden_states(hidden_states.size(0));
      }
    }
    return output;
  }

 private:
  // Layer ids to capture; non-empty iff capture is enabled.
  std::vector<int32_t> layers_to_capture_;
  torch::Tensor buffer_;
};

}  // namespace xllm
