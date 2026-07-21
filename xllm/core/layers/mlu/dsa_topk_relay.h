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

#include <optional>
#include <utility>

#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/layers/mlu/dsa_topk_state.h"

namespace xllm::layer {

// Per-attention transfer prepared by a relay or the MTP bridge. Factory
// methods expose only valid reuse/capture combinations, and output is
// published as one DsaTopkState rather than through independent pointers.
class DsaTopkTransfer final {
 public:
  static DsaTopkTransfer capture_output() {
    return DsaTopkTransfer(/*input=*/std::nullopt,
                           /*captures_output=*/true);
  }

  static DsaTopkTransfer reuse(const DsaTopkState& input) {
    return DsaTopkTransfer(input, /*captures_output=*/false);
  }

  static DsaTopkTransfer reuse_and_capture(const DsaTopkState& input) {
    return DsaTopkTransfer(input, /*captures_output=*/true);
  }

  static DsaTopkTransfer prepare_mtp_step(
      const std::optional<DsaTopkState>& state,
      const torch::Device& device) {
    if (!state.has_value()) {
      return capture_output();
    }
    return reuse_and_capture(state->to(device));
  }

  const DsaTopkState* input() const {
    return input_.has_value() ? &input_.value() : nullptr;
  }

  bool captures_output() const { return captures_output_; }

  void publish_output(DsaTopkState output) {
    CHECK(captures_output_) << "DSA top-k transfer does not capture output.";
    CHECK(!output_.has_value()) << "DSA top-k output was already published.";
    output_ = std::move(output);
  }

  const DsaTopkState* output() const {
    return output_.has_value() ? &output_.value() : nullptr;
  }

  std::optional<DsaTopkState> mtp_output_state() const { return output_; }

 private:
  DsaTopkTransfer(std::optional<DsaTopkState> input, bool captures_output)
      : input_(std::move(input)), captures_output_(captures_output) {}

  std::optional<DsaTopkState> input_;
  bool captures_output_ = false;
  std::optional<DsaTopkState> output_;
};

// Owns the forward-scoped producer-to-consumer state. The model only resets
// and passes this relay; layer-role interpretation and transfer validation stay
// below the model boundary.
class DsaTopkRelay final {
 public:
  void reset() { state_.reset(); }

  std::optional<DsaTopkTransfer> prepare_layer(
      const DsaTopkShareDecision& decision) const {
    CHECK(!(decision.reuse_topk && decision.output_topk))
        << "A DSA layer cannot reuse and publish top-k simultaneously.";
    if (decision.reuse_topk) {
      CHECK(state_.has_value())
          << "DSA top-k reuse requires a previously published state.";
      return DsaTopkTransfer::reuse(state_.value());
    }
    if (decision.output_topk) {
      return DsaTopkTransfer::capture_output();
    }
    return std::nullopt;
  }

  void finish_layer(const DsaTopkShareDecision& decision,
                    const DsaTopkTransfer& transfer) {
    if (!decision.output_topk) {
      return;
    }
    const DsaTopkState* output = transfer.output();
    CHECK(output != nullptr) << "DSA top-k producer did not publish its state.";
    state_ = *output;
  }

 private:
  std::optional<DsaTopkState> state_;
};

}  // namespace xllm::layer
