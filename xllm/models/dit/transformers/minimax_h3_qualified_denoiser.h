/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "models/dit/transformers/minimax_h3_denoiser.h"
#include "models/dit/transformers/minimax_h3_qualified_weights.h"
#include "models/dit/transformers/minimax_h3_tp_uaa_transformer.h"
#include "models/dit/utils/minimax_h3_qualified_execution_context.h"
#include "models/dit/utils/minimax_h3_qualified_input.h"

namespace xllm {

struct H3QualifiedStepInput {
  torch::Tensor video_rows;
  torch::Tensor audio_rows;
  torch::Tensor unique_timesteps;
  torch::Tensor inverse_indices;
  int64_t step_index = 0;
};

struct H3QualifiedPrediction {
  torch::Tensor video_velocity;
  torch::Tensor audio_velocity;
  int64_t executed_blocks = 0;
  bool cache_hit = false;
  float relative_l1 = std::numeric_limits<float>::infinity();
};

// Callbacks consume borrowed tensors synchronously. Retaining a rows tensor
// without cloning observes the pipeline's subsequent in-place scheduler update.
struct H3QualifiedDiagnostics {
  MiniMaxH3LayerObserver layer_observer;
  std::function<void(int64_t, const H3QualifiedRequest&)> before_step;
  std::function<
      void(int64_t, const H3QualifiedRequest&, const MiniMaxH3DenoiserOutput&)>
      after_prediction;
  std::function<void(int64_t, const H3QualifiedRequest&)> after_update;
  std::function<void(int64_t, const torch::Tensor&, const torch::Tensor&)>
      after_scheduler;
  bool retain_preparation = false;
  bool retain_model_boundary = false;
  bool verify_intermediates = false;
};

// Reference mode owns one block and its CPU source; production owns all TP2
// resident shards. Both predict raw velocity; the pipeline owns the schedulers.
class H3QualifiedDenoiser final {
 public:
  explicit H3QualifiedDenoiser(const torch::TensorOptions& options,
                               H3QualifiedDiagnostics diagnostics = {})
      : streaming_(MiniMaxH3StreamingDenoiser(options)),
        diagnostics_(std::move(diagnostics)) {}

  H3QualifiedDenoiser(ProcessGroup* tp_group,
                      ProcessGroup* sp_group,
                      const torch::TensorOptions& options,
                      H3QualifiedDiagnostics diagnostics = {})
      : resident_(MiniMaxH3TpUaaResidentDenoiser(tp_group, sp_group, options)),
        diagnostics_(std::move(diagnostics)) {}

  void load(std::unique_ptr<H3QualifiedWeightSource> source) {
    CHECK(source != nullptr);
    CHECK(!weights_loaded_) << "Qualified weights already loaded";
    if (resident_) {
      resident_->load_weights(*source);
      // Every resident target has independent device storage; release the CPU
      // source and temporary canonical views after this call returns.
    } else {
      StateDict fixed = source->fixed_weights();
      streaming_->load_fixed_weights(fixed);
      weight_source_ = std::move(source);
    }
    weights_loaded_ = true;
  }

  H3QualifiedPrediction predict_step(
      const H3QualifiedRequest& request,
      const H3QualifiedStepInput& input,
      H3QualifiedExecutionContext* execution = nullptr) {
    CHECK(weights_loaded_) << "Qualified weights are not loaded";
    CHECK_GE(input.step_index, 0);
    CHECK_LT(input.step_index, 49);
    if (execution != nullptr && execution->cache != nullptr && !resident_) {
      throw std::invalid_argument(
          "H3 CacheDiT requires the resident qualified denoiser");
    }
    const int64_t before_blocks = resident_ ? resident_->block_forwards() : 0;
    SimilarityResidualCacheDecision cache_decision;
    MiniMaxH3DenoiserOutput result;
    if (resident_) {
      result = resident_->forward(request.layout,
                                  input.video_rows.squeeze(0),
                                  input.audio_rows.squeeze(0),
                                  input.unique_timesteps,
                                  input.inverse_indices,
                                  input.step_index,
                                  diagnostics_.layer_observer,
                                  diagnostics_.retain_preparation,
                                  diagnostics_.retain_model_boundary,
                                  diagnostics_.verify_intermediates,
                                  execution,
                                  &cache_decision);
    } else {
      result = streaming_->forward(*weight_source_,
                                   request.layout,
                                   input.video_rows.squeeze(0),
                                   input.audio_rows.squeeze(0),
                                   input.unique_timesteps,
                                   input.inverse_indices,
                                   input.step_index,
                                   diagnostics_.layer_observer,
                                   diagnostics_.retain_preparation,
                                   diagnostics_.retain_model_boundary,
                                   diagnostics_.verify_intermediates);
    }
    if (resident_) {
      result.executed_layers = resident_->block_forwards() - before_blocks;
    }
    if (diagnostics_.after_prediction) {
      diagnostics_.after_prediction(input.step_index, request, result);
    }
    return {.video_velocity = result.video_velocity.unsqueeze(0),
            .audio_velocity = result.audio_velocity.unsqueeze(0),
            .executed_blocks = resident_
                                   ? resident_->block_forwards() - before_blocks
                                   : result.executed_layers,
            .cache_hit = cache_decision.hit,
            .relative_l1 = cache_decision.relative_l1};
  }

  const H3QualifiedDiagnostics& diagnostics() const { return diagnostics_; }

 private:
  std::unique_ptr<H3QualifiedWeightSource> weight_source_;
  MiniMaxH3StreamingDenoiser streaming_{nullptr};
  MiniMaxH3TpUaaResidentDenoiser resident_{nullptr};
  bool weights_loaded_ = false;
  H3QualifiedDiagnostics diagnostics_;
};

}  // namespace xllm
