/* Copyright 2025-2026 The xLLM Authors.

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

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class LLMWorkerImpl : public WorkerImpl {
 public:
  enum class ForwardSyncPolicy : int8_t {
    LEGACY = 0,
    NO_SYNC,
  };

  LLMWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~LLMWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_no_sync(const ForwardInput& input);
  std::optional<ForwardOutput> execute_no_sync_on_stream(
      const ForwardInput& input,
      Stream& compute_stream,
      bool record_ready_event = true);

  folly::SemiFuture<std::optional<ForwardOutput>> step_async_no_sync(
      const ForwardInput& input);

  std::optional<ForwardOutput> step_internal(
      const ForwardInput& input,
      ForwardSyncPolicy sync_policy = ForwardSyncPolicy::LEGACY,
      bool record_ready_event = true);

 protected:
  std::optional<ForwardOutput> step_for_schedule_overlap(
      const ForwardInput& input) override;
  ForwardInput update_input_by_last_step_output_for_schedule_overlap(
      ForwardInput& input) override;

 public:
#if defined(USE_NPU)
  layer::NpuLmHead get_npu_lm_head() { return model_->get_npu_lm_head(); };

  void set_npu_lm_head(layer::NpuLmHead& head) {
    model_->set_npu_lm_head(head);
  };

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return model_->get_npu_word_embedding();
  };

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) {
    model_->set_npu_word_embedding(embedding);
  };

#endif
  layer::LmHead get_lm_head() { return model_->get_lm_head(); };

  void set_lm_head(layer::LmHead& head) { model_->set_lm_head(head); };

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  };

  void set_word_embedding(layer::WordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  };

 protected:
  std::unique_ptr<BeamSearcher> beam_searcher_;
};

}  // namespace xllm
