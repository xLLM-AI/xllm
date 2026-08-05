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

#include "runtime/dspark_worker_impl.h"

#include <glog/logging.h>

#include <utility>

#include "common/metrics.h"
#include "util/timer.h"

namespace xllm {

DSparkWorkerImpl::DSparkWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options) {}

DSparkWorkerImpl::DraftBlock DSparkWorkerImpl::run_decode_draft(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  Timer timer;

  // Same input build as DFlash, but sample_from_anchor()==true makes the query
  // block N-wide and every position predicts a draft token.
  ForwardInput query_input;
  prepare_query_inputs(input, query_input);
  prepare_validate_inputs(input, validate_input);

  CHECK(input.token_ids_host.defined())
      << "DSpark requires host token ids for the anchor.";
  const int32_t batch_size = input.input_params.meta.num_sequences;
  CHECK_GE(input.token_ids_host.numel(), batch_size)
      << "DSpark anchor token count must cover the decode batch.";
  torch::Tensor anchor_token_ids =
      input.token_ids_host.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size)
          .to(draft_impl_->device(), torch::kLong);
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();

  LLMWorkerImpl::BlockDraftExecutionOutput draft_output =
      draft_impl_->execute_block_draft_no_sync_on_stream(query_input,
                                                         anchor_token_ids,
                                                         input.sampling_params,
                                                         num_speculative_tokens,
                                                         *prepare_stream_,
                                                         *compute_stream_);

  DraftBlock draft_block;
  draft_block.token_ids = std::move(draft_output.token_ids);
  draft_block.probs = std::move(draft_output.probs);
  draft_block.draft_retained_input = std::move(draft_output.retained_input);

  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_block;
}

}  // namespace xllm
