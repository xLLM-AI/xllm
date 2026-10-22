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

#include <algorithm>

#include "core/runtime/options.h"

namespace xllm::mtp_detail {

inline runtime::Options target_options(const runtime::Options& options) {
  runtime::Options target_options = options;
  target_options.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return target_options;
}

inline runtime::Options draft_options(const runtime::Options& options) {
  runtime::Options draft_options = options;
  // MTP drafts consume text/hidden states through LLMWorkerImpl, including
  // when their target uses a VLM execution contract.
  draft_options.backend("llm")
      .enable_schedule_overlap(false)
      .is_draft_engine(true)
      // The task path also restores the verifier prefix (K+1 rows).
      .num_decoding_tokens(
          options.enable_task_pipeline()
              ? std::max(2, options.num_speculative_tokens() + 1)
              : 1)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(true);
  return draft_options;
}

}  // namespace xllm::mtp_detail
