/* Copyright 2026 The xLLM Authors.

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

#include <torch/torch.h>

#include <cstdint>
#include <string_view>

namespace xllm::mtp_async {

enum class CombinedDraftExecutionPath {
  UNSUPPORTED,
  QWEN3_5_PAGED_ATTENTION,
  GLM5_DSA,
};

CombinedDraftExecutionPath classify_combined_draft_execution_path(
    std::string_view model_type);

bool can_use_combined_decode(bool enable_schedule_overlap,
                             bool execution_path_supported);

// Device-resident state derived from target verification. base_positions and
// base_kv_seq_lens point at the logical position immediately after the accepted
// prefix and are therefore the base of the next draft iteration.
struct AcceptedState {
  torch::Tensor accepted_lengths;
  torch::Tensor all_draft_accepted;
  torch::Tensor last_tokens;
  torch::Tensor previous_tokens;
  torch::Tensor last_embeddings;
  torch::Tensor previous_embeddings;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
};

AcceptedState build_accepted_state(const torch::Tensor& accepted_tokens,
                                   const torch::Tensor& accepted_embeddings,
                                   const torch::Tensor& embedding_placeholder,
                                   const torch::Tensor& base_positions,
                                   const torch::Tensor& base_kv_seq_lens);

torch::Tensor make_row_positions(const AcceptedState& state,
                                 const torch::Tensor& offsets);

torch::Tensor make_kv_seq_lens(const AcceptedState& state,
                               const torch::Tensor& offsets,
                               bool use_chunked_prefill);

// The repair row is useful only when all draft tokens were accepted. On a
// rejection it is redirected to a future scratch position so it cannot
// overwrite valid draft KV state.
torch::Tensor make_repair_cache_positions(const AcceptedState& state);

torch::Tensor map_positions_to_cache_slots(const torch::Tensor& block_tables,
                                           const torch::Tensor& positions,
                                           int32_t block_size);

torch::Tensor select_rows(const torch::Tensor& values,
                          const torch::Tensor& selected_rows);

}  // namespace xllm::mtp_async
