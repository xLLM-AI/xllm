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
#include <vector>

namespace xllm::mtp_async {

enum class TargetSpecVerifyMode {
  GENERIC,
  CAUSAL_CHUNKED_PREFILL,
  QWEN3_5_EXPANDED_VERIFY,
  DEEPSEEK_V32_EXPANDED_VERIFY,
};

// Keep target verification policy closed over model types with validated
// layouts. Unknown models retain the generic path.
TargetSpecVerifyMode classify_target_spec_verify_mode(
    std::string_view model_type);

// Shared allocation/launch width for target verification block tables. The
// extra entry covers the speculative token that can cross a block boundary.
int64_t speculative_verify_block_table_capacity(int64_t max_position_embeddings,
                                                int64_t block_size);

enum class CombinedDraftExecutionPath {
  UNSUPPORTED,
  QWEN3_5_PAGED_ATTENTION,
  GLM_MOE_DSA_SPARSE_ATTENTION,
};

CombinedDraftExecutionPath classify_combined_draft_execution_path(
    std::string_view model_type);

bool supports_combined_draft_configuration(
    CombinedDraftExecutionPath execution_path,
    std::string_view npu_backend,
    int32_t dp_size);

// Materialize proposer-owned token columns into the row-major target verify
// input. Graph replay normally performs this copy internally; eager fallback
// must use the same logical tokens before invoking the model.
torch::Tensor materialize_speculative_verify_tokens(
    const torch::Tensor& verify_tokens,
    const std::vector<torch::Tensor>& draft_token_sources);

// Recover the KV length at the first target-verify token. Chunked-prefill
// stores one post-verify length per sequence, while decode stores one length
// per expanded verification row.
torch::Tensor extract_target_base_kv_seq_lens(
    const torch::Tensor& validate_kv_seq_lens,
    int64_t batch_size,
    int64_t num_validate_tokens,
    bool use_chunked_prefill);

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

struct AcceptedTokenMetadata {
  torch::Tensor accepted_lengths;
  torch::Tensor last_tokens;
  torch::Tensor base_positions;
  torch::Tensor base_kv_seq_lens;
};

AcceptedTokenMetadata build_accepted_token_metadata(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens);

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

}  // namespace xllm::mtp_async
