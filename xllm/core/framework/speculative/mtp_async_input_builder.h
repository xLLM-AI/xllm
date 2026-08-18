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

namespace xllm {

struct ForwardInput;

namespace mtp_async {

// Applies accepted target state to the fixed [repair, current] draft layout.
// The NPU path uses one fused preparation kernel when possible and otherwise
// falls back to equivalent Torch tensor operations.
void prepare_next_draft_from_accepted_state(
    ForwardInput& draft_input,
    const ForwardInput& block_table_source,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    bool use_chunked_prefill,
    bool rebuild_expanded_decode_metadata,
    int32_t block_size);

// Builds one-row-per-sequence metadata for a later draft step from the
// accepted device base used by draft-0. The caller must order this work after
// draft-0 metadata correction on the same stream.
void prepare_later_draft_from_device_base(
    ForwardInput& draft_input,
    const ForwardInput& block_table_source,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    int32_t position_offset,
    int32_t block_size);

// Corrects an already prepared fixed-shape target verification template from
// the previous target's accepted device state. Draft token columns remain
// placeholders and are filled after their producing draft forwards.
void prepare_target_verify_from_accepted_state(
    ForwardInput& validate_input,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    int32_t block_size);

}  // namespace mtp_async
}  // namespace xllm
