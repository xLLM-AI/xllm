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
    int32_t block_size);

}  // namespace mtp_async
}  // namespace xllm
