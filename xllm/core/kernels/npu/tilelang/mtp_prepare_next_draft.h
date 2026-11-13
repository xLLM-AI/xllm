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

#include <cstdint>
#include <optional>

namespace xllm::kernel::npu {

struct MtpPrepareNextDraftOutput {
  torch::Tensor token_ids;
  torch::Tensor embeddings;
  torch::Tensor positions;
  torch::Tensor kv_seq_lens;
  torch::Tensor cache_slots;
  torch::Tensor shared_cache_slots;
};

// Owns the fixed-address output for one MTP metadata variant. The owner
// must outlive every graph replay that consumes output.
struct MtpPrepareNextDraftWorkspace {
  MtpPrepareNextDraftOutput output;
};

// Full [B,K+1,H] hidden returns B KV lengths. Compact [2B,H] hidden
// contains previous/current rows and directly returns 2B KV lengths.
std::optional<MtpPrepareNextDraftOutput> try_mtp_prepare_next_draft(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    const torch::Tensor& block_tables,
    int64_t block_size,
    MtpPrepareNextDraftWorkspace* reusable_workspace = nullptr,
    const torch::Tensor& shared_block_tables = {});

}  // namespace xllm::kernel::npu
