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

#include <cstdint>

#include "core/framework/model/mtp_topk_state.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/embedding_cache.h"

namespace xllm {
class LlmForwardInput;
class VlmForwardInput;
}  // namespace xllm

namespace xllm::mtp_detail {
void check_mtp_decode_states(
    const std::vector<EmbeddingCache::DecodeState>& states,
    const std::vector<std::string>& request_ids,
    const torch::Tensor& token_ids_host,
    bool allow_overlap_fake_token);

MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params);

// How the expected KV length is derived for a decode step.
struct ExpectedKvPolicy {
  enum class Kind : uint8_t {
    // expected_kv_len = position + 1.
    kPositionCoupled,
    // mrope-section targets pass the KV length through untouched.
    kDecoupledPassThrough,
    // The chunked-verify clamp on top of the position-coupled derivation.
    kChunkedVerifyClamped,
  };

  Kind kind;
  int32_t verify_lens = 0;
  int32_t logical_block_size = 0;
};

// Derives the expected-KV-length policy for a speculative decode step. MTP and
// the block drafts both rewrite the decode context, so they share one closed
// derivation: the chunked-verify clamp wins over the mrope pass-through.
ExpectedKvPolicy derive_expected_kv_policy(bool clamped_verify,
                                           bool positions_decoupled,
                                           int32_t verify_lens,
                                           int64_t logical_block_size);

// Shared decode-context rewrite for the MTP/DFlash decode-step paths:
// resolves fake tokens against the cached target states and keeps the
// position/kv-length invariant.
template <typename Input>
void update_decode_step_input(
    Input& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    bool enable_cache_correction,
    const ExpectedKvPolicy& kv_policy);
extern template void update_decode_step_input<LlmForwardInput>(
    LlmForwardInput&,
    const std::vector<EmbeddingCache::DecodeState>&,
    bool,
    const ExpectedKvPolicy&);
extern template void update_decode_step_input<VlmForwardInput>(
    VlmForwardInput&,
    const std::vector<EmbeddingCache::DecodeState>&,
    bool,
    const ExpectedKvPolicy&);

}  // namespace xllm::mtp_detail
