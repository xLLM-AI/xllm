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

#include "mlu_ops_api.h"

namespace xllm::kernel::mlu {

void fused_compress_single_kv(
    const torch::Tensor& kv,
    const torch::Tensor& score,
    const torch::Tensor& position,
    const torch::Tensor& ape,
    const torch::Tensor& gamma,
    const torch::Tensor& sin,
    const torch::Tensor& cos,
    const std::optional<torch::Tensor>& hadamard_matrix,
    const torch::Tensor& slot_mapping,
    torch::Tensor& kv_cache,
    const std::optional<torch::Tensor>& kv_cache_scale,
    double eps,
    bool overlap,
    torch::Tensor& state_cache,
    const torch::Tensor& state_bt,
    int64_t state_width,
    int64_t state_block_size,
    const torch::Tensor& cu_query_len,
    int64_t K) {
  tmo::torch_api::fused_compress_single_kv(kv,
                                           score,
                                           position,
                                           ape,
                                           gamma,
                                           sin,
                                           cos,
                                           hadamard_matrix,
                                           slot_mapping,
                                           kv_cache,
                                           kv_cache_scale,
                                           eps,
                                           overlap,
                                           state_cache,
                                           state_bt,
                                           state_width,
                                           state_block_size,
                                           cu_query_len,
                                           K);
}

void fused_compress_multi_kv(const torch::Tensor& kv,
                             const torch::Tensor& score,
                             torch::Tensor& state_cache,
                             const torch::Tensor& state_block_table,
                             const torch::Tensor& cu_seqlens,
                             const torch::Tensor& positions,
                             const torch::Tensor& ape,
                             int64_t max_seqlen,
                             bool overlap,
                             torch::Tensor& compressed_kv) {
  tmo::torch_api::fused_compress_multi_kv(kv,
                                          score,
                                          state_cache,
                                          state_block_table,
                                          cu_seqlens,
                                          positions,
                                          ape,
                                          max_seqlen,
                                          overlap,
                                          compressed_kv);
}

void update_compressor_states(torch::Tensor& kv_state,
                              torch::Tensor& score_state,
                              const torch::Tensor& accept_tokens,
                              const torch::Tensor& batch_to_kv_state,
                              const torch::Tensor& positions,
                              const torch::Tensor& cu_query_len,
                              const bool overlap,
                              const int64_t K) {
  tmo::torch_api::update_compressor_states(kv_state,
                                           score_state,
                                           accept_tokens,
                                           batch_to_kv_state,
                                           positions,
                                           cu_query_len,
                                           overlap,
                                           K);
}
}  // namespace xllm::kernel::mlu
