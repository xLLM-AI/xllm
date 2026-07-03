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

#include "layers/common/attention_metadata.h"

namespace xllm::layer::flashinfer {

void update_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                              const std::string& backend,
                              const AttentionMetadata& attn_meta,
                              torch::ScalarType query_dtype,
                              torch::ScalarType key_dtype,
                              torch::ScalarType output_dtype,
                              int32_t head_dim_qk,
                              int32_t head_dim_vo,
                              int32_t num_qo_heads,
                              int32_t num_kv_heads,
                              bool enable_cuda_graph);

void update_chunked_prefill_plan_info(std::shared_ptr<PlanInfo> plan_info,
                                      const std::string& backend,
                                      const AttentionMetadata& attn_meta,
                                      torch::ScalarType query_dtype,
                                      torch::ScalarType key_dtype,
                                      torch::ScalarType output_dtype,
                                      int32_t head_dim_qk,
                                      int32_t head_dim_vo,
                                      int32_t num_qo_heads,
                                      int32_t num_kv_heads,
                                      int32_t block_size,
                                      int32_t window_size_left,
                                      bool enable_cuda_graph,
                                      bool causal = true,
                                      int32_t max_kv_blocks_per_seq = 0);

// `max_kv_blocks_per_seq`: when non-zero AND `enable_cuda_graph=true`, the
// plan is computed against a synthetic worst-case `paged_kv_indptr_host =
// [0, max, 2*max, ..., bs*max]`. This makes the cached plan_info valid for
// any future runtime block count <= max_kv_blocks_per_seq, which is required
// for CUDA-graph replay correctness: the captured kernel reuses the plan_info
// recorded at warmup time, so a plan computed for a 1-block warmup layout
// would only dispatch attention work for 1 block per sequence, silently
// underreading the KV cache once the sequence crosses a block boundary
// during replay. Computed as
// ceil(max_position_embeddings / block_size) at the call site.
void update_decode_plan_info(std::shared_ptr<PlanInfo> plan_info,
                             const std::string& backend,
                             const AttentionMetadata& attn_meta,
                             torch::ScalarType query_dtype,
                             torch::ScalarType key_dtype,
                             torch::ScalarType output_dtype,
                             int32_t head_dim_qk,
                             int32_t head_dim_vo,
                             int32_t num_qo_heads,
                             int32_t num_kv_heads,
                             int32_t block_size,
                             int32_t window_size_left,
                             bool enable_cuda_graph,
                             bool use_tensor_core,
                             int32_t max_kv_blocks_per_seq = 0);

}  // namespace xllm::layer::flashinfer
