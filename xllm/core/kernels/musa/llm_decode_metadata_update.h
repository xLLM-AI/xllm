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

#include <cuda_runtime.h>

#include <cstdint>

namespace xllm::kernel::cuda {

using LlmDecodeMetadataUpdateStream = cudaStream_t;

struct LlmDecodeMetadataUpdateParams {
  const int32_t* src_tokens;
  const int32_t* src_positions;
  const int32_t* src_new_cache_slots;
  const int32_t* src_kv_seq_lens;
  const int32_t* src_paged_kv_indptr;
  const int32_t* src_paged_kv_indices;
  const int32_t* src_paged_kv_last_page_len;
  int32_t* dst_tokens;
  int32_t* dst_positions;
  int32_t* dst_new_cache_slots;
  int32_t* dst_kv_seq_lens;
  int32_t* dst_kv_seq_lens_delta;
  int32_t* dst_paged_kv_indptr;
  int32_t* dst_paged_kv_indices;
  int32_t* dst_paged_kv_last_page_len;
  int64_t actual_num_tokens;
  int64_t padded_num_tokens;
  int64_t actual_batch_size;
  int64_t actual_indices_size;
  int64_t max_indices_size_for_graph_capacity;
};

void update_llm_decode_metadata(const LlmDecodeMetadataUpdateParams& params,
                                LlmDecodeMetadataUpdateStream stream);

// CPU-direct graph replay path: H2D paged-KV / kv_seq_lens from host mirrors,
// D2D tokens / positions / new_cache_slots from device inputs.
struct LlmDecodeMetadataHostUpdateParams {
  const int32_t* src_tokens;
  const int32_t* src_positions;
  const int32_t* src_new_cache_slots;
  const int32_t* host_kv_seq_lens;
  const int32_t* host_paged_kv_indptr;
  const int32_t* host_paged_kv_indices;
  const int32_t* host_paged_kv_last_page_len;
  int32_t* dst_tokens;
  int32_t* dst_positions;
  int32_t* dst_new_cache_slots;
  int32_t* dst_kv_seq_lens;
  int32_t* dst_kv_seq_lens_delta;
  int32_t* dst_paged_kv_indptr;
  int32_t* dst_paged_kv_indices;
  int32_t* dst_paged_kv_last_page_len;
  int64_t actual_num_tokens;
  int64_t padded_num_tokens;
  int64_t actual_batch_size;
  int64_t actual_indices_size;
};

void update_llm_decode_metadata_from_host(
    const LlmDecodeMetadataHostUpdateParams& params,
    LlmDecodeMetadataUpdateStream stream);

}  // namespace xllm::kernel::cuda
